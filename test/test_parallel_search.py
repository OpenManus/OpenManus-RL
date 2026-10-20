"""GAIA caller tests using the real MCP client against a deterministic HTTP peer."""

import argparse
import asyncio
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import openmanus_rl.environments.env_package.tool_use.manager as tool_use_manager
import openmanus_rl.environments.env_package.tool_use.projection as tool_projection
from openmanus_rl.environments.env_package.tool_use.envs import ToolUseEnvs
from openmanus_rl.tools.parallel_search import tool as parallel
from openmanus_rl.tools.search_defaults import default_search_tool


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    for key in ("GOOGLE_API_KEY", "GOOGLE_CX", "PARALLEL_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    # A developer's saved .env must not affect the anonymous-path tests.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)


@pytest.fixture
def peer(monkeypatch):
    requests = []
    behavior = {"mode": "ok"}

    async def handle(request):
        data = json.loads(request.content)
        requests.append((request, data))
        if "id" not in data:
            return httpx.Response(202)
        if data["method"] == "initialize":
            result = {
                "protocolVersion": data["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-peer", "version": "1"},
            }
        elif data["method"] == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object", "properties": {}}}
                    for name in ("web_search", "web_fetch")
                ]
            }
        else:
            assert data["method"] == "tools/call"
            mode = behavior["mode"]
            if mode == "timeout":
                await asyncio.sleep(5)
            if mode in ("401", "429"):
                return httpx.Response(
                    int(mode),
                    text="Permission denied" if mode == "401" else "Quota exceeded",
                )
            if mode == "rpc_error":
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": data["id"],
                        "error": {"code": -32602, "message": "Invalid arguments"},
                    },
                )
            if mode == "malformed":
                return httpx.Response(
                    200, text="{broken", headers={"content-type": "application/json"}
                )
            output = {
                "results": [
                    {
                        "url": "https://example.org/source",
                        "title": "Source",
                        "excerpts": ["Useful evidence"],
                    }
                ]
            }
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": "Quota exceeded"
                        if mode == "tool_error"
                        else json.dumps(output),
                    }
                ]
            }
            if mode == "tool_error":
                result["isError"] = True
            elif mode != "text":
                result["structuredContent"] = output
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": data["id"], "result": result}
        )

    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            auth=auth,
            transport=httpx.MockTransport(handle),
        )

    monkeypatch.setattr(parallel, "httpx", SimpleNamespace(AsyncClient=factory))
    return requests, behavior


def make_manager(tools=None, count=1):
    if tools is None:
        tools = [default_search_tool()]
    envs = ToolUseEnvs(
        [{"question": "Find evidence", "answer": "done", "pid": "test"}],
        tools,
        env_num=count,
    )
    manager = tool_use_manager.ToolUseEnvironmentManager(
        envs,
        tool_projection.tool_use_projection,
        SimpleNamespace(env=SimpleNamespace(history_length=2, max_steps=20)),
    )
    manager.reset()
    return manager


def action(query="web evidence", urls=None):
    params = {"query": query}
    if urls is not None:
        params["urls"] = urls
    return (
        "<tool_call>\ntool: parallel_search\nparameters: "
        + json.dumps(params)
        + "\n</tool_call>"
    )


def calls(requests):
    return [data["params"] for _, data in requests if data["method"] == "tools/call"]


def test_observation_consumes_search_fetch_and_followup(peer):
    requests, _ = peer
    manager = make_manager()
    for text in (
        action(),
        action(urls=["https://example.org/source"]),
        action("followup evidence"),
    ):
        obs, _, _, _ = manager.step([text])
        assert "Useful evidence" in obs["text"][0]
        assert "https://example.org/source" in obs["text"][0]
    sent = calls(requests)
    assert [call["name"] for call in sent] == ["web_search", "web_fetch", "web_search"]
    assert len({call["arguments"]["session_id"] for call in sent}) == 1
    for request, _ in requests:
        assert str(request.url) == parallel.MCP_URL
        assert request.headers["user-agent"] == parallel.USER_AGENT
        assert "authorization" not in request.headers


def test_interleaved_tasks_reset_and_retry_keep_identity(peer):
    requests, behavior = peer
    manager = make_manager(count=2)
    # Both environment instances use the same production transport factory.
    first = [
        env.tool_manager.available_tools["parallel_search"].session_id
        for env in manager.envs.envs
    ]
    assert first[0] != first[1]
    manager.step([action(), action()])
    behavior["mode"] = "tool_error"
    manager.step([action(), action()])
    behavior["mode"] = "ok"
    manager.step([action(urls=["https://example.org/source"]), action()])
    assert [call["arguments"]["session_id"] for call in calls(requests)] == first * 3
    manager.reset()
    manager.step([action(), action()])
    next_ids = [call["arguments"]["session_id"] for call in calls(requests)[-2:]]
    assert len(set(first + next_ids)) == 4


@pytest.mark.parametrize(
    "variables",
    [
        {},
        {"GOOGLE_API_KEY": "configured"},
        {"GOOGLE_CX": "configured"},
        {"GOOGLE_API_KEY": "key", "GOOGLE_CX": "cx"},
    ],
)
def test_default_respects_configured_credentials(monkeypatch, variables):
    for key, value in variables.items():
        monkeypatch.setenv(key, value)
    assert default_search_tool() == (
        "google_search" if variables else "parallel_search"
    )


@pytest.mark.parametrize(
    "script,option", [("gaia.py", "--tools"), ("openmanus_rollout.py", "--gaia_tools")]
)
@pytest.mark.parametrize("explicit", [False, True])
def test_actual_cli_parser_preserves_explicit_tools(
    monkeypatch, script, option, explicit
):
    class Parsed(Exception):
        pass

    original = argparse.ArgumentParser.parse_args
    captured = {}

    def parse(parser, *args, **kwargs):
        captured.update(vars(original(parser, *args, **kwargs)))
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    monkeypatch.setattr(
        "sys.argv",
        [script]
        + (["--env", "gaia"] if script == "openmanus_rollout.py" else [])
        + ([option, "google_search"] if explicit else []),
    )
    with pytest.raises(Parsed):
        runpy.run_path(
            str(Path(__file__).parents[1] / "scripts/rollout" / script),
            run_name="__main__",
        )
    selected = captured[option[2:]]
    assert selected[0] == ("google_search" if explicit else "parallel_search")


def test_disabled_and_unavailable_tools_do_not_send_requests(peer):
    requests, _ = peer
    manager = make_manager(tools=[])
    obs, _, _, _ = manager.step([action()])
    assert "not available" in obs["text"][0]
    assert requests == []


@pytest.mark.parametrize(
    "mode", ["tool_error", "rpc_error", "401", "429", "malformed", "timeout"]
)
def test_errors_reach_observation_without_fallback(peer, mode):
    requests, behavior = peer
    manager = make_manager()
    tool = manager.envs.envs[0].tool_manager.available_tools["parallel_search"]
    tool.timeout = 1
    session_id = tool.session_id
    behavior["mode"] = mode
    obs, _, _, _ = manager.step([action()])
    assert "error" in obs["text"][0].lower()
    assert tool.session_id == session_id
    assert len(calls(requests)) == 1
    assert all(str(request.url) == parallel.MCP_URL for request, _ in requests)


def test_text_content_is_consumed(peer):
    _, behavior = peer
    behavior["mode"] = "text"
    manager = make_manager()
    obs, _, _, _ = manager.step([action()])
    assert "Useful evidence" in obs["text"][0]


@pytest.mark.parametrize(
    "urls",
    [
        ["file:///etc/passwd"],
        ["https://user:pass@example.org"],
        [],
        "https://example.org",
        ["https://["],
    ],
)
def test_invalid_urls_never_reach_transport(peer, urls):
    requests, _ = peer
    manager = make_manager()
    obs, _, _, _ = manager.step([action(urls=urls)])
    assert "error" in obs["text"][0].lower()
    assert requests == []


def test_cancellation_propagates_through_dispatcher(peer, monkeypatch):
    manager = make_manager()
    tool = manager.envs.envs[0].tool_manager.available_tools["parallel_search"]
    session_id = tool.session_id

    class CancelledClient:
        async def __aenter__(self):
            raise asyncio.CancelledError

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        parallel,
        "httpx",
        SimpleNamespace(AsyncClient=lambda **kwargs: CancelledClient()),
    )
    with pytest.raises(asyncio.CancelledError):
        manager.step([action()])
    assert tool.session_id == session_id


def test_concurrent_conversations_keep_distinct_ids(peer):
    from concurrent.futures import ThreadPoolExecutor

    requests, _ = peer
    managers = [make_manager(), make_manager()]
    ids = [
        manager.envs.envs[0].tool_manager.available_tools["parallel_search"].session_id
        for manager in managers
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        observations = list(
            pool.map(lambda manager: manager.step([action()])[0], managers)
        )
    assert all("Useful evidence" in obs["text"][0] for obs in observations)
    assert {call["arguments"]["session_id"] for call in calls(requests)} == set(ids)
    assert len(set(ids)) == 2


def test_configured_google_is_executed_without_parallel_fallback(peer, monkeypatch):
    requests, _ = peer
    monkeypatch.setenv("GOOGLE_API_KEY", "saved-key")
    monkeypatch.setenv("GOOGLE_CX", "saved-engine")
    observed = []

    def get(url, params, timeout):
        observed.append((url, params, timeout))
        return SimpleNamespace(
            status_code=403,
            json=lambda: {"error": {"message": "Google denied the request"}},
        )

    monkeypatch.setattr("requests.get", get)
    manager = make_manager()
    obs, _, _, _ = manager.step([action().replace("parallel_search", "google_search")])
    assert "Google denied" in obs["text"][0]
    assert observed[0][1]["key"] == "saved-key"
    assert observed[0][1]["cx"] == "saved-engine"
    assert requests == []


@pytest.mark.parametrize(
    "configured", [None, [], ["google_search"], ["parallel_search"]]
)
def test_training_factory_preserves_saved_tool_list(
    peer, monkeypatch, tmp_path, configured
):
    from openmanus_rl.environments.env_manager import make_envs

    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps([{"question": "Find evidence", "answer": "done", "pid": "test"}])
    )
    env = SimpleNamespace(
        env_name="tool_use", seed=42, data_path=str(path), rollout=SimpleNamespace(n=1)
    )
    if configured is not None:
        env.available_tools = configured
    # Keep unrelated optional tools out of this focused factory test.
    from openmanus_rl.environments.env_package.tool_use.envs import ToolManager

    original = ToolManager._load_tool

    def load(manager, name):
        if name in ("google_search", "parallel_search"):
            original(manager, name)

    monkeypatch.setattr(ToolManager, "_load_tool", load)
    train, val = make_envs(
        SimpleNamespace(
            env=env, data=SimpleNamespace(train_batch_size=1, val_batch_size=1)
        )
    )
    for manager in (train, val):
        names = manager.envs.envs[0].available_tools
        assert (
            names == configured
            if configured is not None
            else names[0] == "parallel_search"
        )


def test_saved_dotenv_credentials_are_respected(monkeypatch, tmp_path):
    from dotenv.main import load_dotenv

    path = tmp_path / ".env"
    path.write_text("GOOGLE_API_KEY=saved-key\nGOOGLE_CX=saved-engine\n")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: load_dotenv(path, override=False))
    try:
        assert default_search_tool() == "google_search"
    finally:
        # load_dotenv writes directly to os.environ, outside monkeypatch tracking.
        import os

        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("GOOGLE_CX", None)


@pytest.mark.parametrize("query", ["", "  ", None, 42])
def test_invalid_search_query_is_not_sent(peer, query):
    requests, _ = peer
    manager = make_manager()
    obs, _, _, _ = manager.step([action(query=query)])
    assert "error" in obs["text"][0].lower()
    assert requests == []
