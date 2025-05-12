import asyncio
import collections
import json
import tempfile
from typing import Callable, Dict, Optional, Tuple, Type, Union, cast

import pytest
from gymnasium.vector import AsyncVectorEnv
from playwright.sync_api import Page

from browser_env import (
    Action,
    AsyncScriptBrowserEnv,
    DetachedPage,
    ScriptBrowserEnv,
    create_focus_and_click_action,
    create_goto_url_action,
    create_keyboard_type_action,
    create_playwright_action,
    create_scroll_action,
)
from browser_env.actions import create_id_based_action
from browser_env.env_config import (
    ACCOUNTS,
    GITLAB,
    REDDIT,
    SHOPPING,
    SHOPPING_ADMIN,
)


def test_script_browser_env(script_browser_env: ScriptBrowserEnv) -> None:
    env = script_browser_env
    env.reset()
    env.step(
        create_goto_url_action("http://www.example.com"),
    )
    env.step(
        create_focus_and_click_action(
            element_role="link",
            element_name="More",
        ),
    )
    _, _, _, _, info = env.step(
        create_focus_and_click_action(
            element_role="link",
            element_name="2606",
        )
    )
    assert isinstance(info["page"], DetachedPage)
    assert info["page"].url == "https://www.rfc-editor.org/rfc/rfc2606.html"


@pytest.mark.asyncio
async def test_async_script_browser_env(
    async_script_browser_env: AsyncScriptBrowserEnv,
) -> None:
    env = async_script_browser_env
    await env.areset()
    await env.astep(
        create_goto_url_action("http://www.example.com"),
    )
    await env.astep(
        create_focus_and_click_action(
            element_role="link",
            element_name="More",
        ),
    )
    _, _, _, _, info = await env.astep(
        create_focus_and_click_action(
            element_role="link",
            element_name="2606",
        )
    )
    assert isinstance(info["page"], DetachedPage)
    assert info["page"].url == "https://www.rfc-editor.org/rfc/rfc2606.html"


def collate_actions(actions: list[Action]) -> dict[str, list[object]]:
    action_dict = collections.defaultdict(list)
    for action in actions:
        for key, value in action.items():
            action_dict[key].append(value)
    return action_dict


@pytest.mark.skip(reason="Gym doesn't support self-defined observations")
def test_parallel_script_browser_env() -> None:
    vector_env = AsyncVectorEnv(
        [
            lambda: ScriptBrowserEnv(),
            lambda: ScriptBrowserEnv(),
        ],
        shared_memory=True,
    )
    vector_env.reset()
    vector_env.step(
        collate_actions(
            [
                create_goto_url_action("http://www.example.com"),
            ]
            * 2
        )
    )
    vector_env.step(
        collate_actions(
            [
                create_focus_and_click_action(
                    element_role="link",
                    element_name="More",
                ),
            ]
            * 2
        )
    )
    _, _, _, _, info = vector_env.step(
        collate_actions(
            [
                create_focus_and_click_action(
                    element_role="link",
                    element_name="2606",
                ),
                create_focus_and_click_action(
                    element_role="link",
                    element_name="6761",
                ),
            ]
        )
    )
    # assert is_bearable(info["page"].tolist(), list[DetachedPage])
    assert info["page"][0].url == "https://www.rfc-editor.org/rfc/rfc2606.html"
    assert info["page"][1].url == "https://www.rfc-editor.org/rfc/rfc6761.html"
    vector_env.close()  # type: ignore[no-untyped-call]


def test_focus_placeholder_and_label(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    env.reset()
    for action in [
        create_goto_url_action("https://demo.applitools.com"),
        create_focus_and_click_action("placeholder", "Enter your username"),
        create_keyboard_type_action("abc"),
        create_focus_and_click_action("placeholder", "Enter your password"),
        create_keyboard_type_action("123"),
        create_focus_and_click_action("label", "Remember Me"),
        create_focus_and_click_action("link", "Sign in"),
    ]:
        _, success, _, _, info = env.step(action)
        assert success
    assert info["page"].url == "https://demo.applitools.com/app.html"


def test_html_current_viewport(
    current_viewport_script_browser_env: ScriptBrowserEnv,
) -> None:
    s1 = "detailed information about how mammals could be classified."
    s2 = "Types of mammals"
    env = current_viewport_script_browser_env
    env.reset()
    obs, success, _, _, info = env.step(
        create_playwright_action(
            'page.goto("https://russmaxdesign.github.io/exercise/")'
        )
    )
    assert success
    assert s1 in obs["text"] and s2 not in obs["text"]
    obs, success, _, _, info = env.step(create_scroll_action("down"))
    assert success
    assert s1 not in obs["text"] and s2 in obs["text"]


def test_accessibility_tree(
    accessibility_tree_script_browser_env: ScriptBrowserEnv,
) -> None:
    s1 = "checkbox 'Yes'"
    s2 = "button 'Submit'"
    env = accessibility_tree_script_browser_env
    env.reset()
    obs, success, _, _, info = env.step(
        create_playwright_action(
            'page.goto("https://russmaxdesign.github.io/exercise/")'
        )
    )
    assert success
    assert s1 in obs["text"] and s2 in obs["text"]


def test_accessibility_tree_viewport(
    accessibility_tree_current_viewport_script_browser_env: ScriptBrowserEnv,
) -> None:
    s1 = "combobox 'Favourite mammal'"
    s2 = "gridcell 'Canyon bat'"
    s3 = "heading 'Useful links'"
    env = accessibility_tree_current_viewport_script_browser_env
    env.reset()

    obs, success, _, _, info = env.step(
        create_playwright_action(
            'page.goto("https://russmaxdesign.github.io/exercise/")'
        )
    )
    assert success
    assert (
        s1 in obs["text"] and s2 not in obs["text"] and s3 not in obs["text"]
    )
    obs, success, _, _, info = env.step(create_scroll_action("down"))
    assert success
    assert (
        s1 not in obs["text"] and s2 in obs["text"] and s3 not in obs["text"]
    )

    obs, success, _, _, info = env.step(create_scroll_action("down"))
    assert success
    assert s1 not in obs["text"] and s2 in obs["text"] and s3 in obs["text"]


def test_multiple_start_url(script_browser_env: ScriptBrowserEnv) -> None:
    temp_config = tempfile.NamedTemporaryFile("w", delete=False)
    config = {
        "require_login": False,
        "start_url": f"{REDDIT} |AND| {REDDIT}/forums",
    }
    json.dump(config, temp_config)
    temp_config.close()

    env = script_browser_env
    env.reset(options={"config_file": temp_config.name})
    assert len(env.context.pages) == 2
    assert env.context.pages[0].url == f"{REDDIT}/"
    assert env.context.pages[1].url == f"{REDDIT}/forums", env.context.pages[
        1
    ].url


def test_observation_tab_information(
    accessibility_tree_current_viewport_script_browser_env: ScriptBrowserEnv,
) -> None:
    env = accessibility_tree_current_viewport_script_browser_env
    env.reset()
    obs, *_ = env.step(
        create_id_based_action(
            "goto [https://russmaxdesign.github.io/exercise/]"
        )
    )
    obs, *_ = env.step(create_id_based_action("new_tab"))

    obs, *_ = env.step(
        create_id_based_action("goto [https:///www.google.com]")
    )
    assert obs["text"].startswith(  # type: ignore[union-attr]
        "Tab 0: Exercise page for keyboard and screen reader use | Tab 1 (current): Google"
    )

    obs, *_ = env.step(create_id_based_action("tab_focus [0]"))

    assert obs["text"].startswith(  # type: ignore[union-attr]
        "Tab 0 (current): Exercise page for keyboard and screen reader use | Tab 1: Google"
    )


def test_accessibility_tree_observation_update(
    accessibility_tree_current_viewport_script_browser_env: ScriptBrowserEnv,
) -> None:
    env = accessibility_tree_current_viewport_script_browser_env
    env.reset()
    obs, *_ = env.step(
        create_playwright_action(
            "page.goto('https://russmaxdesign.github.io/exercise/')"
        )
    )
    obs, *_ = env.step(
        create_playwright_action(
            'page.get_by_label("Full name").fill("UNIQUE_NAME")'
        )
    )
    assert "UNIQUE_NAME" in obs["text"]
