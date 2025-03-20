import asyncio
import json

from browser_env import *

auth_json = {
    "cookies": [
        {
            "name": "session-username",
            "value": "standard_user",
            "domain": "www.saucedemo.com",
            "path": "/",
            "httpOnly": False,
            "secure": False,
            "sameSite": "Lax",
        }
    ],
    "origins": [],
}


def test_auth_cookie() -> None:
    env = ScriptBrowserEnv()
    env.reset()
    _, reward, _, _, info = env.step(
        create_goto_url_action("https://www.saucedemo.com/inventory.html"),
    )
    assert reward == 1
    assert "page" in info and isinstance(info["page"], DetachedPage)
    assert info["page"].url == "https://www.saucedemo.com/"
    json.dump(auth_json, open("/tmp/auth.json", "w"))
    instance_config = {"storage_state": "/tmp/auth.json"}
    json.dump(instance_config, open("/tmp/config.json", "w"))
    env.reset(options={"config_file": "/tmp/config.json"})
    _, reward, _, _, info = env.step(
        create_goto_url_action("https://www.saucedemo.com/inventory.html"),
    )
    assert reward == 1
    assert "page" in info and isinstance(info["page"], DetachedPage)
    assert info["page"].url == "https://www.saucedemo.com/inventory.html"
    env.close()


def test_async_auth_cookie() -> None:
    env = AsyncScriptBrowserEnv()

    async def _test() -> None:
        await env.areset()
        _, reward, _, _, info = await env.astep(
            create_goto_url_action("https://www.saucedemo.com/inventory.html"),
        )
        assert reward == 1
        assert "page" in info and isinstance(info["page"], DetachedPage)
        assert info["page"].url == "https://www.saucedemo.com/"
        json.dump(auth_json, open("/tmp/auth.json", "w"))
        instance_config = {"storage_state": "/tmp/auth.json"}
        json.dump(instance_config, open("/tmp/config.json", "w"))
        await env.areset(options={"config_file": "/tmp/config.json"})
        _, reward, _, _, info = await env.astep(
            create_goto_url_action("https://www.saucedemo.com/inventory.html"),
        )
        assert reward == 1
        assert "page" in info and isinstance(info["page"], DetachedPage)
        assert info["page"].url == "https://www.saucedemo.com/inventory.html"
        await env.aclose()

    asyncio.run(_test())
