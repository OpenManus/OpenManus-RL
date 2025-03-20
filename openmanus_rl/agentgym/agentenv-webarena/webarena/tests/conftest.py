from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

from browser_env import AsyncScriptBrowserEnv, ScriptBrowserEnv

HEADLESS = True
SLOW_MO = 0


@pytest.fixture(scope="function")
def script_browser_env() -> Generator[ScriptBrowserEnv, None, None]:
    """Create a ScriptBrowserEnv instance for testing.
    It is automatically closed after the test session.
    This is helpful when the test failed and the browser is still open.
    """
    env = ScriptBrowserEnv(
        headless=HEADLESS,
        slow_mo=SLOW_MO,
    )
    yield env
    env.close()


@pytest.fixture(scope="function")
def current_viewport_script_browser_env() -> Generator[
    ScriptBrowserEnv, None, None
]:
    env = ScriptBrowserEnv(
        headless=HEADLESS,
        slow_mo=SLOW_MO,
        current_viewport_only=True,
    )
    yield env
    env.close()


@pytest.fixture(scope="function")
def accessibility_tree_script_browser_env() -> Generator[
    ScriptBrowserEnv, None, None
]:
    env = ScriptBrowserEnv(
        headless=HEADLESS,
        slow_mo=SLOW_MO,
        observation_type="accessibility_tree",
    )
    yield env
    env.close()


@pytest.fixture(scope="function")
def accessibility_tree_current_viewport_script_browser_env() -> Generator[
    ScriptBrowserEnv, None, None
]:
    env = ScriptBrowserEnv(
        headless=HEADLESS,
        slow_mo=SLOW_MO,
        observation_type="accessibility_tree",
        current_viewport_only=True,
    )
    yield env
    env.close()


@pytest_asyncio.fixture(scope="function", autouse=True)
async def async_script_browser_env() -> AsyncGenerator[
    AsyncScriptBrowserEnv, None
]:
    env = AsyncScriptBrowserEnv(headless=HEADLESS, slow_mo=SLOW_MO)
    yield env
    await env.aclose()
