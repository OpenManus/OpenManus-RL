from typing import Dict, Generator, Optional, Tuple, Type, Union, cast

import pytest
from playwright.sync_api import Page

from browser_env import ScriptBrowserEnv, create_playwright_action

HEADLESS = True
SLOW_MO = 0


def test_frame_locator(script_browser_env: ScriptBrowserEnv) -> None:
    env = script_browser_env
    seq = """page.goto("https://www.littlewebhut.com/articles/html_iframe_example/")
    page.frame_locator("iframe[name=\\"imgbox\\"]").get_by_role("img").click()"""

    env.reset()
    for action in seq.split("\n"):
        action = action.strip()
        _, success, _, _, info = env.step(create_playwright_action(action))
        assert success


def test_basic(script_browser_env: ScriptBrowserEnv) -> None:
    # click, fill, press, check, goto
    env = script_browser_env
    seq = """page.goto("https://demo.playwright.dev/todomvc/")
    page.get_by_placeholder("What needs to be done?").click()
    page.get_by_placeholder("What needs to be done?").fill("hello")
    page.get_by_placeholder("What needs to be done?").press("Enter")
    page.get_by_placeholder("What needs to be done?").fill("world")
    page.get_by_placeholder("What needs to be done?").press("Enter")
    page.get_by_placeholder("What needs to be done?").fill("yes")
    page.get_by_placeholder("What needs to be done?").press("Enter")
    page.get_by_placeholder("What needs to be done?").fill("no")
    page.get_by_placeholder("What needs to be done?").press("Enter")
    page.get_by_role("listitem").filter(has_text="world").get_by_role("checkbox", name="Toggle Todo").check()
    page.get_by_role("button", name="Clear completed").click()"""

    env.reset()
    for action in seq.split("\n"):
        action = action.strip()
        _, success, _, _, info = env.step(create_playwright_action(action))
        assert success


@pytest.mark.skip(reason="not important, but the site is flaky")
def test_hover(script_browser_env: ScriptBrowserEnv) -> None:
    env = script_browser_env
    seq = """page.goto("https://www.w3schools.com/cssref/tryit.php?filename=trycss_sel_hover")
    page.frame_locator("iframe[name=\\'iframeResult\\']").get_by_role("link", name="w3schools.com").hover()"""

    env.reset()
    for action in seq.split("\n"):
        action = action.strip()
        _, success, _, _, info = env.step(create_playwright_action(action))
        assert success


@pytest.mark.skip(reason="not important, but the site is flaky")
def test_select_option(script_browser_env: ScriptBrowserEnv) -> None:
    env = script_browser_env
    seq = """page.goto("https://www.w3schools.com/tags/tryit.asp?filename=tryhtml_select")
    page.frame_locator("iframe[name=\\'iframeResult\\']").get_by_role("combobox", name="Choose a car:").select_option("opel")"""

    env.reset()
    for action in seq.split("\n"):
        action = action.strip()
        _, success, _, _, info = env.step(create_playwright_action(action))
        assert success


def test_xpath(script_browser_env: ScriptBrowserEnv) -> None:
    env = script_browser_env
    seq = """page.goto("https://demo.playwright.dev/todomvc/")
    page.goto("https://demo.playwright.dev/todomvc/#/")
    page.get_by_placeholder("What needs to be done?").click()
    page.get_by_placeholder("What needs to be done?").fill("hello")
    page.get_by_placeholder("What needs to be done?").press("Enter")
    page.get_by_role("link", name="Completed").click()
    page.locator("xpath=/html/body/section/div/header/input").fill("no")
    page.get_by_placeholder("What needs to be done?").press("Enter")
    page.goto("https://bic-berkeley.github.io/psych-214-fall-2016/string_literals.html")
    page.locator("xpath=//*[@id=\'searchbox\']/div/form/input[1]").fill("type")"""
    env.reset()
    for action in seq.split("\n"):
        action = action.strip()
        _, success, _, _, info = env.step(create_playwright_action(action))
        assert success
