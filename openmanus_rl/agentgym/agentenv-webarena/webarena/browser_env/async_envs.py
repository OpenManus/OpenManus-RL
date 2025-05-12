import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.spaces import Box, Text
from playwright.async_api import Page, ViewportSize, async_playwright

from .actions import Action, aexecute_action, get_action_space
from .utils import DetachedPage, png_bytes_to_numpy


class AsyncScriptBrowserEnv(Env[npt.NDArray[np.uint8], Action]):
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """

    def __init__(
        self,
        max_page_length: int = 2048,
        headless: bool = True,
        slow_mo: int = 0,
        timeout: int = 30000,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
    ):
        self.observation_space = Box(
            0,
            255,
            (viewport_size["height"], viewport_size["width"], 4),
            np.uint8,
        )
        # TODO: make Space[Action] = ActionSpace
        self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.reset_finished = False
        self.timeout = timeout
        self.viewport_size = viewport_size

    async def setup(self, config_file: Path | None = None) -> None:
        self.context_manager = async_playwright()
        self.playwright = await self.context_manager.__aenter__()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo
        )
        if config_file:
            with open(config_file, "r") as f:
                instance_config = json.load(f)
        else:
            instance_config = {}

        storage_state = instance_config.get("storage_state", None)
        start_url = instance_config.get("start_url", None)
        geolocation = instance_config.get("geolocation", None)

        self.context = await self.browser.new_context(
            viewport=self.viewport_size,
            storage_state=storage_state,
            geolocation=geolocation,
            device_scale_factor=1,
        )
        self.page = await self.context.new_page()
        if start_url:
            await self.page.goto(start_url)

    async def areset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[npt.NDArray[np.uint8], dict[str, object]]:
        """
        Reset the environment.
        :param options: options for the environment. The options are:
            - storage_state: the path to the storage state file
        """
        super().reset(seed=seed, options=options)
        if self.reset_finished:
            await self.context_manager.__aexit__()
        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                await self.setup(config_file=config_file)
            else:
                raise ValueError(f"Config state {config_file} does not exist.")
        else:
            await self.setup()
        self.reset_finished = True
        content = await self.page.content()
        screenshot = png_bytes_to_numpy(await self.page.screenshot())
        return (
            screenshot,
            {"page": DetachedPage(self.page.url, content)},
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[npt.NDArray[np.uint8], dict[str, object]]:
        return asyncio.run(self.areset(seed=seed, options=options))

    async def aclose(self) -> None:
        if self.reset_finished:
            await self.context_manager.__aexit__()

    def close(self) -> None:
        asyncio.run(self.aclose())

    async def astep(
        self, action: Action
    ) -> tuple[npt.NDArray[np.uint8], float, bool, bool, dict[str, object]]:
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")
        success = False
        fail_error = ""
        try:
            self.page = await aexecute_action(action, self.page, self.context)
            success = True
        except Exception as e:
            fail_error = str(e)

        try:
            content = await self.page.content()
            screenshot = png_bytes_to_numpy(await self.page.screenshot())
        except:
            await self.page.wait_for_load_state("load")
            content = await self.page.content()
            screenshot = png_bytes_to_numpy(await self.page.screenshot())

        return (
            screenshot,
            float(success),
            False,
            False,
            {
                "page": DetachedPage(self.page.url, content),
                "fail_error": fail_error,
            },
        )

    def step(
        self, action: Action
    ) -> tuple[npt.NDArray[np.uint8], float, bool, bool, dict[str, object]]:
        return asyncio.run(self.astep(action), debug=True)
