"""
WebarenaEnvServer
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_id_based_action,
    create_stop_action,
)
from browser_env.actions import ActionParsingError
from browser_env.env_config import URL_MAPPINGS
from browser_env.helper_functions import RenderHelper, get_action_description
from browser_env.utils import Observation
from evaluation_harness import evaluator_router


class PromptConstructor:
    """
    Construct prompt
    """

    def __init__(
        self,
        instruction_path: str | Path,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction = instruction

    def construct(
        self,
        trajectory: list,
        intent: str,
        meta_data: dict[str, Any] = {},
    ):
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])

        return current

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(f"Cannot parse action from response {response}")

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class WebarenaEnvServer:
    """
    WebarenaEnvServer
    """

    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}
        self.trajectory = {}
        self.meta_data = {}
        self.intent = {}  # question in config_file
        self.prompt_constructor = PromptConstructor(
            instruction_path="./agent/prompts/jsons/p_cot_id_actree_2s.json"
        )

    def create(self) -> int:
        """
        Only call this create function once.
        """
        env_idx = self._max_id
        self.env[self._max_id] = ScriptBrowserEnv(
            headless=True,
            slow_mo=100,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720},
        )
        self.trajectory[self._max_id] = []
        self.meta_data[self._max_id] = {}
        self.intent[self._max_id] = ""
        self.env[self._max_id].reset()
        self._max_id += 1
        return env_idx

    def step(
        self,
        env_idx: int,
        response: str,
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        """
        Return:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
        """
        try:
            force_prefix = self.prompt_constructor.instruction["meta_data"].get(
                "force_prefix", ""
            )
            response = f"{force_prefix}{response}"
            parsed_response = self.prompt_constructor.extract_action(response)
            action = create_id_based_action(parsed_response)
            action["raw_prediction"] = response

            self.trajectory[env_idx].append(action)

            action_str = get_action_description(
                action,
                self.state_info["info"]["observation_metadata"],
                action_set_tag="id_accessibility_tree",
                prompt_constructor=self.prompt_constructor,
            )
            self.meta_data[env_idx]["action_history"].append(action_str)

            obs, reward, terminated, truncated, info = self.env[env_idx].step(action)

            self.state_info = {"observation": obs, "info": info}
            self.trajectory[env_idx].append(self.state_info)

            prompt = self.prompt_constructor.construct(
                self.trajectory[env_idx], self.intent[env_idx], self.meta_data[env_idx]
            )

            if terminated:
                # add a action place holder
                self.trajectory.append(create_stop_action(""))

            if terminated or action["action_type"] == ActionTypes.STOP:
                terminated = True
                evaluator = evaluator_router(self.config_file)
                reward = evaluator(
                    trajectory=self.trajectory[env_idx],
                    config_file=self.config_file,
                    page=self.env[env_idx].page,
                    client=self.env[env_idx].get_page_client(self.env[env_idx].page),
                )

            return (prompt, reward, terminated, truncated, info)
        except Exception as e:
            return (str(e), 0, False, False, None)

    def observation(self, env_idx) -> dict[str, Observation]:
        """
        Return
            {"text": text_obs, "image": image_obs}

        Example text:
        [4] RootWebArea 'Projects · Dashboard · GitLab' focused: True
        [12] link 'Skip to content'
        [28] link 'Dashboard'
        [2266] button '' hasPopup: menu expanded: False
        [63] textbox 'Search GitLab' required: False
        [61] generic 'Use the shortcut key <kbd>/</kbd> to start a search'
        [79] link 'Create new...'
        [95] link 'Issues'
                [97] generic '13 assigned issues'
        [101] link 'Merge requests'
                [104] generic '8 merge requests
        """
        return self.prompt_constructor.construct(
            self.trajectory[env_idx], self.intent[env_idx], self.meta_data[env_idx]
        )

    def observation_metadata(self, env_idx):
        """
        Return
        {
            "text": self.text_processor.meta_data,
            "image": self.image_processor.meta_data,
        }
        """
        return self.env[env_idx]._get_obs_metadata()

    def reset(
        self, env_idx, seed: int | None = None, options: dict[str, str] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        options={"config_file": config_file}
        Return:
            (observation, info)
        """
        self.config_file = Path(options["config_file"])
        with open(self.config_file) as f:
            _c = json.load(f)
            self.intent[env_idx] = _c["intent"]
        obs, info = self.env[env_idx].reset(seed=seed, options=options)

        self.trajectory[env_idx] = []
        self.state_info = {"observation": obs, "info": info}
        self.trajectory[env_idx].append(self.state_info)

        self.meta_data[env_idx] = {"action_history": ["None"]}

        return (obs, info)

    def close(self, env_idx) -> None:
        self.env[env_idx].close()


webarena_env_server = WebarenaEnvServer()
