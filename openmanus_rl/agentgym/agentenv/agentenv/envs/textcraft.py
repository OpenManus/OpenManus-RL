from typing import Any, Mapping

import re

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class TextCraftEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": 'You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]".\nEvery round I will give you an observation, you have to respond an action based on the state and instruction. You can "get" an object (ingredients) from the inventory or the environment, look-up the game inventory by "inventory", or "craft" (target) using any of the crafting commands. You can use ONLY these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead.\nYour response should use the following format:\n\nThought:\n ... \n\nAction:\n ... ',
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        minecraft_dir: str = "agentenv_textcraft/",
        commands: str = None,
        goal: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        dir_info = {"minecraft_dir": minecraft_dir, "commands": commands, "goal": goal}
        ok = requests.post(
            f"{self.env_server_base}/create", timeout=self.timeout, json=dir_info
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        print(ok)
        self.env_id = ok["id"]
        self.info = {
            "observation": ok["observation"],
            "reward": 0,
            "done": False,
        }

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        action = action.split("Instruction:")[0].split("Action:")[-1]
        action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
        action = " ".join(action.split()).strip()
        response = self._post("step", {"action": action})
        print(response)
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": idx})
        self.info.update(
            {
                "observation": response["observation"],
                "reward": 0,
                "done": False,
            }
        )
        return response


class TextCraftTask(BaseTask):
    env_client_cls = TextCraftEnvClient
    env_name = "TextCraft"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
