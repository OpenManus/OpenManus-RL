from typing import Any, Mapping

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class WebshopEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought:\nI think ... \n\nAction: \nclick[something]",
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(
            f"{self.env_server_base}/create",
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    def __len__(self):
        return self.data_len


    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["env_idx"] = self.env_id
        max_retries = 5
        for attempt in range(max_retries):
            res = requests.post(
                f"{self.env_server_base}/{path}",
                json=data,
                timeout=self.timeout,
            )
            if res.status_code == 503:
                import time

                time.sleep(0.1)
            elif res.status_code == 200:
                break
            else:
                print("---------------------")
                print(res.status_code)
                print(data)
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["state"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int) -> dict[str, Any]:
        response = self._post("reset", {"session_id": idx})
        response[0] = self.observe()
        return response


class WebshopTask(BaseTask):
    env_client_cls = WebshopEnvClient
    env_name = "WebShop"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
