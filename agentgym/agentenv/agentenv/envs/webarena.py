from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class WebarenaEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere's the information you'll have:\nThe user's objective: This is the task you're trying to complete.\nThe current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\nThe current web page's URL: This is the page you're currently navigating.\nThe open tabs: These are the tabs you have open.\nThe previous action: This is the action you just performed. It may be helpful to track your progress.\n\nThe actions you can perform fall into several categories:\n\nPage Operation Actions:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n`hover [id]`: Hover over an element with id.\n`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n`scroll [direction=down|up]`: Scroll the page up or down.\n\nTab Management Actions:\n`new_tab`: Open a new, empty browser tab.\n`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n`close_tab`: Close the currently active tab.\n\nURL Navigation Actions:\n`goto [url]`: Navigate to a specific URL.\n`go_back`: Navigate to the previously viewed page.\n`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\nCompletion Action:\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\nHomepage:\nIf you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.\nhttp://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the websites.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation\n2. You should only issue one action at a time.\n3. You should follow the examples to reason step by step and then issue the next action.\n4. Generate the action in the correct format. Start with a \"In summary, the next action I will perform is\" phrase, followed by action inside ``````. For example, \"In summary, the next action I will perform is ```click [1234]```\".\n5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.",
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

        self.env_id = ok.json()["env_idx"]

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> Dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["terminated"],
        )

    def reset(self, idx: int) -> Dict[str, Any]:
        response = self._post("reset", {"seed": 0, "idx": idx})
        return response


class WebarenaTask(BaseTask):
    env_client_cls = WebarenaEnvClient
    env_name = "Webarena"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
