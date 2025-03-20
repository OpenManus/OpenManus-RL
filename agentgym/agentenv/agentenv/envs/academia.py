from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class AcademiaEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are an autonomous intelligent agent. You can use actions to help people solve problems.\nWe detail name, description, input(parameters) and output(returns) of each action as follows:\nName: loadPaperNet()\nDescription: Load PaperNet. In this net, nodes are papers and edges are citation relationships between papers.\n\nName: loadAuthorNet()\nDescription: Load AuthorNet. In this net, nodes are authors and edges are collaboration relationships between authors.\n\nName: neighbourCheck(graph, node)\nDescription: List the first-order neighbors connect to the node. In paperNet, neigbours are cited papers of the paper. In authorNet, neigbours are collaborators of the author.\nParameters:\n- graph (Type: string, Enum: [PaperNet, AuthorNet]): The name of the graph to check\n- node (Type: string): The node for which neighbors will be listed\nReturns:\n- neighbors (Type: array)\n\nName: paperNodeCheck(node)\nDescription: Return detailed attribute information of a specified paper in PaperNet\nParameters:\n- node (Type: string): Name of the paper.\nReturns:\n- authors : The authors of the paper\n- year : The puslished year of the paper\n- venue : The published venue of the paper\n- n_citation : The number of citations of the paper\n- keywords : The keywords of the paper\n- doc_type : The document type of the paper\n\nName: authorNodeCheck(node)\nDescription: Return detailed attribute information of a specified author in AuthorNet\nParameters:\n- node (Type: string): name of the author.\nReturns:\n- name : The name of the author\n- org : The organization of the author\n\nName: authorEdgeCheck(node1, node2)\nDescription: Return detailed attribute information of the edge between two specified nodes in a AuthorNet.\nParameters:\n- node1 (Type: string): The first node of the edge\n- node2 (Type: string): The second node of the edge\nReturns:\n- papers : All papers that the two authors have co-authored\n\nName: finish(answer)\nDescription: Return an answer and finish the task\nParameters:\n- answer (Type: ['string', 'number', 'array']): The answer to be returned\n\nYou should call loadPaperNet or loadAuthorNet first! If you are finished, you will call \"finish\" action.\nPlease refer to the format of examples below to solve the requested goal. Please provide your thought to solve the question. You should give the thought with no more than 3 sentences. You need to give your thought together with your action!\n\nYour response must be in the format of:\nThought: [your thought]\n\nAction: [your action] with Action Input: [your action input]\n\nHere is an example:\n\nGoal: When was the paper Learning the Principle of Least Action with Reinforcement Learning. published?\nThought: The question is asking some basic information of a paper. I need to load the paper graph.\n\nAction: loadPaperNet with Action Input: {}\nObservation: PaperNet is loaded.\nThought: The question is asking the published date of a paper, we need to check the node from the PaperNet.\n\nAction: paperNodeCheck with Action Input: {\"node\":\"Learning the Principle of Least Action with Reinforcement Learning.\"}\nObservation: {'year': 2021, 'venue': 'AAAI Spring Symposium - MLPS', 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference'}\nThought: The published date of the paper is 2021.\n\nAction: finish with Action Input: {\"answer\": \"2021\"}\nObservation: 2021\n",
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
        self.id = 0
        data = dict()
        data["id"] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

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
            done=response["done"],
        )

    def reset(self, id: int) -> Dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response


class AcademiaTask(BaseTask):
    env_client_cls = AcademiaEnvClient
    env_name = "Academia"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
