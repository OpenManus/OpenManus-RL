from typing import List

from src.client import AgentClient


class CountHistoryAgent(AgentClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, history: List[dict]) -> str:
        return "I received {} items in history.".format(len(history))
