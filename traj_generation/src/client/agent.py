from typing import List


class AgentClient:
    def __init__(self, *args, **kwargs):
        pass

    def inference(self, history: List[dict]) -> str:
        raise NotImplementedError()
