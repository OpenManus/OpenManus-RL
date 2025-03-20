from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool


class BaseEnvClient(metaclass=ABCMeta):
    conversation_start = ()

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total size of the environment.
        """

    @abstractmethod
    def observe(self) -> str:
        """
        Parse env server response and give a text message to prompt the LLM.
        """

    @abstractmethod
    def step(self, action) -> StepOutput:
        """
        Parse model output from the action and call the env server.
        """

    @abstractmethod
    def reset(self, idx: int) -> None:
        """
        Reset the environment.
        """
