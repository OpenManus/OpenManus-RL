from dataclasses import dataclass
from typing import Optional, Sequence, TypedDict

import numpy as np

from agentenv.controller.agent import Agent
from agentenv.controller.task import BaseTask, ExperienceOutput, GenerationConfig

ConversationMessage = TypedDict(
    "ConversationMessage", {"from": str, "loss": Optional[bool], "value": str}
)


@dataclass
class EvaluationOutput:
    experiences: Sequence[ExperienceOutput]
    score: float
    success: float


class BaseAgentEnvController:
    def __init__(self, agent: Agent, tasks: Sequence[BaseTask]) -> None:
        self.agent = agent
        self.tasks = tasks

    def generate_experience(
        self,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        experience = []
        if isinstance(idxs[0], int):
            experience += self.tasks[0].generate_experience(
                self.agent.model,
                self.agent.tokenizer,
                idxs,
                generation_config,
                max_rounds,
            )
        elif isinstance(idxs[0], Sequence):
            for idx, task in enumerate(self.tasks):
                experience += task.generate_experience(
                    self.agent.model,
                    self.agent.tokenizer,
                    idxs[idx],
                    generation_config,
                    max_rounds,
                )
        else:
            raise ValueError("Incorrect Format for idxs")

        return experience


class Evaluator(BaseAgentEnvController):
    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
    ) -> EvaluationOutput:
        exps = self.generate_experience(
            idxs=idxs if idxs is not None else [list(range(len(task.clients[0]))) for task in self.tasks],
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
        rewards = np.array([exp.reward for exp in exps])
        return EvaluationOutput(
            experiences=exps, score=rewards.mean(), success=(rewards == 1).mean()
        )


class BaseTrainer(BaseAgentEnvController):
    def __init__(self, agent: Agent, tasks: Sequence[BaseTask]) -> None:
        super().__init__(agent, tasks)

    def train(self):
        pass

    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] = None,
    ) -> EvaluationOutput:
        exps = self.generate_experience(
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
        rewards = np.array([exp.reward for exp in exps])
        return EvaluationOutput(
            experiences=exps, score=rewards.mean(), success=(rewards == 1).mean()
        )

    def save_model(self):
        pass
