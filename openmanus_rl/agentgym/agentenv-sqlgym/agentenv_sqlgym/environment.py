"""
SqlGymEnvServer
"""

import os
import random
import time
from typing import Literal, Mapping, Optional, Tuple

from sqlgym import SqlGymEnv
from sqlgym.datasets import BirdDataset


class NotInitializedError(Exception):
    pass


SqlGymMode = Literal["not_initialized"] | Literal["bird_train"] | Literal["bird_dev"]

ITEM_RANGE = {
    "bird_train": (0, 9428),  # 0 <= item_id < 9428
    "bird_dev": (9428, 10962),
}


class SqlGymEnvServer:
    """
    SqlGymEnvServer
    """

    def __init__(self) -> None:

        self.env: Mapping[int, Tuple[SqlGymEnv | None, SqlGymMode]] = {}
        self.ls = []
        self.sz = 8
        self.now = -1

    def create(self) -> int:
        random.seed(time.time())
        idx = random.randint(0, 489576)
        print(f"-------Env {idx} created--------")
        if len(self.env) == self.sz:
            self.now = self.now + 1
            if self.now == self.sz:
                self.now = 0
            return self.ls[self.now]

        self.env[idx] = (None, "not_initialized")
        self.ls.append(idx)
        return idx

    def observation(self, env_idx):
        self._check_env_idx(env_idx)
        return self.env[env_idx][0].observation

    def step(self, env_idx, action: str):
        self._check_env_idx(env_idx)
        execution_result, reward, terminated, info, _ = self.env[env_idx][0].step(
            action
        )
        execution_result = str(execution_result)
        if len(execution_result) > 100:
            execution_result = execution_result[:100] + "..."
        return execution_result, reward, terminated, info

    def reset(self, env_idx, item_id: Optional[int]):
        try:
            self._check_env_idx(env_idx)
        except NotInitializedError:
            print(f"env_idx {env_idx} not initialized, initializing...")

        _id = None
        for mode, r in ITEM_RANGE.items():
            if r[0] <= item_id < r[1]:
                if self.env[env_idx][1] != mode:
                    self.env[env_idx] = (
                        SqlGymEnv(self._get_dataset_from_mode(mode)),
                        mode,
                    )
                _id = item_id - r[0]
                break
        if _id is None:
            raise ValueError(f"Item id {item_id} is out of range.")

        return self.env[env_idx][0].reset(_id)

    def _get_dataset_from_mode(self, mode: SqlGymMode) -> SqlGymEnv:
        if mode == "bird_train":
            bird_path = self._get_bird_path()
            return BirdDataset(bird_path, "train")
        elif mode == "bird_dev":
            bird_path = self._get_bird_path()
            return BirdDataset(bird_path, "dev")
        else:
            raise ValueError(f"Mode {mode} not supported")

    def _get_bird_path(self):
        bird_path = os.environ.get("AGENTENV_SQLGYM_BIRD_PATH", None)
        if bird_path is None:
            raise ValueError("Please set AGENTENV_SQLGYM_BIRD_PATH")
        return bird_path

    def _check_env_idx(self, env_idx):
        if env_idx not in self.env:
            raise IndexError(f"Env {env_idx} not found")
        if self.env[env_idx] is None:
            raise NotInitializedError(f"Env {env_idx} not initialized")


sqlgym_env_server = SqlGymEnvServer()
