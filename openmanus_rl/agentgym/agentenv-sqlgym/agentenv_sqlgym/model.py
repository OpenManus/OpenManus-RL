from typing import Any, List, Optional

from pydantic import BaseModel


class StepQuery(BaseModel):
    env_idx: int
    action: str


class StepResponse(BaseModel):
    state: str
    reward: float
    done: bool
    info: Any


class ResetQuery(BaseModel):
    env_idx: int
    item_id: int
