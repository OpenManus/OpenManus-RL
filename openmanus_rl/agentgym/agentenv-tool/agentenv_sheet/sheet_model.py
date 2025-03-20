from typing import List, Optional

from pydantic import BaseModel


class CreateQuery(BaseModel):
    id: int


class StepQuery(BaseModel):
    env_idx: int
    action: str


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool


class ResetQuery(BaseModel):
    env_idx: int
    id: Optional[int] = None
