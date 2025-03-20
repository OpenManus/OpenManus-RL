from typing import List, Optional

from pydantic import BaseModel


class StepQuery(BaseModel):
    env_idx: int
    action: str


class StepResponse(BaseModel):
    state: str
    reward: float
    done: bool
    info: None


class AvailableActionsResponse(BaseModel):
    has_search_bar: bool
    clickables: List[str]


class StateResponse(BaseModel):
    url: str
    html: str
    instruction_text: str


class ResetQuery(BaseModel):
    env_idx: int
    session_id: Optional[int] = None
