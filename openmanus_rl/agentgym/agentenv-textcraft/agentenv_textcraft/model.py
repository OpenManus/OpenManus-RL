from pydantic import BaseModel
from typing import Optional


class CreateRequestBody(BaseModel):
    commands: Optional[str] = None
    goal: Optional[str] = None


class StepRequestBody(BaseModel):
    id: int
    action: str


class ResetRequestBody(BaseModel):
    id: int
    data_idx: Optional[int] = 0
