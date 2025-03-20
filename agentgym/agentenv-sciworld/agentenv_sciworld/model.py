from pydantic import BaseModel
from typing import Optional


class StepRequestBody(BaseModel):
    id: int
    action: str


class ResetRequestBody(BaseModel):
    id: int
    data_idx: int
