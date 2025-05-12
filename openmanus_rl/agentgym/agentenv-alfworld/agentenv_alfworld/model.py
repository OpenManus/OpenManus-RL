from pydantic import BaseModel


class StepRequestBody(BaseModel):
    id: int
    action: str


class ResetRequestBody(BaseModel):
    id: int
    game: int
    world_type: str
