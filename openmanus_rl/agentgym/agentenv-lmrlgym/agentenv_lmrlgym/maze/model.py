from pydantic import BaseModel


class MazeStepRequestBody(BaseModel):
    id: int
    action: str


class MazeResetRequestBody(BaseModel):
    id: int
    game: int
