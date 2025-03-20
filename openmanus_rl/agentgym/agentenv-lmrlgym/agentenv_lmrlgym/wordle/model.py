from pydantic import BaseModel


class WordleStepRequestBody(BaseModel):
    id: int
    action: str


class WordleResetRequestBody(BaseModel):
    id: int
    seed: int
