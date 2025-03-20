from typing import List

from pydantic import BaseModel

from .general import SampleIndex
from .output import AgentOutput, TaskOutput


class RegisterRequest(BaseModel):
    name: str
    address: str
    concurrency: int
    indices: list


class StartSampleRequest(BaseModel):
    name: str
    index: SampleIndex


class InteractRequest(BaseModel):
    session_id: int
    agent_response: AgentOutput


class CancelRequest(BaseModel):
    session_id: int


class HeartbeatRequest(BaseModel):
    name: str
    address: str


class CalculateOverallRequest(BaseModel):
    name: str
    results: List[TaskOutput]


class WorkerStartSampleRequest(BaseModel):
    index: SampleIndex
    session_id: int


class SampleStatusRequest(BaseModel):
    session_id: int
