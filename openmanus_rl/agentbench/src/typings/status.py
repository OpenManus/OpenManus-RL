from enum import IntEnum, Enum


class SampleStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    AGENT_CONTEXT_LIMIT = "agent context limit"
    AGENT_VALIDATION_FAILED = "agent validation failed"
    AGENT_INVALID_ACTION = "agent invalid action"
    TASK_LIMIT_REACHED = "task limit reached"
    UNKNOWN = "unknown"
    TASK_ERROR = "task error"


class WorkerStatus(IntEnum):
    ALIVE = 0
    COMA = 1
    DEAD = 2


class AgentOutputStatus(str, Enum):
    NORMAL = "normal"
    CANCELLED = "cancelled"
    AGENT_CONTEXT_LIMIT = "agent context limit"
