# Extend AgentBench

[🌏中文版](Extension_cn.md)

## Task Introduction

The Task interface is defined as follows：
```
class Task:
    def __init__(self, name: str, concurrency: int = 1, *args, **kwargs):
        self.name = name
        self.concurrency = concurrency

    def get_indices(self) -> List[SampleIndex]:
        raise NotImplementedError()

    async def start_sample(
        self, index: SampleIndex, session: Session
    ) -> TaskSampleExecutionResult:
        raise NotImplementedError()

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        raise NotImplementedError()

    def release(self):
        pass
```

To implement your own Task, you just need to inherit from Task and implement the corresponding interfaces. The specific interfaces are described as follows:
- `name`: Task name, usually specified in the config
- `concurrency`: The maximum concurrency supported within a worker
- `get_indices`: Returns the indices of all samples
- `start_sample`: Logic within a single sample, where `index` is the index of the sample to be tested, and `session` is a proxy of the Agent.
- `calculate_overall`: Calculates the score after all samples have been tested; the return format is arbitrary and will eventually be saved to `overall.json`.
- `release`: Cleanup tasks that need to be executed after the task_worker process ends. Note that this is after the entire worker process ends, not after a particular sample ends.

The definition of the structures in the program is as follows:

```
SampleIndex = Union[int, str]
JSONSerializable = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]

class TaskSampleExecutionResult(BaseModel):
    status: SampleStatus = SampleStatus.COMPLETED
    result: JSONSerializable = None

class TaskOutput(BaseModel):
    index: Union[None, SampleIndex] = None
    status: SampleStatus = SampleStatus.RUNNING # directly from TaskSampleExecutionResult
    result: JSONSerializable = None # directly from TaskSampleExecutionResult
    history: Union[None, List[ChatHistoryItem]] = None

class SampleStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    AGENT_CONTEXT_LIMIT = "agent context limit"
    AGENT_VALIDATION_FAILED = "agent validation failed"
    AGENT_INVALID_ACTION = "agent invalid action"
    TASK_LIMIT_REACHED = "task limit reached"
    UNKNOWN = "unknown"
    TASK_ERROR = "task error"

class ChatHistoryItem(BaseModel):
    role: Literal["user", "agent"]
    content: str
```

Note that when returning `TaskSampleExecutionResult` in `start_sample`, you should carefully examine the completion status of the sample. If it is completed normally, it should be marked as `COMPLETED`. The relevant data of the completion status of the sample will be automatically counted by the framework.

The `Session` implements the following interfaces:
- `def inject(self, item: Union[ChatHistoryItem, List[ChatHistoryItem]])`: Inserts one or more historical records.
- `async def action(self, *injection) -> AgentOutput`: Waits for the Agent's response, and for convenience, it also supports inserting one or more historical records at this time.

The definition of `AgentOutput` is as follows:
```
class AgentOutput(BaseModel):
    status: AgentOutputStatus = AgentOutputStatus.NORMAL
    content: Union[str, None] = None

class AgentOutputStatus(str, Enum):
    NORMAL = "normal"
    CANCELLED = "cancelled"
    AGENT_CONTEXT_LIMIT = "agent context limit"
```

After obtaining `AgentOutput`, you need to handle it carefully and determine whether the `AgentOutputStatus` is normal. If it is not normal, corresponding processing is required. If the status is `CANCELLED`, it means that the client needs to cancel the test of this sample for some reason. At this time, you can quickly end this sample in any way to ensure that it does not affect subsequent tests.

## Implementation Example

A simple implementation is as follows:

```
class VirtualTask(Task):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(name="virtual-task", *args, **kwargs)

    def get_indices(self) -> List[Any]:
        return list(range(10))

    async def start_sample(self, index, session: Session):
        print("task start sample")
        for loop_times in range(3):
            await asyncio.sleep(1)
            res = await session.action(
                {"role": "user", "content": "Loop: %d" % loop_times}
            )
            print("TASK", res.content)
        return TaskSampleExecutionResult(
            status=SampleStatus.COMPLETED,
            result={"result": "ok"},
        )

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        return {"score": 0.4}
```

## Migrating from AgentBench v0.1

### Step 1: Migrate from `get_data` to `get_indices`

The data in the original `get_data` can be directly bound to `self` in the `__init__` method, and the corresponding data can be obtained from `self.data` in `start_sample` according to `index`. On this basis, implement `get_indices`. If the full set of test samples is a list, you can directly return `list(range(len(self.data)))`.

### Step 2: Change `predict_single` to `start_sample`

First, change the original `def` to `async def`. At the same time, change the original `session.action` to `await session.action`. Finally, the return value needs to set an additional `status` compared to the original. This helps to automatically count the reasons for sample errors, which is beneficial for further experimental analysis.

### Step 3: Change `metrics` to `calculate_overall`

This change was initially made to facilitate the counting of samples. If you don't want to change the original `metrics`, you can also create a new `calculate_overall` function and call `self.metrics` within the function.

### Additional Reminder

If you originally overrode the `predict_all` method, it cannot be used in the new framework.
