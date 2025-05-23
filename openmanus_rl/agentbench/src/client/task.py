import enum

import requests

from src.typings import *
from src.utils import *
from .agent import AgentClient


class TaskError(enum.Enum):
    START_FAILED = "START_FAILED"
    INTERACT_FAILED = "INTERACT_FAILED"
    AGENT_FAILED = "AGENT_FAILED"
    NETWORK_ERROR = "NETWORK_ERROR"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class TaskClient:
    def __init__(
        self, name: str, controller_address: str, **configs
    ) -> None:
        self.controller_address = controller_address
        self.name = name
        self.configs = configs
        print("TaskClient created: {} ({})".format(name, controller_address))

    def get_indices(self) -> List[SampleIndex]:
        try:
            result = requests.get(
                self.controller_address + "/get_indices",
                params={"name": self.name},
            )
            if result.status_code != 200:
                return []
            return result.json()["indices"]
        except Exception:
            return []

    def get_concurrency(self) -> int:
        try:
            # Try the list_workers endpoint instead, which seems to be available
            result = requests.get(
                self.controller_address + "/list_workers"
            )
            if result.status_code != 200:
                print(f"Warning: Could not get concurrency for task {self.name}: {result.status_code} {result.text}")
                return 1
            return result.json()["concurrency"]
        except Exception:
            return 1

    def run_sample(self, index: SampleIndex, agent: AgentClient) -> TaskClientOutput:
        try:
            result = requests.post(
                self.controller_address + "/start",
                json=StartSampleRequest(name=self.name, index=index).model_dump(),
            )
            if result.status_code == 404:
                return TaskClientOutput(error=TaskError.NOT_AVAILABLE.value)
            if result.status_code != 200:
                return TaskClientOutput(
                    error=TaskError.START_FAILED.value, info=result.text
                )
            result = result.json()
            sid = result["session_id"]
            latest_result = None
        except Exception as e:
            return TaskClientOutput(
                error=TaskError.NETWORK_ERROR.value, info=str(e)
            )

        while SampleStatus(result["output"]["status"]) == SampleStatus.RUNNING:
            try:
                content = agent.inference(result["output"]["history"])
                response = AgentOutput(content=content)
            except AgentContextLimitException:
                response = AgentOutput(status=AgentOutputStatus.AGENT_CONTEXT_LIMIT)
            except Exception as e:
                if hasattr(agent, "model_name"):
                    model_name = agent.model_name
                elif hasattr(agent, "name"):
                    model_name = agent.name
                else:
                    model_name = agent.__class__.__name__
                print(f"ERROR: {model_name}/{self.name} agent error", e)
                try:
                    # Ensure we have a valid latest_result with required fields
                    if latest_result is None:
                        latest_result = TaskOutput(
                            status=SampleStatus.UNKNOWN,
                            index=index,
                            result=None,
                            history=[]
                        )
                    requests.post(
                        self.controller_address + "/cancel",
                        json=CancelRequest(session_id=sid).model_dump(),
                    )
                except Exception:
                    pass  # Ignore errors in cancel request
                return TaskClientOutput(
                    error=TaskError.AGENT_FAILED.value,
                    info=str(e),
                    output=latest_result,
                )

            try:
                result = requests.post(
                    self.controller_address + "/interact",
                    json=InteractRequest(
                        session_id=sid,
                        agent_response=response,
                    ).model_dump(),
                )
            except Exception as e:
                return TaskClientOutput(
                    error=TaskError.NETWORK_ERROR.value,
                    info=str(e),
                    output=latest_result,
                )
            
            if result.status_code != 200:
                requests.post(
                    self.controller_address + "/cancel",
                    json=CancelRequest(session_id=sid).model_dump(),
                )
                return TaskClientOutput(
                    error=TaskError.INTERACT_FAILED.value,
                    info=result.text,
                    output=latest_result,
                )

            try:
                result = result.json()
                latest_result = result["output"]
            except Exception as e:
                # Handle malformed JSON response
                return TaskClientOutput(
                    error=TaskError.INTERACT_FAILED.value,
                    info=f"Failed to parse response: {str(e)}",
                    output=latest_result,
                )
        # TODO: check this type and check where history is
        return TaskClientOutput(output=result["output"])

    def calculate_overall(self, results: List[TaskOutput]) -> JSONSerializable:
        statistics = {s: 0 for s in SampleStatus}
        for result in results:
            statistics[SampleStatus(result.status)] += 1
        for s in SampleStatus:
            statistics[s] /= len(results)
        statistics["average_history_length"] = sum(
            [len(result.history) for result in results if result.history is not None]
        ) / len(results)
        statistics["max_history_length"] = max(
            [len(result.history) for result in results if result.history is not None] or [0]
        )
        statistics["min_history_length"] = min(
            [len(result.history) for result in results if result.history is not None] or [0]
        )
        ret = {
            "total": len(results),
            "validation": statistics,
        }
        try:
            res = requests.post(
                self.controller_address + "/calculate_overall",
                json=CalculateOverallRequest(name=self.name, results=results).model_dump(),
            )
            if res.status_code != 200:
                raise TaskNetworkException(res.text)
            ret["custom"] = res.json()
        except Exception as e:
            print(f"Warning: Failed to calculate custom metrics for {self.name}: {str(e)}")
            ret["custom"] = {"error": str(e)}
        return ret
