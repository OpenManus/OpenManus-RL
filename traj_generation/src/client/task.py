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
        self, name: str, controller_address: str = "http://localhost:5000/api", *_, **__,
    ) -> None:
        self.name = name
        self.controller_address = controller_address
        print("TaskClient created: {} ({})".format(name, controller_address))

    def get_indices(self) -> List[SampleIndex]:
        result = requests.get(
            self.controller_address + "/get_indices", params={"name": self.name}
        )
        if result.status_code != 200:
            raise AgentBenchException(result.text, result.status_code, self.name)
        return result.json()

    def get_concurrency(self) -> int:
        try:
            result = requests.get(
                self.controller_address + "/list_workers"
            )
        except Exception as e:
            print(ColorMessage.yellow(f"Warning task {self.name} cannot connect to controller {e}"))
            return 0
        if result.status_code != 200:
            raise AgentBenchException(result.text, result.status_code, self.name)
        result = result.json()
        if self.name not in result:
            print(ColorMessage.yellow(f"task {self.name} not found in worker list"))
            return 0
        concurrency = 0
        for worker in result[self.name]["workers"].values():
            if worker["status"] == WorkerStatus.ALIVE:
                concurrency += worker["capacity"] - worker["current"]
        return concurrency

    def run_sample(self, index: SampleIndex, agent: AgentClient) -> TaskClientOutput:
        try:
            result = requests.post(
                self.controller_address + "/start_sample",
                json=StartSampleRequest(name=self.name, index=index).dict(),
            )
        except Exception as e:
            return TaskClientOutput(error=TaskError.NETWORK_ERROR.value, info=str(e))
        if result.status_code == 406:
            return TaskClientOutput(
                error=TaskError.NOT_AVAILABLE.value, info=result.text
            )
        if result.status_code != 200:
            return TaskClientOutput(
                error=TaskError.START_FAILED.value, info=result.text
            )
        result = result.json()
        sid = result["session_id"]
        latest_result = result
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
                requests.post(
                    self.controller_address + "/cancel",
                    json=CancelRequest(session_id=sid).dict(),
                )
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
                    ).dict(),
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
                    json=CancelRequest(session_id=sid).dict(),
                )
                return TaskClientOutput(
                    error=TaskError.INTERACT_FAILED.value,
                    info=result.text,
                    output=latest_result,
                )

            result = result.json()
            latest_result = result
        # TODO: check this type and check where history is
        return TaskClientOutput(output=result["output"])

    def calculate_overall(self, results: List[TaskOutput]) -> JSONSerializable:
        statistics = {s: 0 for s in SampleStatus}
        for result in results:
            statistics[SampleStatus(result.status)] += 1
        for s in SampleStatus:
            statistics[s] /= len(results)
        statistics["average_history_length"] = sum(
            [len(result.history) for result in results]
        ) / len(results)
        statistics["max_history_length"] = max(
            [len(result.history) for result in results]
        )
        statistics["min_history_length"] = min(
            [len(result.history) for result in results]
        )
        ret = {
            "total": len(results),
            "validation": statistics,
        }
        res = requests.post(
            self.controller_address + "/calculate_overall",
            json=CalculateOverallRequest(name=self.name, results=results).dict(),
        )
        if res.status_code != 200:
            raise TaskNetworkException(res.text)
        ret["custom"] = res.json()
        return ret
