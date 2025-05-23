from typing import Union


class AgentBenchException(Exception):
    pass


class ClientException(AgentBenchException):
    def __init__(self, reason: str, detail: Union[str, None] = None) -> None:
        super().__init__()
        self.reason = reason
        self.detail = detail

    def __str__(self) -> str:
        if not self.detail:
            return "{CLASS_NAME}[{REASON}]".format(
                CLASS_NAME=self.__class__.__name__, REASON=self.reason
            )
        else:
            return "{CLASS_NAME}[{REASON}]: {DETAIL}".format(
                CLASS_NAME=self.__class__.__name__,
                REASON=self.reason,
                DETAIL=self.detail,
            )


class ServerException(AgentBenchException):
    pass


class AgentClientException(ClientException):
    pass


class TaskClientException(ClientException):
    pass


class AgentContextLimitException(AgentClientException):
    def __init__(self, detail: Union[str, None] = None) -> None:
        super().__init__("agent_context_limit", detail)


class AgentTimeoutException(AgentClientException):
    def __init__(self, detail: Union[str, None] = None) -> None:
        super().__init__("agent_timeout", detail)


class AgentNetworkException(AgentClientException):
    def __init__(self, detail: Union[str, None] = None) -> None:
        super().__init__("agent_network", detail)


class TaskTimeoutException(TaskClientException):
    def __init__(self, detail: Union[str, None] = None) -> None:
        super().__init__("task_timeout", detail)


class TaskNetworkException(TaskClientException):
    def __init__(self, detail: Union[str, None] = None) -> None:
        super().__init__("task_network", detail)
