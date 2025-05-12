from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class TodoEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are an autonomous intelligent agent. You can use actions to help people solve problems.\nWe detail name, description, input(parameters) and output(returns) of each action as follows:\nName: get_user_current_date()\nDescription: Get the user's current date.\nReturns:\nThe current date in 'YYYY-MM-DD' format.\n\nName: get_user_current_location()\nDescription: Get the user's current city.\nReturns:\nThe user's current city.\n\nName: get_projects()\nDescription: Get all projects in the Todoist account\nReturns:\n- Array of objects with properties:\n  - id (Type: string)\n  - name (Type: string)\n  - order (Type: integer)\n  - color (Type: string)\n  - is_favorite (Type: boolean)\n\nName: update_project(project_id, is_favorite)\nDescription: Update a project\nParameters:\n- project_id (Type: string)\n- is_favorite (Type: string, Enum: [True, False])\nReturns:\nInformation of the updated project\n\nName: get_tasks(project_id)\nDescription: Get all tasks for a given project\nParameters:\n- project_id (Type: string)\nReturns:\n- Array of objects with properties:\n  - id (Type: string)\n  - project_id (Type: \u201dstring)\n  - order (Type: integer)\n  - content (Type: string): Name of the task.\n  - is_completed (Type: boolean)\n  - priority (Type: integer): Task priority from 1 (normal) to 4 (urgent).\n  - due_date (Type: string): The due date of the task.\n\nName: get_task_description(task_id)\nDescription: Get the description of a specific task in the Todoist account.\nParameters:\n- task_id (Type: string)\nReturns:\n- id (Type: string): Unique identifier of the task.\n- content (Type: string): Name of the task.\n- description (Type: string): Description of the task. Incluing the Place, Tips, etc.\n\nName: get_task_duration(task_id)\nDescription: Get the duration of a specific task in the Todoist account.\nParameters:\n- task_id (Type: string)\nReturns:\n- id (Type: string)\n- content (Type: string): Name of the task.\n- duration (Type: string): Duration of the task in the format of 'amount(unit)'.\n\nName: complete_task(task_id)\nDescription: Mark a task as completed\nParameters:\n- task_id (Type: string)\nReturns:\ninformation of the completed task\n\nName: update_task(task_id, due_date)\nDescription: Update a task\nParameters:\n- task_id (Type: string)\n- due_date (Type: string)\nReturns:\nInformation of the updated task\n\nName: delete_task(task_id)\nDescription: Delete a specific task from the Todoist account.\nParameters:\n- task_id (Type: string): Unique identifier of the task to delete.\nReturns:\nInformation of the deleted task.\n\nName: check_valid_actions()\nDescription: Get supported actions for current tool.\nReturns:\nSupported actions for current tool.\n\nName: finish(answer)\nDescription: Call this action, when find the answer for the current task or complete essential operations.\nParameters:\n- answer (Type: ['string', 'number', 'array']): If the task is a question answering task, this is the answer to be returned. If the task is an operation task, the answer in 'done'\n\nIf you want to get the project_id or task_id, please call \"get_projects\" or \"get_tasks\". Do not generate it by yourself which maybe wrong. If you are finished, you will call \"finish\" action. \nPlease refer to the format of examples below to solve the requested goal. Please provide your thought to solve the question. You should give the thought with no more than 3 sentences. You need to give your thought together with your action!\n\nYour response must be in the format of:\nThought: [your thought]\n\nAction: [your action] with Action Input: [your action input]\n\nHere is an example:\n\nGoal: Is Prepare for history quiz a task of School project? Please answer yes or no.\nThought: To check whether Prepare for history quiz is a task of School project. I should first get the project id of School.\n\nAction: get_projects with Action Input: {}\nObservation: [{'id': '12345', 'order': 0, 'color': 'charcoal', 'name': 'School', 'is_favorite': false}]\nThought: From the result, the project id of School is 12345, then I can get all tasks of this project. And check whether Prepare for history quiz is in the result.\n\nAction: get_tasks with Action Input: {\"project_id\": \"12345\"}\nObservation: [{'id': '123451', 'order': 0, 'content': 'Prepare for history quiz', 'is_completed': false, 'priority': 1, 'due_date': '2030-10-10'}, {'id': '123452', 'order': 1, 'content': 'Prepare for math quiz', 'is_completed': false, 'priority': 1, 'due_date': '2030-11-10'}]\nThought: Prepare for history quiz is in the result, so it is a task of School project. I will call action: finish to terminate the conversation.\n\nAction: finish with Action Input: {\"answer\": \"yes\"}\nObservation: yes\n",
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.id = 0
        data = dict()
        data["id"] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> Dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> Dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response


class TodoTask(BaseTask):
    env_client_cls = TodoEnvClient
    env_name = "Todo"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
