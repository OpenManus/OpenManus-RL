"""
TodoEnvServer
"""

from typing import Optional

from environment.todo_env import TodoEnv
from utils.tool.data_utils import ToolDataset
from utils.tool.helpers import extract_action_name_and_action_input
from copy import deepcopy
def clean_answer(answer):
    # remove all "id" in observation
    new_answer = deepcopy(answer)
    if isinstance(new_answer, list):
        for item in new_answer:
            item.pop("id")
    elif isinstance(new_answer, dict):
        new_answer.pop("id")
    return new_answer

def get_todo_status(tool):
    # get all projects
    _, projects = tool._get_all_projects()
    projects = clean_answer(projects)

    # get all tasks
    _, tasks = tool._get_all_tasks()
    tasks = clean_answer(tasks)

    result = {
        "projects": projects,
        "tasks": tasks
    }

    return result

class TodoEnvServer:
    """
    TodoEnvServer
    """

    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}
        dataset_path = "Toolusage/data/todo.jsonl"
        self.dataset = ToolDataset(test_file=dataset_path)

    def create(self, id: int = 0) -> int:
        env_idx = self._max_id
        dataset = self.dataset
        dataset_i = dict()
        dataset_i["goal"] = dataset.goals[id]
        dataset_i["ground_truth"] = dataset.ground_truths[id]
        dataset_i["ground_truth_subgoals"] = dataset.ground_truth_subgoals[id]
        dataset_i["tool"] = dataset.tools[id]

        self.env[self._max_id] = TodoEnv(dataset=dataset_i)
        self.env[self._max_id].todo_toolkits.current_date = self.dataset.init_configs[id]["current_date"]
        self.env[self._max_id].todo_toolkits.current_location = (
            self.dataset.init_configs[id]["current_location"]
        )
        self._max_id += 1
        return env_idx

    def reset(self, env_idx, id: Optional[int] = None):
        if id is not None:
            print(id)
            dataset = self.dataset
            dataset_i = dict()
            dataset_i["goal"] = dataset.goals[id]
            dataset_i["ground_truth"] = dataset.ground_truths[id]
            dataset_i["ground_truth_subgoals"] = dataset.ground_truth_subgoals[id]
            dataset_i["tool"] = dataset.tools[id]

            self.env[env_idx].__init__(dataset=dataset_i)
            self.env[env_idx].todo_toolkits.current_date = self.dataset.init_configs[
                id
            ]["current_date"]
            self.env[env_idx].todo_toolkits.current_location = (
                self.dataset.init_configs[id]["current_location"]
            )
        else:
            print(None)
            date = self.env[env_idx].todo_toolkits.current_date
            location = self.env[env_idx].todo_toolkits.current_date
            self.env[env_idx].__init__(dataset=self.env[env_idx].dataset)
            self.env[env_idx].todo_toolkits.current_date = date
            self.env[env_idx].todo_toolkits.current_location = location

    def step(self, env_idx, message: str):
        """
        observation, reward, done, None
        """
        action, action_input = extract_action_name_and_action_input(message)
        if action_input == None:
            observation, done = (
                'Format error, please response in the format of  "Action: [your action] with Action Input: [your action input]',
                False,
            )
            reward = self.env[env_idx].reward
            return observation, reward, done, None
        else:
            action_with_action_input = action + " with Action Input: " + action_input
            observation, reward, done, _ = self.env[env_idx].step(
                action=action_with_action_input
            )
            observation = "Observation: " + observation + "\nGive me one action."
            if done:
                current_stat = get_todo_status(self.env[env_idx].todo_toolkits)
                ground_stat = self.env[env_idx].dataset["ground_truth"]
                reward = (reward == 1.0) and (current_stat == ground_stat)
            return observation, reward, done, None

    def observation(self, env_idx):
        """
        Return:
            [{'id': '12345', 'order': 0, 'color': 'charcoal', 'name': 'School', 'is_favorite': false}]
        """
        if "New trial starts." in self.env[env_idx].get_obs():
            return (
                "Now new trial starts.\nYou should perform actions to accomplish the goal: "
                + self.env[env_idx].goal
                + "\nGive me one action."
            )
        return "Observation: " + self.env[env_idx].get_obs()


todo_env_server = TodoEnvServer()
