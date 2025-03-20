"""
SheetEnvServer
"""

from typing import Optional

from environment.sheet_env import SheetEnv
from utils.tool.data_utils import ToolDataset
from utils.tool.helpers import extract_action_name_and_action_input
import time


class SheetEnvServer:
    """
    SheetEnvServer
    """

    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}
        dataset_path = "Toolusage/data/sheet.jsonl"
        self.dataset = ToolDataset(test_file=dataset_path)

    def create(self, id: int = 0) -> int:
        env_idx = self._max_id
        dataset = self.dataset
        dataset_i = dict()
        dataset_i["goal"] = dataset.goals[id]
        dataset_i["ground_truth"] = dataset.ground_truths[id]
        dataset_i["ground_truth_subgoals"] = dataset.ground_truth_subgoals[id]
        dataset_i["tool"] = dataset.tools[id]

        self.env[self._max_id] = SheetEnv(dataset=dataset_i)
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
        else:
            print(None)
            self.env[env_idx].__init__(dataset=self.env[env_idx].dataset)

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
            count = 0
            while count < 4:
                action_with_action_input = (
                    action + " with Action Input: " + action_input
                )
                observation, reward, done, _ = self.env[env_idx].step(
                    action=action_with_action_input
                )
                observation = "Observation: " + observation + "\nGive me one action."
                if "error" not in observation.lower():
                    break
                count += 1
                time.sleep(10)
            return observation, reward, done, None

    def observation(self, env_idx):
        """
        Return:
            [['Category', 'Brand', 'Size', 'Price', 'Stock', 'Price Update'], ['T-shirt', 'Nike', 'M', '30.00', '100', '20.00%'], ['Jeans', \"Levi's\", '32', '53.10', '150', '-10.00%']]
        """
        if "New trial starts." in self.env[env_idx].get_obs():
            return (
                "Now new trial starts.\nYou should perform actions to accomplish the goal: "
                + self.env[env_idx].goal
                + "\nGive me one action."
            )
        return "Observation: " + self.env[env_idx].get_obs()


sheet_env_server = SheetEnvServer()
