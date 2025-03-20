"""
WeatherEnvServer
"""

from typing import Optional

from environment.weather_env import WeatherEnv
from utils.tool.data_utils import ToolDataset
from utils.tool.helpers import extract_action_name_and_action_input


class WeatherEnvServer:
    """
    WeatherEnvServer
    """

    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}
        dataset_path = "Toolusage/data/weather.jsonl"
        self.dataset = ToolDataset(test_file=dataset_path)

    def create(self, id: int = 0) -> int:
        env_idx = self._max_id
        dataset = self.dataset
        dataset_i = dict()
        dataset_i["goal"] = dataset.goals[id]
        dataset_i["ground_truth"] = dataset.ground_truths[id]
        dataset_i["ground_truth_subgoals"] = dataset.ground_truth_subgoals[id]
        dataset_i["tool"] = dataset.tools[id]
        dataset_i["current_date"] = dataset.init_configs[id]["current_date"]
        dataset_i["current_location"] = dataset.init_configs[id]["current_location"]

        self.env[self._max_id] = WeatherEnv(dataset=dataset_i)
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
            dataset_i["current_date"] = dataset.init_configs[id]["current_date"]
            dataset_i["current_location"] = dataset.init_configs[id]["current_location"]

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
            if reward > 0:
                reward = 1.0
            return observation, reward, done, None
        else:
            action_with_action_input = action + " with Action Input: " + action_input
            observation, reward, done, _ = self.env[env_idx].step(
                action=action_with_action_input
            )
            observation = "Observation: " + observation + "\nGive me one action."
            if reward > 0:
                reward = 1.0
            return observation, reward, done, None

    def observation(self, env_idx):
        """
        Return:
            {'results': [{'name': 'Shanghai', 'latitude': 31.22222, 'longitude': 121.45806, 'country_code': 'CN'}, {'name': 'Shanghai', 'latitude': 34.85009, 'longitude': -87.08501, 'country_code': 'US'}, {'name': 'Cornelia', 'latitude': 38.64363, 'longitude': -93.73938, 'country_code': 'US'}]}
            {'latitude': 31.200005, 'longitude': 121.5, 'daily_units': {'time': 'iso8601', 'temperature_2m_max': '\u00b0C', 'temperature_2m_min': '\u00b0C', 'temperature_2m_mean': '\u00b0C'}, 'daily': {'time': ['2015-01-01'], 'temperature_2m_max': [4.3], 'temperature_2m_min': [-3.6], 'temperature_2m_mean': [-0.1]}}
        """
        if "New trial starts." in self.env[env_idx].get_obs():
            return (
                "Now new trial starts.\nYou should perform actions to accomplish the goal: "
                + self.env[env_idx].goal
                + "\nGive me one action."
            )
        return "Observation: " + self.env[env_idx].get_obs()


weather_env_server = WeatherEnvServer()
