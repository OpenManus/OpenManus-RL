import subprocess
import os
import re
import json
import logging
from common.registry import registry
from utils.weather.weather_tools import weather_toolkits
from utils.tool.helpers import parse_action, is_same_location

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

@registry.register_environment("weather")
class WeatherEnv:
    def __init__(self, dataset):
        super().__init__()
        self.action_path = []
        self.weather_toolkits = weather_toolkits()

        self.dataset = dataset
        self.reset()


    def get_info(self):
        return self.infos

    def get_obs(self):
        return self.states[-1]

    def get_goal(self):
        return self.goal

    def get_history(self):
        return self.history
        
    def get_action_space(self, with_input=False):
        if not with_input:
            action_space = [ item["name"] for item in json.load( open("{}/toolusage/prompts/Raw/weather_raw.json".format(os.environ["PROJECT_PATH"]), "r") )["tool_set_message"] ]
            return action_space
        else:
            raise NotImplemented("Action space with input is not implemented yet.")
                

    def is_done(self):
        return self.done

    def reset(self):
        self.weather_toolkits.current_date = self.dataset["current_date"]
        self.weather_toolkits.current_location = self.dataset["current_location"]
        self.goal = self.dataset["goal"]
        self.ground_truth = self.dataset["ground_truth"]
        self.ground_truth_subgoals = self.dataset["ground_truth_subgoals"]
        self.num_subgoals = len(self.ground_truth_subgoals)
        
        self.init_obs = "New trial starts. Once you have finished the goal, please remember to take 'finish' action to end this goal."
        self.infos = dict()

        self.states = [self.init_obs]  # record a stream of states
        self.history = [("state", self.init_obs)]   # record a stream of s0, a0, r0, s1, a1, r1, ...
        self.steps = 0

        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]

        self.reward = 0
        self.done = False


    def check_equality(self, subgoal, predicted_subgoal):
        subgoal = subgoal.copy()
        predicted_subgoal = predicted_subgoal.copy()

        coord1 = (subgoal["latitude"], subgoal["longitude"])
        coord2 = (predicted_subgoal["latitude"], predicted_subgoal["longitude"])

        subgoal.pop("latitude")
        subgoal.pop("longitude")
        predicted_subgoal.pop("latitude")
        predicted_subgoal.pop("longitude")

        if is_same_location(coord1, coord2) and subgoal == predicted_subgoal:
            return True
        else:
            return False
        
    def update(self, action_path, observation, reward, done, info):
        predicted_subgoals = [ item["Subgoal"]  for item in action_path]
        count = 0

        for subgoal in self.ground_truth_subgoals: 
            # Special case for latitude and longitude
            if type(subgoal) == dict and "latitude" in subgoal and 'longitude' in subgoal:
                for predicted_subgoal in predicted_subgoals:
                    if type(predicted_subgoal) == dict and "latitude" in predicted_subgoal and 'longitude' in predicted_subgoal:
                        if self.check_equality(subgoal, predicted_subgoal):
                            predicted_subgoals.remove(predicted_subgoal)
                            count += 1
                            break

            elif subgoal in predicted_subgoals:
                predicted_subgoals.remove(subgoal)
                count += 1
        reward = count / self.num_subgoals
        self.reward = max(self.reward, reward)
        
        self.done = done
    
    def get_unfinished(self):
        predicted_subgoals = [ item["Subgoal"]  for item in self.action_path]
        unfinished_subgoals = []
        for idx, subgoal in enumerate(self.ground_truth_subgoals):
            tag = False
            if type(subgoal) == dict and "latitude" in subgoal and 'longitude' in subgoal:
                for predicted_subgoal in predicted_subgoals:
                    if type(predicted_subgoal) == dict and "latitude" in predicted_subgoal and 'longitude' in predicted_subgoal:
                        if self.check_equality(subgoal, predicted_subgoal):
                            tag = True
                            break

            elif subgoal  in predicted_subgoals:
                tag = True
            if not tag:
                unfinished_subgoals.append(subgoal)
        return unfinished_subgoals
    
    def step(self, action, action_path=None):
        if action_path == None:
            action_path = self.action_path
        try:
            action_type, params = parse_action(action)
        except Exception as e:
            observation = "ERROR | " + type(e).__name__ + "(" + str(e) + ")"
            done = False
            return observation, self.reward, self.done, None

        try:
            if action_type == "get_user_current_date":
                observation = self.weather_toolkits.get_user_current_date(action_path=action_path)
            elif action_type == "get_user_current_location":
                observation = self.weather_toolkits.get_user_current_location(action_path=action_path)
            elif action_type == "get_historical_temp":
                observation = self.weather_toolkits.get_historical_temp(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    action_path=action_path
                )
            elif action_type == "get_historical_rain":
                observation = self.weather_toolkits.get_historical_rain(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    action_path=action_path
                )
            elif action_type == "get_historical_snow":
                observation = self.weather_toolkits.get_historical_snow(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    action_path=action_path
                )
            elif action_type == "get_snow_forecast":
                observation = self.weather_toolkits.get_snow_forecast(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    action_path=action_path
                )
            elif action_type == "get_current_snow":
                observation = self.weather_toolkits.get_current_snow(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    current_date=params["current_date"],
                    action_path=action_path
                )
            elif action_type == "get_current_temp":
                observation = self.weather_toolkits.get_current_temp(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    current_date=params["current_date"],
                    action_path=action_path
                )
            elif action_type == "get_latitude_longitude":
                observation = self.weather_toolkits.get_latitude_longitude(
                    name=params["name"],
                    action_path=action_path
                )
            # elif action_type == "get_air_quality":
            #     observation = self.weather_toolkits.get_air_quality(
            #         latitude=params["latitude"],
            #         longitude=params["longitude"],
            #         action_path=action_path
            #     )
            elif action_type == "get_elevation":
                observation = self.weather_toolkits.get_elevation(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    action_path=action_path
                )
            elif action_type == "get_temp_forecast":
                observation = self.weather_toolkits.get_temp_forecast(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    action_path=action_path
                )
            elif action_type == "get_rain_forecast":
                observation = self.weather_toolkits.get_rain_forecast(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    action_path=action_path
                )
            elif action_type == "get_current_rain":
                observation = self.weather_toolkits.get_current_rain(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    current_date=params["current_date"],
                    action_path=action_path
                )
            elif action_type == "get_distance":
                observation = self.weather_toolkits.get_distance(
                    latitude1=params["latitude1"],
                    longitude1=params["longitude1"],
                    latitude2=params["latitude2"],
                    longitude2=params["longitude2"],
                    action_path=action_path
                )
            elif action_type == "get_historical_air_quality_index":
                observation = self.weather_toolkits.get_historical_air_quality_index(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    action_path=action_path
                )
            elif action_type == "get_current_air_quality_index":
                observation = self.weather_toolkits.get_current_air_quality_index(
                    latitude=params["latitude"],
                    longitude=params["longitude"],
                    current_date=params["current_date"],
                    action_path=action_path
                )
            elif action_type == "get_air_quality_level":
                observation = self.weather_toolkits.get_air_quality_level(
                    air_quality_index=params["air_quality_index"],
                    action_path=action_path
                )
            elif action_type == "convert_zipcode_to_address":
                observation = self.weather_toolkits.convert_zipcode_to_address(
                    zipcode=params["zipcode"],
                    action_path=action_path
                )
            elif action_type == "finish":
                observation = self.weather_toolkits.finish(answer=params["answer"], action_path=action_path)
            elif action_type == "check_valid_actions":
                observation = "You can use following valid actions: {}".format(self.get_action_space(with_input=False))
            else:
                observation = "ERROR | Invalid action: {}.".format(action_type)
        except Exception as e:
            observation = "ERROR | " + type(e).__name__ + "(" + str(e) + ")"
            done = False
            return  observation, self.reward, self.done, None

        done = "Finish" in action or "finish" in action
        self.done = done 

        if self.dataset is not None and "Invalid" not in str(observation):
            self.update(action_path, observation, self.reward, done, None)

        return str(observation), self.reward, self.done, None

    @classmethod
    def from_config(cls, cfg):
        env = cls(dataset = cfg.get("dataset"))

        return env