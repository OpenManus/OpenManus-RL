import subprocess
import os
import re
import json
import logging
from common.registry import registry
from utils.movie.movie_tools import movie_toolkits
from utils.tool.helpers import parse_action

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

@registry.register_environment("movie")
class MovieEnv:
    def __init__(self, dataset=None, use_dataset=True):
        super().__init__()
        self.action_path = []
        self.movie_toolkits = movie_toolkits()
        if not use_dataset:
            self.dataset = None
        else:
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
            action_space = [ item["name"] for item in json.load( open("{}/toolusage/prompts/Raw/movie_raw.json".format(os.environ["PROJECT_PATH"]), "r") )["tool_set_message"] ]
            return action_space
        else:
            raise NotImplemented("Action space with input is not implemented yet.")
            

    def is_done(self):
        return self.done

    def reset(self):
        if self.dataset is not None:
            self.goal = self.dataset["goal"]
            self.ground_truth = self.dataset["ground_truth"]
            self.ground_truth_subgoals = self.dataset["ground_truth_subgoals"]
            self.num_subgoals = len(self.ground_truth_subgoals)
        else:
            self.goal = None
            
        self.init_obs = "New trial starts. Once you have finished the goal, please remember to take 'finish' action to end this goal."
        # self.goal = None    # "goal" maybe the human instruction
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
        
        
        
    def update(self, action_path, observation, reward, done, info):        
        predicted_subgoals = [ item["Subgoal"]  for item in action_path]
        count = 0
        for subgoal in self.ground_truth_subgoals:
            if subgoal in predicted_subgoals:
                predicted_subgoals.remove(subgoal)
                count += 1
        reward = count / self.num_subgoals
        self.reward = max(self.reward, reward)
        
        self.done = done

    def get_unfinished(self):
        predicted_subgoals = [ item["Subgoal"]  for item in self.action_path]
        unfinished_subgoals = []
        for idx, subgoal in enumerate(self.ground_truth_subgoals):
            if subgoal not in predicted_subgoals:
                unfinished_subgoals.append( subgoal )
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
            if action_type == "get_search_movie":
                observation = self.movie_toolkits.get_search_movie(movie_name=params["movie_name"], action_path=action_path)
            elif action_type == "get_movie_details":
                observation = self.movie_toolkits.get_movie_details(movie_id=params["movie_id"], action_path=action_path)
            elif action_type == "get_movie_production_companies":
                observation = self.movie_toolkits.get_movie_production_companies(movie_id=params["movie_id"], action_path=action_path)
            elif action_type == "get_movie_production_countries":
                observation = self.movie_toolkits.get_movie_production_countries(movie_id=params["movie_id"], action_path=action_path)
            elif action_type == "get_movie_keywords":
                observation = self.movie_toolkits.get_movie_keywords(movie_id=params["movie_id"], action_path=action_path)
            elif action_type == "get_search_person":
                observation = self.movie_toolkits.get_search_person(person_name=params["person_name"], action_path=action_path)
            elif action_type == "get_person_details":
                observation = self.movie_toolkits.get_person_details(person_id=params["person_id"], action_path=action_path)
            elif action_type == "get_movie_cast":
                observation = self.movie_toolkits.get_movie_cast(movie_id=params["movie_id"], action_path=action_path)
            elif action_type == "get_movie_crew":
                observation = self.movie_toolkits.get_movie_crew(movie_id=params["movie_id"], action_path=action_path)
            elif action_type == "get_person_cast":
                observation = self.movie_toolkits.get_person_cast(person_id=params["person_id"], action_path=action_path)
            elif action_type == "get_person_crew":
                observation = self.movie_toolkits.get_person_crew(person_id=params["person_id"], action_path=action_path)
            elif action_type == "get_person_external_ids":
                observation = self.movie_toolkits.get_person_external_ids(person_id=params["person_id"], action_path=action_path)
            elif action_type == "get_movie_alternative_titles":
                observation = self.movie_toolkits.get_movie_alternative_titles(movie_id=params["movie_id"], action_path=action_path)
            elif action_type == "get_movie_translation":
                observation = self.movie_toolkits.get_movie_translation(movie_id=params["movie_id"], action_path=action_path)
            elif action_type == "finish":
                observation = self.movie_toolkits.finish(answer=params["answer"], action_path=action_path)
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
