import os
import json
from common.registry import registry
from utils.todo.todo_tools import todo_toolkits
from utils.tool.helpers import parse_action
import time


@registry.register_environment("todo")
class TodoEnv:
    def __init__(self, dataset):
        super().__init__()
        self.action_path = []

        for attempt in range(5):  
            try:
               self.todo_toolkits = todo_toolkits()
               break
            except Exception as e:
                print(f"Error on attempt {attempt + 1} to initialize todo_toolkits.")
                if attempt <  4:  # If not the last attempt
                    time.sleep(10)  # Wait before retrying 
                else:
                    print("Failed to initialize todo_toolkits after multiple attempts.")
                    raise e
        
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
            action_space = [ item["name"] for item in json.load( open("{}/toolusage/prompts/Raw/todo_raw.json".format(os.environ["PROJECT_PATH"]), "r") )["tool_set_message"] ]
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
            return  observation, self.reward, self.done, None

        try:
            if action_type == "get_user_current_date":
                observation = self.todo_toolkits.get_user_current_date(action_path=action_path)
            elif action_type == "get_user_current_location":
                observation = self.todo_toolkits.get_user_current_location(action_path=action_path)
            elif action_type == "get_projects":
                observation = self.todo_toolkits.get_projects(action_path=action_path)
            elif action_type == "update_project":
                observation = self.todo_toolkits.update_project(project_id=params["project_id"], is_favorite=params["is_favorite"], action_path=action_path)
            elif action_type == "get_tasks":
                observation = self.todo_toolkits.get_tasks(project_id=params["project_id"], action_path=action_path)
            elif action_type == "get_task_description":
                observation = self.todo_toolkits.get_task_description(task_id=params["task_id"], action_path=action_path)
            elif action_type == "get_task_duration":
                observation = self.todo_toolkits.get_task_duration(task_id=params["task_id"], action_path=action_path)
            elif action_type == "complete_task":
                observation = self.todo_toolkits.complete_task(task_id=params["task_id"], action_path=action_path)
            elif action_type == "update_task":
                observation = self.todo_toolkits.update_task(task_id=params["task_id"], due_date=params["due_date"], action_path=action_path)
            elif action_type == "delete_task":
                observation = self.todo_toolkits.delete_task(task_id=params["task_id"], action_path=action_path)
            elif action_type == "finish":
                observation = self.todo_toolkits.finish(answer=params["answer"], action_path=action_path)
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
