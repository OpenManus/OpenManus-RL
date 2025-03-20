import pdb
import subprocess
import os
import re
import json
import logging
from common.registry import registry
from utils.sheet.sheets_tools import sheet_toolkits
from utils.tool.helpers import parse_action

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# extract current sheet id and open sheet first
def extract_sheet_name(s):
    match = re.search(r'"Sheet(\d{1,2})"', s)
    if match:
        return "Sheet" + match.group(1)
    else:
        return None

@registry.register_environment("sheet")
class SheetEnv:
    def __init__(self, dataset):
        super().__init__()
        self.action_path = []
        self.sheet_toolkits = sheet_toolkits()
        self.sheet_toolkits.init_sheets()
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

    def get_action_space(self):
        action_space = [item["name"] for item in json.load(
            open("{}/toolusage/prompts/Raw/sheet_raw.json".format(os.environ["PROJECT_PATH"]), "r"))[
            "tool_set_message"]]
        return action_space

    def is_done(self):
        return self.done
    
    def get_match_score(self, sheet1, sheet2): # only check content
        if sheet1 == 'Please open a sheet before operating it':
            return 0.0

        # Get minimum number of rows and columns for both sheets
        min_rows = min(len(sheet1), len(sheet2))
        min_cols = min(len(sheet1[0]) if sheet1 else 0, len(sheet2[0]) if sheet2 else 0)

        # Count of matching cells
        matching_count = 0
        total_cells = 0

        for i in range(min_rows):
            for j in range(min_cols):
                total_cells += 1
                if sheet1[i][j] == sheet2[i][j]:
                    matching_count += 1

        matching_degree = matching_count / total_cells if total_cells else 0

        logger.info("count: {}".format(matching_count))
        logger.info(f"Matching Degree: {matching_degree * 100:.2f}%")

        return matching_degree
        # [To-Do] only check format
        # [To-Do] only check format and content

    def clean_up_text(self, text):
        cleaned_text = re.sub(r'\n', '', text).replace('>', '')
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
        return cleaned_text

    def reset(self):
        if self.dataset is not None:
            self.goal = self.dataset["goal"]
            self.sheet_name = extract_sheet_name(self.goal)
            self.ground_truth = self.dataset["ground_truth"]
        else:
            self.goal = None

        self.init_obs = "New trial starts. Once you have finished the goal, please remember to take 'finish' action to end this goal."

        self.infos = dict()
        self.states = [self.init_obs]   # record a stream of states
        self.history = [("state", self.init_obs)]   # record a stream of s0, a0, r0, s1, a1, r1, ...
        self.steps = 0

        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]

        self.reward = 0
        self.done = False

    # for each question: 1.open sheet first 2.provide valid function[load from prompt file]
    def reset_question_env(self, question, sheet_name):
        _, _, _, _ = self.step(f"Action: open_sheet with Action Input:{{'name': '{sheet_name}'}}")

        self.states.append(f"You have open the {sheet_name}")
        self.history.append( ("state", self.states[-1]) )

        action = f"open_sheet with Action Input: {{'name': '{sheet_name}'}}"
        observation = f"You are now in {sheet_name}\n" + str(self.observation_space)

        # since self.step() call update() function, we need to reset reward and done
        self.reward = 0
        self.done = False

        return action, observation

    def update(self, observation_space, done, info):
        reward = self.get_match_score(observation_space, json.loads(self.ground_truth))
        self.reward = max(self.reward, reward)
        self.done = done

    def get_unfinished(self):
        return None

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
            if action_type == "open_sheet":
                observation = self.sheet_toolkits.open_sheet(name=params["name"], action_path=action_path)
            elif action_type == "del_sheet":
                observation = self.sheet_toolkits.del_sheet(name=params["name"], action_path=action_path)
            elif action_type == "freeze_data":
                observation = self.sheet_toolkits.freeze_data(dimension=params["dimension"], num=params["num"],
                                                              action_path=action_path)
            elif action_type == "get_A1_annotation":
                observation = self.sheet_toolkits.get_A1_annotation(row=params["row"], col=params["col"],
                                                                    action_path=action_path)
            elif action_type == "insert_cols":
                observation = self.sheet_toolkits.insert_cols(values_list=params["values_list"],
                                                              col_idx=params["col_idx"], action_path=action_path)
            elif action_type == "insert_rows":
                observation = self.sheet_toolkits.insert_rows(values_list=params["values_list"],
                                                              row_idx=params["row_idx"], action_path=action_path)
            elif action_type == "delete_batch_data":
                observation = self.sheet_toolkits.delete_batch_data(dimension=params["dimension"],
                                                                    index_list=params["index_list"],
                                                                    action_path=action_path)
            elif action_type == "update_cell":
                observation = self.sheet_toolkits.update_cell(position=params["position"], value=params["value"],
                                                              action_path=action_path)
            elif action_type == "update_cell_by_formula":
                if "position_list" in params:
                    observation = self.sheet_toolkits.update_cell_by_formula(position_list=params["position_list"],
                                                                           operator=params["operator"],
                                                                           result_position=params["result_position"],
                                                                           action_path=action_path)
                else:
                    observation = self.sheet_toolkits.update_cell_by_formula(start_position=params["start_position"],
                                                                           end_position=params["end_position"],
                                                                           result_position=params["result_position"],
                                                                           operator=params["operator"],
                                                                           action_path=action_path)
            elif action_type == "update_range":
                observation = self.sheet_toolkits.update_range(start_position=params["start_position"],
                                                               end_position=params["end_position"],
                                                               values_list=params["values_list"],
                                                               action_path=action_path)
            elif action_type == "sort_sheet_by_col":
                observation = self.sheet_toolkits.sort_sheet_by_col(col_num=params["col_num"],
                                                                    order=params["order"],
                                                                    action_path=action_path)
            elif action_type == "merge_cells":
                observation = self.sheet_toolkits.merge_cells(start_position=params["start_position"],
                                                              end_position=params["end_position"],
                                                              action_path=action_path)
            elif action_type == "update_note":
                observation = self.sheet_toolkits.update_note(position=params["position"],
                                                              content=params["content"],
                                                              action_path=action_path)
            elif action_type == "get_all_values":
                observation = self.sheet_toolkits.get_all_values(action_path=action_path)
            elif action_type == "update_note":
                observation = self.sheet_toolkits.update_note(position=params["position"],
                                                              content=params["content"],
                                                              action_path=action_path)
            elif action_type == "get_range_values":
                observation = self.sheet_toolkits.get_range_values(start_position=params["start_position"],
                                                                   end_position=params["end_position"],
                                                                   action_path=action_path)
            elif action_type == "get_cell_value":
                observation = self.sheet_toolkits.get_cell_value(position=params["position"],
                                                                 action_path=action_path)
            elif action_type == "get_value_by_formula":
                if "position_list" in params:
                    observation = self.sheet_toolkits.get_value_by_formula(position_list=params["position_list"],
                                                                           operator=params["operator"],
                                                                           action_path=action_path)
                else:
                    observation = self.sheet_toolkits.get_value_by_formula(start_position=params["start_position"],
                                                                           end_position=params["end_position"],
                                                                           operator=params["operator"],
                                                                           action_path=action_path)
            elif action_type == "filter_cells":
                observation = self.sheet_toolkits.filter_cells(query=params.get("query"),
                                                               in_row=params.get("in_row"),
                                                               in_column=params.get("in_column"),
                                                               action_path=action_path)
            elif action_type == "get_note":
                observation = self.sheet_toolkits.get_note(position=params["position"],
                                                           action_path=action_path)
            elif action_type == "finish":
                observation = self.sheet_toolkits.finish(action_path=action_path)

            elif action_type == "get_observation_space":
                observation = ""  # just get observation space for calculate
            else:
                observation = "ERROR | Invalid action: {}. Please choose another valid action from {}".format(action_type,
                                                                                                      self.get_action_space())
        except Exception as e:
            observation_space = self.sheet_toolkits.get_all_values()[1]
            self.observation_space = observation_space
            observation = "ERROR | " + str(type(e).__name__) + "(" + str(e) + ")"
            self.done = False

            if self.dataset is not None and "Invalid" not in str(observation):
                self.update(observation_space, self.done, None)

            return str(observation), self.reward, self.done, None

        done = "finish" in action.lower()
        self.done = done

        observation_space = self.sheet_toolkits.get_all_values()[1]
        self.observation_space = observation_space

        if self.dataset is not None and "Invalid" not in str(observation):
            self.update(observation_space, done, None)

        return str(observation), self.reward, self.done, None

    @classmethod
    def from_config(cls, cfg):
        env = cls(dataset=cfg.get("dataset"))

        return env
