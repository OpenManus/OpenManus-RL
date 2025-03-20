import json
import os

class ToolDataset:
    def __init__(self, test_file) -> None:
        super().__init__()
        self._load_data(test_file)

    def _load_data(self, test_file_path):
        
        data = None
        with open(test_file_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        self.goals = [ i["goal"] for i in data ] 
        
        if "answer" in data[0]["additional_info"]:
            self.ground_truths = [ i["additional_info"]["answer"] for i in data ]

        if "subgoals" in data[0]:
            self.ground_truth_subgoals = [ i["subgoals"] for i in data ]

        if "init_config" in data[0]["additional_info"]: 
            self.init_configs = [ i["additional_info"]["init_config"] for i in data ]
        
        if "goal_type" in data[0]["additional_info"]: 
            self.goal_types = [ i["additional_info"]["goal_type"] for i in data ]

        if "tool" in data[0]["additional_info"]: 
            self.tools = [ i["additional_info"]["tool"] for i in data ]
        
        if "difficulty" in data[0]:
            self.difficulties = [ i["difficulty"] for i in data ]
        
    def __len__(self):
        return len(self.goals)