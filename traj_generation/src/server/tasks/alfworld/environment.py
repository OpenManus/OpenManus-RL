import os
import sys
import json
import glob
import random
import numpy as np 

import textworld
import textworld.agents
import textworld.gym
import gym

from alfworld.agents.utils.misc import Demangler, get_templated_task_desc, add_task_to_grammar
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

class SingleAlfredTWEnv(AlfredTWEnv):
    '''
    Interface for Textworld Env 
    Contains only one game_file per environment
    '''

    def __init__(self, config, name, train_eval="eval_out_of_distribution"):
        print("Initializing AlfredTWEnv...")
        self.config = config
        self.train_eval = train_eval

        self.goal_desc_human_anns_prob = self.config['env']['goal_desc_human_anns_prob']
        self.get_game_logic()
        # self.gen_game_files(regen_game_files=self.config['env']['regen_game_files'])

        self.random_seed = 42

        self.game_files = [name]
        self.num_games = 1

def get_all_game_files(config, split="eval_out_of_distribution"):
    env = AlfredTWEnv(config, train_eval=split)
    game_files = env.game_files
    del env
    return game_files

# if __name__=="__main__":
#     os.environ["ALFWORLD_DATA"] = "/data/share/leixy/ReAct/alfworld/data"
#     config=load_config("/data/share/leixy/AgentBench/src/tasks/alfworld/configs/base_config.yaml")
#     game_files = get_all_game_files(config, "train")
#     game_files = [game.split("data/")[-1] for game in game_files]
#     with open("train.json", "w") as f:
#         f.write(json.dumps(game_files, indent=2))
#         f.close()
#     print(len(game_files))
#     print(game_files[0])
