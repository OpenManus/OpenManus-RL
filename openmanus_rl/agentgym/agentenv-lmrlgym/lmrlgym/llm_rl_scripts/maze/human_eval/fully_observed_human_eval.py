import os
import json
import time
import jax
import pickle as pkl
from llm_rl_scripts.maze.env.maze_utils import setup_maze_env
from llm_rl_scripts.maze.env.policies import UserPolicy
from LLM_RL.environment import TextHistory, text_env_eval
from llm_rl_scripts.wordle.env.game import Vocabulary
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import create_path
from LLM_RL.utils import convert_path
from IPython import embed


YOUR_NAME = input("Enter your name: ").strip()

data_name = "gcs://rl-llm-bench-dataset-internal/maze/fully_observed_maze_data.jsonl"
# get dataset trajectory 
example_game = ""
with open(data_name, "r") as f:
    item = f.readline()
    obj = json.loads(item)
    example_game = obj[0]["state"] + "\n" + obj[0]["action"]

INTRO_TEXT = f"""\
Your objective is to reach the goal in as few steps as possible. At each step you will be given information about where the goal is, your current position,
and the walls that surround you. 

When you move right you increase your y position by 1, when you move down you increase your x position by 1. 

Here is an example. 
```
{example_game}

```

Your possible actions are ["move up\n", "move down\n", "move left\n", "move right\n"].

You can type 'w' to go up, 'd' to go right, 's' to go down, and 'a' to go left.

""".strip()



if __name__ == "__main__":
    N_INTERACTIONS = int(input("Enter number of trials: "))
    OUTPUTS_PATH = f"data/outputs/maze/human_eval/fully_observed/user_interactions_{YOUR_NAME}_test1_temp/"

    def text_history_to_str(text_history: TextHistory) -> str:
        return '\n'.join(map(lambda x: x.text, text_history))

    env = setup_maze_env(maze_name="double_t_maze", describe_function="describe_observation_give_position", reward_function="standard_reward", last_k=40)

    policy = UserPolicy()

    def print_interaction(interaction):
        if interaction[-1].reward == 0:
            print('YOU WON!')
        else:
            print('YOU LOST :(')

    print()
    print('='*25)
    print(INTRO_TEXT)
    print('='*25)
    
    interation_raw_results, interaction_summary_results = text_env_eval(
        env=env,
        policy=policy,
        n_rollouts=N_INTERACTIONS,
        interaction_callback=print_interaction,
    )

    print(interaction_summary_results)

    create_path(OUTPUTS_PATH)
    with open(os.path.join(OUTPUTS_PATH, 'interactions.pkl'), 'wb') as f:
        pkl.dump(interation_raw_results, f)
    with open(os.path.join(OUTPUTS_PATH, 'interactions_summary.json'), 'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: float(x), interaction_summary_results), f)



