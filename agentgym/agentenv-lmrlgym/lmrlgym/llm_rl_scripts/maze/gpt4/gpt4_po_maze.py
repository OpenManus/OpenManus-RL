import os
import json
import time
import openai
import jax
import pickle as pkl
from llm_rl_scripts.maze.env.maze_utils import setup_maze_env
from LLM_RL.environment import TextPolicy, TextHistory, Text, text_env_eval
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import create_path
from LLM_RL.utils import convert_path
from IPython import embed
import tiktoken
import numpy as np
from collections import defaultdict
from flax.traverse_util import flatten_dict, unflatten_dict

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = "You are an expert maze solver. You only respond in json."

data_name = "gcs://rl-llm-bench-dataset/maze/double_t_maze/double_t_maze_submazes_dialogue_history.jsonl"
# get dataset trajectory 
example_game = ""
with open(data_name, "r") as f:
    item = f.readline()
    obj = json.loads(item)
    example_game = " ".join(obj["text_history"])

MAIN_PROMPT = f"""\
Your objective is to reach the goal in as few steps as possible. At each step you will see your move history, and the walls that surround you.

Here are some examples. 
```
{example_game}

```
""" + \
"""
Your possible actions are "move up\n", "move up\n", "move left\n", "move right\n".

Now let's start a new game. Return your action in a json array with a key "action", like in the example above. Now, make the optimal action given the current environment state:

```
{{game_content}}
```
""".strip()


TOKENIZER = tiktoken.encoding_for_model("gpt-4")
INPUT_TOKEN_COUNT = 0
OUTPUT_TOKEN_COUNT = 0

class GPT4MazePolicy(TextPolicy):
    
    def __init__(self):
        self.prompt = MAIN_PROMPT

    def act(self, text_history: TextHistory) -> TextHistory:
        global INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT
        game_content = ""
        for item in text_history:
            game_content += f" {item.text} \n\n"
        game_content = game_content.strip()
        prompt = self.prompt.replace('{{game_content}}', game_content)
        print(prompt)
        INPUT_TOKEN_COUNT += len(TOKENIZER.encode(prompt))
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            except openai.OpenAIError as e:
                print(e)
                time.sleep(10)
                continue
            break
        response_text = response.choices[0].message.content
        OUTPUT_TOKEN_COUNT += len(TOKENIZER.encode(response_text))
        print(response_text)
        try:
            response_json = json.loads(response_text)
        except: 
            response_json = {"action": ""}
        print(f"total cost: {compute_cost(INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT)}; total input tokens: {INPUT_TOKEN_COUNT}; total output tokens: {OUTPUT_TOKEN_COUNT}")
        return text_history+(Text(response_json['action'].strip() + "\n", True),)
    
    
def compute_cost(input_token_count: int, output_token_count: int) -> float:
    return ((0.03 * input_token_count) / 1000) + ((0.06 * output_token_count) / 1000)

if __name__ == "__main__":
    OUTPUTS_PATH = "data/outputs/gpt4_maze/partially_observed/"

    def text_history_to_str(text_history: TextHistory) -> str:
        return '\n'.join(map(lambda x: x.text, text_history))

    env = setup_maze_env(maze_name="double_t_maze", 
                         describe_function="describe_observation_only_walls", 
                         reward_function="standard_reward", 
                         last_k=20, 
                         max_steps=100)

    policy = GPT4MazePolicy()

    def print_interaction(interaction):
        print('='*25)
        print(text_history_to_str(interaction[-1].post_transition_history))
        print('='*25)
        
    possible_positions = list(zip(*np.where(env.maze==0)))
    for goal in env.valid_goals:
        possible_positions.remove(tuple(goal.tolist()))


    interactions = dict()
    results = dict()
    avg_dict = defaultdict(float)
    for position in possible_positions:
        position = tuple(position)
        interactions[str(position)], results[str(position)] = text_env_eval(
            env=env,
            policy=policy,
            n_rollouts=1, 
            verbose=True,
            env_options={"init_position": position},
            bsize=1,
        )
        for k, v in flatten_dict(results[str(position)]).items():
            avg_dict[k] += v
    for k, v in avg_dict.items():
        avg_dict[k] = v/len(possible_positions)
    results["avg_reward"] = unflatten_dict(dict(avg_dict))



    print(results)

    create_path(OUTPUTS_PATH)
    with open(os.path.join(OUTPUTS_PATH, 'interactions.pkl'), 'wb') as f:
        pkl.dump(interactions, f)
    with open(os.path.join(OUTPUTS_PATH, 'interactions_summary.json'), 'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: float(x), results), f)


