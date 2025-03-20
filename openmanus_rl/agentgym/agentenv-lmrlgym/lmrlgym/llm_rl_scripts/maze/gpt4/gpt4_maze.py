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

openai.api_key = os.getenv("OPENAI_API_KEY")

data_name = "gcs://rl-llm-bench-dataset-internal/maze/fully_observed_maze_data.jsonl"
# get dataset trajectory 
example_game = ""
num_trajectories = 0
with open(data_name, "r") as f:
    for item in f:
        obj = json.loads(item)
        if len(obj) < 3:
            continue
        num_trajectories += 1
        for i in range(min(3, len(obj))):
            example_game += "environment: " + obj[i]["state"] + "\n" + "action: " + obj[i]["action"]
        example_game += "------ \n new game \n ------\n"
        if num_trajectories == 1:
            break

SYSTEM_PROMPT = "You are an expert maze solver. You only respond in json."

MAIN_PROMPT = """\
Your objective is to reach the goal in as few steps as possible. At each step you will be given information about where the goal is, your current position,
and the walls that surround you. 

When you move right you increase your y position by 1, when you move down you increase your x position by 1. 

Here are some examples. """ + \
f""" 
```
{example_game}
```
""" + \
"""
Your possible actions are ["move up\n", "move up\n", "move left\n", "move right\n"].

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
        for i, item in enumerate(text_history):
            if i % 2 == 1:
                game_content += f"action: {item.text}"
            else:
                game_content += f"environment: {item.text}"
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
                    temperature=1.0,
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
    N_INTERACTIONS = 25
    OUTPUTS_PATH = "data/outputs/gpt4_maze/fully_observed"

    def text_history_to_str(text_history: TextHistory) -> str:
        return '\n'.join(map(lambda x: x.text, text_history))

    env = setup_maze_env(maze_name="double_t_maze", describe_function="describe_observation_give_position", reward_function="standard_reward", last_k=7)

    policy = GPT4MazePolicy()

    def print_interaction(interaction):
        print('='*25)
        print(text_history_to_str(interaction[-1].post_transition_history))
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


