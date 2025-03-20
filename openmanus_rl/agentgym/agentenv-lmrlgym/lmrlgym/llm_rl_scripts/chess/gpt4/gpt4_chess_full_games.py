import os
import json
import time
import openai
import jax
import pickle as pkl
from llm_rl_scripts.chess.env.data import get_data_from_bucket, get_random_positions_not_in_test
from llm_rl_scripts.chess.env.env import FenChessHistoryEnv, text_env_eval_chess_positions
from llm_rl_scripts.maze.env.maze_utils import setup_maze_env
from LLM_RL.environment import TextPolicy, TextHistory, Text, text_env_eval
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import create_path
from LLM_RL.utils import convert_path
from IPython import embed
import tiktoken

data_name = "gcs://rl-llm-bench-dataset/chess/complete_background_generated/train_trajectories.jsonl"
# get dataset trajectory 
example_game = ""
with open(data_name, "r") as f:
    for item in f:
        obj = json.loads(item)
        embed()
        for i in range(3):
            step = obj[i]
            example_game += f"environment: {step['state']}\n"
            example_game += f"action: {step['action']}\n"
        break

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = "You are a chess grandmaster. You only respond in json."

MAIN_PROMPT = """\
You are playing chess. The environment will give you a string in FEN notation and you will output the optimal action.

Here are some examples. 
```
{{example_game}}

```

Now let's start a new game. Return your action in a json array with a key "action", like in the example above. Now, make the best move given the current environment state:

```
{{game_content}}
```
""".strip()


TOKENIZER = tiktoken.encoding_for_model("gpt-4")
INPUT_TOKEN_COUNT = 0
OUTPUT_TOKEN_COUNT = 0

class GPT4EndgamesPolicy(TextPolicy):
    
    def __init__(self):
        self.prompt = MAIN_PROMPT

    def act(self, text_history: TextHistory) -> TextHistory:
        global INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT
        game_content = ""
        # for i, item in enumerate(text_history[1:]):
        #     if i % 2 == 0:
        #         game_content += f"action: {item.text}"
        #     else:
        #         game_content += f"environment: {item.text}"
        game_content = f"environment: {text_history[-1].text}"
        game_content = game_content.strip()
        prompt = self.prompt.replace('{{game_content}}', game_content)
        prompt = prompt.replace('{{example_game}}', example_game)
        print(prompt)
        print(text_history[-1].text)
        
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
    OUTPUTS_PATH = "data/outputs/gpt4_full_games/"

    def text_history_to_str(text_history: TextHistory) -> str:
        return '\n'.join(map(lambda x: x.text, text_history))

    # bucket_name = "rl-llm-bench-dataset"
    # blob_name = "endgames/test_positions.jsonl"
    # test_positions = get_data_from_bucket(bucket_name, blob_name)
    # test_positions = [position.replace("\n", "").replace("\"", "") for position in test_positions if position != ""]
    # test_positions = test_positions[:N_INTERACTIONS]
    policy = GPT4EndgamesPolicy()

    def print_interaction(interaction):
        print('='*25)
        print(text_history_to_str(interaction[-1].post_transition_history))
        print('='*25)
    
    env = FenChessHistoryEnv(max_moves=100)
    
    interaction_raw_results, interaction_summary_results = text_env_eval(
        env=env,
        policy=policy,
        n_rollouts=N_INTERACTIONS,
        interaction_callback=print_interaction,
    )

    print(interaction_summary_results)

    create_path(OUTPUTS_PATH)
    with open(os.path.join(OUTPUTS_PATH, 'interactions.pkl'), 'wb') as f:
        pkl.dump(interaction_raw_results, f)
    with open(os.path.join(OUTPUTS_PATH, 'interactions_summary.json'), 'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: float(x), interaction_summary_results), f)


