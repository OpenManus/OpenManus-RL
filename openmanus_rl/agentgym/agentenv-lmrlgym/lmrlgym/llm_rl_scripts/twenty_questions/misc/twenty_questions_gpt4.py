import os
import json
import time
import openai
import jax
import pickle as pkl
import pickle as pkl
from llm_rl_scripts.twenty_questions.env.policies import twenty_questions_env, GPT4TwentyQuestionsPolicy
from LLM_RL.environment import text_env_eval
from LLM_RL.environment import TextPolicy, TextHistory, Text, text_env_eval
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import create_path
from LLM_RL.utils import convert_path

MAIN_PROMPT = """\
Welcome to the game of Twenty Questions! Your objective is to guess what the object is within twenty questions. At every turn, you will have the oppurnity to ask a yes/no question, and receive an answer from the oracle. You can ask tweny questions but must ask as few questions as possible. 
```
""".strip()


if __name__ == "__main__":
    N_INTERACTIONS = 100
    OUTPUTS_PATH = f"gcs://rail-tpus-marwa/twenty_questions/LLM_RL_outputs/twenty_questions/gp4_eval/gpt4_test1_temp__/"

    def text_history_to_str(text_history: TextHistory) -> str:
        return '\n'.join(map(lambda x: x.text, text_history))

    print("Loading Environment")
    env = twenty_questions_env()
    print("Loaded Environment")
    policy = GPT4TwentyQuestionsPolicy(MAIN_PROMPT)

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
    with open(os.path.join(convert_path(OUTPUTS_PATH), 'interactions.pkl'), 'wb') as f:
        pkl.dump(interation_raw_results, f)
    with open(os.path.join(convert_path(OUTPUTS_PATH), 'interactions_summary.json'), 'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: float(x), interaction_summary_results), f)



