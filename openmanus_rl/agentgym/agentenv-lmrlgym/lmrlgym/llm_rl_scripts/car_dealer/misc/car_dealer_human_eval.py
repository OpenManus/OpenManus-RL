import os
import json
import time
import jax
import pickle as pkl
from llm_rl_scripts.car_dealer.env.policies import car_dealer_env, UserPolicy, eval
from LLM_RL.environment import text_env_eval

from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import create_path
from LLM_RL.utils import convert_path

INTRO_TEXT = """\
Welcome to the Car Salesman Bargaining Task. 
You will act as a seller, who wants to maximize the sell price of a car. 

Now that you know the rules, let's get started!
""".strip()

if __name__ == "__main__":
    YOUR_NAME = "" 
    buyer_model_path = ""
    N_INTERACTIONS = 5
    OUTPUTS_PATH = f"data/outputs/car_dealer/human_eval/user_interactions_test1_{YOUR_NAME}/"
    env_deterministic = False
    verbose = False
    print("Loading Environment")
    env = car_dealer_env(buyer_model_path)
    print("Loaded Environment")
    policy = UserPolicy()
    evaluation = eval(env, policy)

    def print_interaction(interaction):
        if interaction[-1].reward == 0:
            print('YOU WON!')
            print('='*25)
        else:
            print('YOU LOST :(')
            print('='*25)

    print()
    print('='*25)
    print(INTRO_TEXT)
    print('='*25)
    
    
    interation_raw_results, interaction_summary_results = text_env_eval(
        env=env,
        policy=policy,
        n_rollouts=N_INTERACTIONS,
        interaction_callback=print_interaction,
        env_options={"verbose": verbose}
    )

    print(interaction_summary_results)
    print('='*25)

    create_path(OUTPUTS_PATH)
    with open(os.path.join(OUTPUTS_PATH, 'interactions.pkl'), 'wb') as f:
        pkl.dump(interation_raw_results, f)
    with open(os.path.join(OUTPUTS_PATH, 'interactions_summary.json'), 'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: float(x), interaction_summary_results), f)


