import textworld
from llm_rl_scripts.text_nav.env import TextNavEnv
from llm_rl_scripts.text_nav.env.env import play

num_games = int(input("Enter number of games: "))
observability = input("Would you like to play fully observed? Else defaulting to partially observed. (Y/N): ").strip()
display_bool = (observability == "Y")
print("You have selected a " + ("fully" if display_bool else "partially") + " observed game!\n")

env = TextNavEnv(
    display_location=display_bool, display_inventory=display_bool)
env.render()

from IPython.display import clear_output
import time

total_reward = 0
for _ in range(num_games):
    total_reward += play(env)
    time.sleep(1)
    clear_output(wait=True)

print('Average score over {} games: {}'.format(num_games, total_reward / num_games))
