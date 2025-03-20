import os
import json
import time
import jax
import pickle as pkl
from llm_rl_scripts.wordle.env.env import WordleEnvironment
from llm_rl_scripts.wordle.env.game import Vocabulary
from llm_rl_scripts.wordle.env.scripted_policies import UserPolicy
from LLM_RL.environment import TextHistory, text_env_eval
from llm_rl_scripts.wordle.env.game import Vocabulary
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import create_path
from LLM_RL.utils import convert_path

VOCAB_FILE = "llm_rl_scripts/wordle/vocab/wordle_official_400.txt"

INTRO_TEXT = """\
Welcome to the game of Wordle!
Your objective is to guess a hidden 5 letter word.
You have 6 attempts to guess it correctly and you should try to guess it in as few attempts as possible.
After guessing the word, you will receive feedback from the game environment in the form of a color-coded version of your word. The color given for each letter means the following:

black/white (default text color): If the environment returns a black or white letter (depending on your system setup), it means that the letter at that position in your guessed word is not in the hidden word.
yellow: If the environment returns a yellow letter, it means that the letter at that position in your guessed word is in the hidden word but is not in the correct position.
green: If the environment returns a green letter, it means that the letter at that position in your guessed word is in the hidden word and is in the correct position.

As a note, if you guess an invalid word (e.g. not a 5 letter word or a word not in the vocabulary), the environment will respond with an "invalid word" message.
For your reference, here is the complete list of valid vocabulary words that are accepted by the game:

=====================
{{vocab}}
=====================

Now that you know the rules, let's get started!
""".strip()

INTRO_TEXT_WITHOUT_VOCAB = """\
Welcome to the game of Wordle!
Your objective is to guess a hidden 5 letter word.
You have 6 attempts to guess it correctly and you should try to guess it in as few attempts as possible.
After guessing the word, you will receive feedback from the game environment in the form of a color-coded version of your word. The color given for each letter means the following:

black/white (default text color): If the environment returns a black or white letter (depending on your system setup), it means that the letter at that position in your guessed word is not in the hidden word.
yellow: If the environment returns a yellow letter, it means that the letter at that position in your guessed word is in the hidden word but is not in the correct position.
green: If the environment returns a green letter, it means that the letter at that position in your guessed word is in the hidden word and is in the correct position.

As a note, if you guess an invalid word (e.g. not a 5 letter word or a word not in the vocabulary), the environment will respond with an "invalid word" message.

For your reference, the list of valid vocab words is at the path `llm_rl_scripts/wordle/vocab/wordle_official_400.txt`. Feel free to reference this list when playing the game.

Now that you know the rules, let's get started!
""".strip()

if __name__ == "__main__":
    YOUR_NAME = input("Enter your name: ").strip()
    N_INTERACTIONS = int(input("Enter number of trials: "))
    OUTPUTS_PATH = f"gcs://rail-tpus-csnell-us/LLM_RL_outputs/wordle/user_interactions_test1_{YOUR_NAME}/"

    def text_history_to_str(text_history: TextHistory) -> str:
        return '\n'.join(map(lambda x: x.text, text_history))

    vocab = Vocabulary.from_file(
        vocab_file=convert_path(VOCAB_FILE), 
        fill_cache=False, 
    )

    env = WordleEnvironment(vocab)

    policy = UserPolicy(vocab)

    def print_interaction(interaction):
        if interaction[-1].reward == 0:
            print('YOU WON!')
        else:
            print('YOU LOST :(')

    print()
    print('='*25)
    # vocab_text = '\n'.join(vocab.all_vocab)
    # print(INTRO_TEXT.replace('{{vocab}}', vocab_text))
    print(INTRO_TEXT_WITHOUT_VOCAB)
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



