import os
import json
import time
import openai
import jax
import pickle as pkl
from llm_rl_scripts.wordle.env.env import WordleEnvironment, ReformatWordleEnvironment
from llm_rl_scripts.wordle.env.game import Vocabulary
from LLM_RL.environment import TextPolicy, TextHistory, Text, text_env_eval
from llm_rl_scripts.wordle.env.game import Vocabulary
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import create_path
from LLM_RL.utils import convert_path
import tiktoken

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = "You are an expert wordle player. You only respond in json."

MAIN_PROMPT = """\
Welcome to the game of Wordle. Your objective is to guess a hidden 5 letter word. You have 6 attempts to guess it correctly and you should try to guess it in as few attempts as possible. When guessing the word, you should format your word as a space separated sequence of letters, like "s h i r e" for example. After guessing the word, you will receive feedback from the game environment in the form of a sequence of 5 space separated letters like "b y g g b", where each letter indicates some information about the hidden word. The environment will return one of three letters – "b", "g", or "y" – for each letter in the word you guessed. We describe the meaning of each letter below:

"b": If the environment returns a "b", it means that the letter at that position in your guessed word is not in the hidden word.
"y": If the environment returns a "y", it means that the letter at that position in your guessed word is in the hidden word but is not in the correct position.
"g": If the environment returns a "g", it means that the letter at that position in your guessed word is in the hidden word and is in the correct position.

As a note, if you guess an invalid word (e.g. not a 5 letter word or a word not in the vocabulary), the environment will respond with an "invalid word" message. In general though, you should use this information returned by the environment to update your belief about what the hidden word might be and adjust your next guess accordingly.

Here is the complete list of valid vocabulary words that are accepted by the game:
```
{{vocab}}
```

Here is an example. If the current status of the game is given as:
```
guess 1: p a n i c
feedback 1: b b y b b
guess 2: f e l o n
feedback 2: g b b y g
```
Based on the feedback from the environment, you know that the first letter is "f", the last letter is "n", and there is an "o" somehwere in the word, but it is not in the second to last position. You also know that there is not a "p", "a", "i", "c", "e", or "l" in the word. Knowing this, you might guess the next word to be:
{"thought": "I know that the first letter is "f", the last letter is "n", and there is an "o" somehwere in the word, but it is not in the second to last position. I also know that there is not a "p", "a", "i", "c", "e", or "l" in the word. A good word from the vocabulary to try might therefore be \"f r o w n\", since it is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of "o". Therefore this is a good guess to try next.", "guess": "f r o w n"}

The guessed word is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of "o", so it is a good guess to try next.

Now let's start a new game. Return your word as a space separated sequence of 5 letters in a json array with key "thought" followed by key "guess", like in the example above. Now, guess the next word given the current game state:

```
{{game_content}}
```
""".strip()

VOCAB_FILE = "llm_rl_scripts/wordle/vocab/wordle_official_400.txt"

TOKENIZER = tiktoken.encoding_for_model("gpt-4")
INPUT_TOKEN_COUNT = 0
OUTPUT_TOKEN_COUNT = 0

class GPT4WordlePolicy(TextPolicy):
    def __init__(self, vocab: Vocabulary):
        vocab_text = '\n'.join(map(lambda x: ' '.join(list(x)), vocab.all_vocab))
        self.prompt = MAIN_PROMPT.replace("{{vocab}}", vocab_text)

    def act(self, text_history: TextHistory) -> TextHistory:
        global INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT
        game_content = ""
        for i, item in enumerate(text_history[1:]):
            if i % 2 == 0:
                game_content += f"guess {(i//2)+1}: {item.text}"
            else:
                if len(item.text.strip()) == 0:
                    game_content += f"feedback {(i//2)+1}: invalid word\n"
                else:
                    game_content += f"feedback {(i//2)+1}: {item.text}"
        game_content = game_content.strip()
        prompt = self.prompt.replace('{{game_content}}', game_content)
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
                    temperature=0.0,
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
        response_json = json.loads(response_text)
        print(f"total cost: {compute_cost(INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT)}; total input tokens: {INPUT_TOKEN_COUNT}; total output tokens: {OUTPUT_TOKEN_COUNT}")
        return text_history+(Text(response_json['guess'].strip(), True),)

def compute_cost(input_token_count: int, output_token_count: int) -> float:
    return ((0.03 * input_token_count) / 1000) + ((0.06 * output_token_count) / 1000)

if __name__ == "__main__":
    N_INTERACTIONS = 64
    OUTPUTS_PATH = "gcs://charlie-bucket2/LLM_RL_outputs/wordle/gpt4_eval/gpt4_test1_temp__/"

    def text_history_to_str(text_history: TextHistory) -> str:
        return '\n'.join(map(lambda x: x.text, text_history))

    vocab = Vocabulary.from_file(
        vocab_file=convert_path(VOCAB_FILE), 
        fill_cache=False, 
    )

    env = ReformatWordleEnvironment(WordleEnvironment(vocab))

    policy = GPT4WordlePolicy(vocab)

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



