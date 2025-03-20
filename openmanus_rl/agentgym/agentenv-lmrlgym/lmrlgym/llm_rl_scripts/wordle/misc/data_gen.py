from JaxSeq.bucket_manager import open_with_bucket as open
from llm_rl_scripts.wordle.env.env import WordleEnvironment, reformat_history
from llm_rl_scripts.wordle.env.game import Vocabulary
from llm_rl_scripts.wordle.env.scripted_policies import RandomMixturePolicy
from llm_rl_scripts.wordle.env.data import PolicyDataGenerator
from tqdm.auto import tqdm
from typing import Any
import multiprocessing as mp
import tyro
import json
from LLM_RL.utils import convert_path
from functools import partial
from JaxSeq.utils import create_path
import os

class Worker:
    def __init__(self, vocab_path: str, prob_smart: float):
        self.vocab = Vocabulary.from_file(
            vocab_file=convert_path(vocab_path), 
            fill_cache=False, 
            rng=None, 
        )
        self.data_gen = PolicyDataGenerator(
            env=WordleEnvironment(self.vocab, require_words_in_vocab=True), 
            policy=RandomMixturePolicy(prob_smart=prob_smart, vocab=self.vocab), 
            seed=None, 
        )
    
    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data_gen)

worker = None
def worker_init(vocab_path: str, prob_smart: float):
    global worker
    worker = Worker(vocab_path, prob_smart)

def get_data_item(_) -> Any:
    return next(worker)

def main(
    n_data: int, 
    n_proc: int, 
    vocab_path: str, 
    prob_smart: float, 
    out_path: str, 
):
    data = []
    with mp.Pool(n_proc, initializer=partial(worker_init, vocab_path, prob_smart)) as pool:
        for item in tqdm(pool.imap_unordered(get_data_item, range(n_data)), total=n_data):
            data.append(
                dict(
                    sequence=[(text.text, float(text.is_action)) for text in reformat_history(item.text_history)], 
                    reward=item.reward, 
                    done=item.done, 
                )
            )
    
    create_path(os.path.dirname(convert_path(out_path)))
    with open(convert_path(out_path), 'w') as f:
        for item in data:
            f.write(json.dumps(item)+'\n')

if __name__ == "__main__":
    tyro.cli(main)
