from examples_jaxseq.misc.commandline_server_client import Client
from JaxSeq.utils import strip_prompt_from_completion
from LLM_RL.environment import TextPolicy, TextHistory, text_history_to_str, Text, text_env_eval
from llm_rl_scripts.chess.env import FenChessHistoryEnvSingleTurn, postprocess_state
import tyro
from typing import Union, List, Optional
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import jsonl_load
from tqdm.auto import tqdm

class HostPolicy(TextPolicy):
    def __init__(self, host: Union[str, List[str]], n_retries: int=3, move: Optional[str]=None, eos_token_id: Optional[int]=None):
        if isinstance(host, str):
            host = [host]
        self.client = Client(host)
        self.n_retries = n_retries
        self.move = move
        self.eos_token_id = eos_token_id

    def act(self, text_history: TextHistory) -> TextHistory:
        if self.move is not None:
            return text_history+(Text(self.move, True),)
        prompt = text_history_to_str(text_history)
        for _ in range(self.n_retries):
            response = self.client.generate(
                prompts=[prompt], 
                seed=None, 
                max_input_length=128, 
                max_new_tokens=16, 
                do_sample=True, 
                num_beams=1, 
                temperature=None, 
                top_p=None, 
                top_k=None, 
                eos_token_id=self.eos_token_id, 
            )
            if response['status'] != 'error':
                break
        if response['status'] == 'error':
            raise Exception(f"Error: {response['error']}")
        move = strip_prompt_from_completion(prompt, response['data'][0])+'\n'
        return text_history + (Text(move, True),)

def main(
    host: str, 
    data_file: str, 
    max_iters: Optional[int], 
    n_retries: int=3, 
    eos_token_id: Optional[int]=None, 
):
    # policy = HostPolicy(host, n_retries=n_retries)

    with open(data_file, 'r') as f:
        data = jsonl_load(f)
    if max_iters is None:
        max_iters = len(data)
    
    wins, losses, draws = 0, 0, 0
    valid_moves, total_moves = 0, 0
    for i, d in tqdm(enumerate(data), total=max_iters):
        if i >= max_iters:
            break
        policy = HostPolicy(host, n_retries=n_retries, eos_token_id=eos_token_id)
        env = FenChessHistoryEnvSingleTurn(initial_history=(Text('', False),), from_position=postprocess_state(d['in_text']))
        all_data, summary = text_env_eval(
            env=env, 
            policy=policy, 
            n_rollouts=1, 
            initial_text_history=None, 
            seed_generator=None, 
            env_options=None, 
            interaction_callback=None, 
            bsize=1, 
            verbose=False, 
        )
        final_reward = all_data[0][-1].reward
        if final_reward == -1:
            losses += 1
        elif final_reward == 0:
            draws += 1
        elif final_reward == 1:
            wins += 1
        valid_moves += sum(list(map(lambda x: int(x.reward != -1), all_data[0][:-1])))
        total_moves += len(all_data[0])-1
        print(summary)
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}, Win Rate: {wins/(wins+losses+draws)}")
        print(f"Valid Moves: {valid_moves}, Total Moves: {total_moves}, Valid Move Rate: {valid_moves/total_moves}")

if __name__ == "__main__":
    tyro.cli(main)
