from JaxSeq.bucket_manager import open_with_bucket as open
import json
from LLM_RL.environment import Text, TextTrajectory, TextTrajectoryChain, TokenTrajectoryChain
from tqdm.auto import tqdm
import random
from transformers import GPT2TokenizerFast
from LLM_RL.algorithms.ilql.data import ILQLData

if __name__ == "__main__":
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    PATH = "put your path here"
    STEPS_BACK = 10 # maximum number of steps back to look
    EVAL_FRAC = 0.1 # fraction of data to use for evaluation

    all_items = []
    with open(PATH, "r") as f:
        for line in tqdm(f):
            all_items.append(json.loads(line))
    # create splits
    random.seed(0)
    random.shuffle(all_items)
    train_items = all_items[int(len(all_items)*EVAL_FRAC):]
    eval_items = all_items[:int(len(all_items)*EVAL_FRAC)]

    # code to load chains
    def chains_from_item(items):
        trajectories = []
        for i in range(1, len(item['text_history']), 2):
            text_trajectory = TextTrajectory(
                [
                    Text(''.join(item['text_history'][max(0, i-STEPS_BACK):i]), False),
                    Text(item['text_history'][i], True),
                ],
                [0.0, item['rewards'][i]],
                item['dones'][i],
            )
            trajectories.append(text_trajectory)
        
        chains = []
        curr_chain = None
        for i in range(len(trajectories)-1, -1, -1):
            curr_chain = TextTrajectoryChain(trajectories[i], curr_chain)
            if i == len(trajectories)-1:
                # handles special case of having None in last trajectory
                chains.append(TextTrajectoryChain(trajectories[i], TextTrajectoryChain(trajectories[i], None)))
            else:
                chains.append(curr_chain)
        return chains

    # load train / eval chains seperately
    train_chains = []
    for item in tqdm(train_items):
        train_chains.extend(chains_from_item(item))
    
    eval_chains = []
    for item in tqdm(eval_items):
        eval_chains.extend(chains_from_item(item))

    # use iterable class so that we can perform multiple epochs
    class ILQLDataIterable:
        def __init__(self, chains):
            self.chains = chains
        
        def __iter__(self):
            def ilql_data_generator(chains):
                for chain in chains:
                    token_trajectory_chain = TokenTrajectoryChain.from_text_trajectory_chain(chain, tokenizer)
                    yield ILQLData.from_token_trajectory_chain(token_trajectory_chain)
            # IMPORTANT: reshuffle data before each epoch to decorelate batch
            shuffle_idxs = list(range(len(self.chains)))
            random.shuffle(shuffle_idxs)
            shuffled_chains = [self.chains[i] for i in shuffle_idxs]
            return ilql_data_generator(shuffled_chains)
    
    # example of iterating through the data
    train_iterable = ILQLDataIterable(train_chains)
    for item in train_iterable:
        print(item)
        break
