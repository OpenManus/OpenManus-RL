import numpy as np
import os
import random
import json
from collections import defaultdict
from tqdm.auto import tqdm

initial_state = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

if __name__ == "__main__":
    states = np.load(open(os.path.join('elo-2800-dataset', 'states.npy'), 'rb'))
    actions = np.load(open(os.path.join('elo-2800-dataset', 'actions.npy'), 'rb'))
    
    states = np.concatenate((np.full((states.shape[0], 1), initial_state), states), axis=1)

    data = []
    for i in tqdm(range(actions.shape[0])):
        for x in range(actions.shape[1]):
            if actions[i, x] == '':
                assert states[i, x+1] == ''
                continue
            data.append({
                "in_text": " ".join(states[i, x])+"\n", 
                "out_text": " ".join(actions[i, x])+"\n", 
            })

    # get all stockfish actions at each state
    actions_per_state = defaultdict(set)
    for item in tqdm(data):
        actions_per_state[item['in_text']].add(item['out_text'])
    for item in tqdm(data):
        item['stockfish_actions'] = list(actions_per_state[item['in_text']])
    
    # split data by state
    data_per_state = defaultdict(list)
    for item in tqdm(data):
        data_per_state[item['in_text']].append(item)
    all_states = list(data_per_state.keys())
    # shuffle
    random.shuffle(all_states)

    # 90% of states for training data
    train_data = []
    for i in tqdm(range(int(len(all_states)*0.9))):
        train_data.extend(data_per_state[all_states[i]])
    val_data = []
    for i in tqdm(range(int(len(all_states)*0.9), len(all_states))):
        val_data.append(data_per_state[all_states[i]][0])

    with open('train.jsonl', 'w') as f:
        for d in tqdm(train_data):
            f.write(json.dumps(d)+"\n")
    with open('val.jsonl', 'w') as f:
        for d in tqdm(val_data):
            f.write(json.dumps(d)+"\n")
