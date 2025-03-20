import os
from llm_rl_scripts.chess.env.data import get_saved_text_chains
import random 
import json
from tqdm.auto import tqdm
import numpy as np

def reformat_chains_to_bc_dataset(chains, rounds):
    states = []
    actions = []
    rewards = []
    dones = []
    data = []
    for idx, chain in enumerate(chains):
        state = chain.text_trajectory.text_history[0].text
        action = chain.text_trajectory.text_history[1].text
        reward = chain.text_trajectory.reward[1]
        done = chain.text_trajectory.done
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        sample = {"from_state": state, "action": action, "reward": reward, "done": done, "generated_by": "ppo", "round": rounds[idx]}
        data.append(sample)
    random.shuffle(data)
    return states, actions, rewards, dones, data

def save_bc_dataset(states, actions, rewards, dones, data, data_path):
    np.save(os.path.join(data_path, "states.npy"), states)
    np.save(os.path.join(data_path, "actions.npy"), actions)
    np.save(os.path.join(data_path, "reward.npy"), rewards)
    np.save(os.path.join(data_path, "done.npy"), dones)
    with open(os.path.join(data_path, "states.jsonl"), "w") as f:
        for state in tqdm(states):
            f.write(json.dumps(state) + "\n")
    with open(os.path.join(data_path, "actions.jsonl"), "w") as f:
        for action in tqdm(actions):
            f.write(json.dumps(action) + "\n")
    with open(os.path.join(data_path, "reward.jsonl"), "w") as f:
        f.write(json.dumps(rewards) + "\n")
    with open(os.path.join(data_path, "done.jsonl"), "w") as f:
        f.write(json.dumps(dones) + "\n")
    with open(os.path.join(data_path, "data.jsonl"), "w") as f:
        for item in tqdm(data):
            f.write(json.dumps(item) + "\n")

def save_chains_as_bc_dataset(chains, rounds, data_path):
    states, actions, rewards, dones, data = reformat_chains_to_bc_dataset(chains, rounds)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    save_bc_dataset(states, actions, rewards, dones, data, data_path)
bucket_name = "rail-tpus-isadora"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/nfs/nfs1/users/isadoracw/rail-tpus.json"
os.environ["GCLOUD_PROJECT"] = "rail-tpus"
path = "llm-rl-outputs/outputs/chess/ppo_online_endgames_lr1e-5_bsize256_64roll_4pos/ppo_online_endgames_lr1e-5_bsize256_64roll_4pos.2023-06-11-19-41-19.979.f04a0c5e088f11ee8b308de166d61c57/"
path = "llm-rl-outputs/outputs/chess/ppo_online_endgames_lr1e-5_bsize256_4roll_64pos/ppo_online_endgames_lr1e-5_bsize256_4roll_64pos.2023-06-07-19-29-12.870.953f0450056911eeadcca9664a75c8d1/"
data_path = os.path.join("/nfs/nfs1/users/isadoracw/ILQL5/src/environments/chess/ppo_dataset_4roll_64pos/")
chains, rounds = get_saved_text_chains(bucket_name, path)
for idx in range(10):
    print(chains[idx].text_trajectory)
save_chains_as_bc_dataset(chains, rounds, data_path)