import json
from JaxSeq.bucket_manager import open_with_bucket as open
from tqdm.auto import tqdm

if __name__ == "__main__":
    percentage = 0.1

    all_data = []
    with open('gcs://rail-tpus-csnell-us/LLM_RL_data/wordle/bc_data1.jsonl', 'r') as f:
        for item in tqdm(f):
            item = json.loads(item)
            all_data.append(item)
    
    all_data_filtered = sorted(all_data, key=lambda x: sum(x['reward']), reverse=True)[:int(len(all_data) * percentage)]

    with open(f'gcs://rail-tpus-csnell-us/LLM_RL_data/wordle/bc_data1_filtered_{str(percentage*100)}.jsonl', 'w') as f:
        for item in all_data_filtered:
            f.write(json.dumps(item)+'\n')
