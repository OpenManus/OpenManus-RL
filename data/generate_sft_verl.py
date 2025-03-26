import os
import argparse
from datasets import Dataset, load_dataset
from tqdm import tqdm
from pprint import pprint

def make_map_fn(split):
    def process_fn(example, idx):
        return {
            "data_source": "openmanus-rl",
            "prompt": example['conversations'],
            "ability": "instruction-following",
            "reward_model": {
                "style": "none",
                "ground_truth": None
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "id": example['id']
            }
        }
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help="Output directory for processed parquet")
    parser.add_argument('--split', type=str, default="train")

    args = parser.parse_args()

    # Load from Hugging Face Hub
    dataset = load_dataset("CharlieDreemur/OpenManus-RL", split=args.split)

    # Apply mapping to Verl format
    dataset = dataset.map(function=make_map_fn(args.split), with_indices=True)

    # Pretty preview the first sample 
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.to_parquet(os.path.join(args.output_dir, f"{args.split}.parquet"))

    pprint(dataset[0])
