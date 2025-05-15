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
    parser.add_argument('--valid_ratio', type=float, default=0.1, help="Ratio for validation split (default: 0.1)")
    args = parser.parse_args()
    
    print(f"Loading dataset...")
    # Load from Hugging Face Hub - full dataset
    dataset = load_dataset("CharlieDreemur/OpenManus-RL", split="train")
    
    print(f"Splitting dataset into train/valid with {args.valid_ratio*100}% validation...")
    # Split into train and validation
    splits = dataset.train_test_split(test_size=args.valid_ratio, seed=42)
    train_dataset = splits["train"]
    valid_dataset = splits["test"]  # The test_split method names the second split "test"
    
    print(f"Train set: {len(train_dataset)} examples")
    print(f"Validation set: {len(valid_dataset)} examples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process and save train set
    print("Processing train set...")
    processed_train = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    processed_train.to_parquet(train_path)
    print(f"Saved train set to {train_path}")
    
    # Process and save validation set
    print("Processing validation set...")
    processed_valid = valid_dataset.map(function=make_map_fn("valid"), with_indices=True)
    valid_path = os.path.join(args.output_dir, "valid.parquet")
    processed_valid.to_parquet(valid_path)
    print(f"Saved validation set to {valid_path}")
    
    # Show sample from each set
    print("\nSample from train set:")
    pprint(processed_train[0])
    print("\nSample from validation set:")
    pprint(processed_valid[0])