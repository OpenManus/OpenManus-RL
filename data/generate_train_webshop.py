import os
import json
import argparse
from datasets import Dataset
from tqdm import tqdm
from pprint import pprint

def load_items_human_ins(file_path):
    """
    Load the items_human_ins.json file
    
    Args:
        file_path: Path to the items_human_ins.json file
        
    Returns:
        Dictionary of items
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def make_map_fn(split):
    """
    Create a mapping function to convert webshop data to Verl format
    
    Args:
        split: Dataset split (train, val, test)
        
    Returns:
        Function to process each example
    """
    def process_fn(example, idx):
        # Extract the instruction as prompt
        prompt = example['instruction']
        
        # Combine attributes into a comma-separated string for ground truth
        attributes = example.get('attributes', [])
        ground_truth = ','.join(attributes) if attributes else None
        
        return {
            "data_source": "webshop",
            "prompt": prompt,
            "ability": "webshop-search",
            "reward_model": {
                "style": "ground_truth",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "id": example.get('asin', ''),
                "options": example.get('options', []),
                "instruction_attributes": example.get('instruction_attributes', []),
                "instruction_options": example.get('instruction_options', [])
            }
        }
    return process_fn

def flatten_data(items_data):
    """
    Flatten the nested dictionary structure to a list of examples
    
    Args:
        items_data: The loaded items_human_ins.json data
        
    Returns:
        Flattened list of examples
    """
    flattened = []
    for asin, instructions in items_data.items():
        for instruction in instructions:
            # Add asin to the instruction object if not present
            if 'asin' not in instruction:
                instruction['asin'] = asin
            flattened.append(instruction)
    return flattened

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="openmanus_rl/agentgym/agentenv-webshop/webshop/data/items_human_ins.json", 
                        help="Path to items_human_ins.json")
    parser.add_argument('--output_dir', required=False, default="data/webshop", help="Output directory for processed parquet")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--train_ratio', type=float, default=0.90, 
                        help="Ratio of data to use for training (rest for val/test)")
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help="Ratio of data to use for validation")

    args = parser.parse_args()

    # Load and flatten data
    print(f"Loading data from {args.input_file}...")
    items_data = load_items_human_ins(args.input_file)
    flattened_data = flatten_data(items_data)
    
    # Create dataset
    dataset = Dataset.from_list(flattened_data)
    
    # Split data
    if args.split == "all":
        # Process all data with the same split label
        dataset = dataset.map(function=make_map_fn(args.split), with_indices=True)
        # Save the entire dataset
        output_path = os.path.join(args.output_dir, f"{args.split}.parquet")
        dataset.to_parquet(output_path)
        print(f"Processed {len(dataset)} examples")
        print(f"Data saved to {output_path}")
    else:
        # Split the dataset
        splits = dataset.train_test_split(
            test_size=(1.0 - args.train_ratio),
            seed=42
        )
        
        # Further split the test set into validation and test
        if args.val_ratio > 0:
            # Calculate the ratio for the validation set from the remaining data
            remaining_ratio = 1.0 - args.train_ratio
            val_test_ratio = max(0.5, args.val_ratio / remaining_ratio)  # Ensure ratio is valid
            test_val_split = splits["test"].train_test_split(
                test_size=0.01,  # Split remaining data equally between val and test
                seed=42
            )
            splits = {
                "train": splits["train"],
                "validation": test_val_split["train"],
                "test": test_val_split["test"]
            }
            
            # Process and save all splits
            for split_name, split_dataset in splits.items():
                processed_dataset = split_dataset.map(function=make_map_fn(split_name), with_indices=True)
                output_path = os.path.join(args.output_dir, f"{split_name}.parquet")
                processed_dataset.to_parquet(output_path)
                print(f"Processed {len(processed_dataset)} examples for {split_name}")
                print(f"Data saved to {output_path}")
                
                # Print sample for the requested split
                if split_name == args.split:
                    dataset = processed_dataset
        else:
            # If no validation split is requested, just process train and test
            for split_name, split_dataset in splits.items():
                processed_dataset = split_dataset.map(function=make_map_fn(split_name), with_indices=True)
                output_path = os.path.join(args.output_dir, f"{split_name}.parquet")
                processed_dataset.to_parquet(output_path)
                print(f"Processed {len(processed_dataset)} examples for {split_name}")
                print(f"Data saved to {output_path}")
                
                # Set the dataset for the requested split
                if split_name == args.split:
                    dataset = processed_dataset

    # Print sample from the requested split
    print(f"\nSample from {args.split} split:")
    pprint(dataset[0]) 