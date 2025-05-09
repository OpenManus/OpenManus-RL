from datasets import load_dataset
import re
import os
import pandas as pd
from collections import defaultdict
import numpy as np

# Define environments to extract with correct naming
ENVIRONMENTS = [
    "alfworld", "babyai", "maze", "wordle", 
    "sciworld", "sqlgym", "textcraft", "movie", 
    "todo", "weather", "webshop", "webarena"
]

# Environment ID mapping (map item_id prefixes to standard environment names)
ENV_ID_MAPPING = {
    "alfworld": ["alfworld", "alworld"],  # Fix potential typo in original code
    "babyai": ["babyai"],
    "maze": ["lmrlgym_maze", "maze"],
    "wordle": ["lmrlgum_wordle", "wordle"],
    "sciworld": ["sciworld"],
    "sqlgym": ["sqlgym"],
    "textcraft": ["textcraft"],
    "movie": ["movie"],
    "todo": ["todo"],
    "weather": ["weather"],
    "webshop": ["webshop"],
    "webarena": ["webarena"]
}

def make_prefix(question, environment):
    """
    Create instruction prefix for the OpenManus agent.
    
    Args:
        question: The question or task to be solved
        environment: The environment type
        
    Returns:
        Formatted prefix with OpenManus template
    """
    prefix = f"""You are an OpenManus agent tasked to solve the following problem.
You must conduct reasoning inside <think> and </think> tags first every time you get new information.
With the <think> tags, you aslo need to have a <memory> </memory> tag to record your memory, like update your memory with new information and summarize your memory with new experiences, 
and you also need to have a <plan> </plan> tag to record your plan, like update your plan with new information and summarize your plan with new experiences.
After reasoning, you can perform actions using <act> action_description </act> tags.
When you have a final answer, provide it inside <answer> and </answer> tags, without detailed illustrations.

Task: {question}\n"""
    return prefix

def extract_solution(solution_str):
    """
    Extract numerical solution from a string.
    
    Args:
        solution_str: String containing the solution
        
    Returns:
        Extracted numerical value
    """
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

def process_group_data(group_name, group_samples):
    """
    Process samples for a specific environment group.
    
    Args:
        group_name: Name of the environment group
        group_samples: List of samples belonging to this group
        
    Returns:
        List of processed data samples
    """
    processed_data = []
    
    for idx, sample in enumerate(group_samples):
        item_id = sample['item_id']
        conversations = sample['conversations']
        
        # Process each conversation
        dialog_data = []
        for conversation in conversations:
            dialog_data.append({
                "from": conversation['from'],
                "value": conversation['value'],
                "loss": conversation['loss']
            })
        
        # Extract question/task from the first user message
        user_messages = [conv['value'] for conv in conversations if conv['from'] == 'human']
        question = user_messages[0] if user_messages else "No question found"
        
        # Create formatted prompt
        formatted_question = make_prefix(question, group_name)
        
        # Build final data structure
        data = {
            "data_source": group_name,  # Use environment name as data source
            "item_id": item_id,
            "conversations": dialog_data,
            "prompt": [{
                "role": "user",
                "content": formatted_question,
            }],
            "ability": "agent-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "environment": group_name,
                    "task_id": item_id
                }
            },
            "extra_info": {
                'split': group_name,
                'index': idx,
            }
        }
        processed_data.append(data)
    
    return processed_data

def group_samples_by_environment(data, env_mapping):
    """
    Group data samples by their environment based on item_id.
    
    Args:
        data: Dataset samples
        env_mapping: Dictionary mapping environment names to potential ID prefixes
        
    Returns:
        Dictionary with environment names as keys and sample lists as values
    """
    env_groups = defaultdict(list)
    
    for sample in data:
        item_id = sample['item_id']
        
        # Check which environment this sample belongs to
        matched = False
        for env_name, prefixes in env_mapping.items():
            for prefix in prefixes:
                if prefix in item_id:
                    env_groups[env_name].append(sample)
                    matched = True
                    break
            if matched:
                break
        
        # If not matched to any known environment, use a fallback
        if not matched:
            print(f"Warning: Could not match sample with item_id '{item_id}' to any environment")
    
    return env_groups

def split_data(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        samples: List of data samples
        train_ratio: Ratio of training samples (default 0.8)
        val_ratio: Ratio of validation samples (default 0.1)
        test_ratio: Ratio of test samples (default 0.1)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'validation', 'test' keys containing corresponding samples
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle indices
    indices = np.random.permutation(len(samples))
    
    # Calculate split sizes
    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create splits
    splits = {
        'train': [samples[i] for i in train_indices],
        'validation': [samples[i] for i in val_indices],
        'test': [samples[i] for i in test_indices],
    }
    
    return splits

def save_environment_data(env_groups, output_base_dir):
    """
    Save environment data to separate directories with train/test/validation splits.
    
    Args:
        env_groups: Dictionary with environment name as key and samples as value
        output_base_dir: Base directory where environment subdirectories will be created
    """
    # Ensure base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each environment group
    for env_name, samples in env_groups.items():
        if not samples:
            print(f"Warning: No samples found for environment '{env_name}'. Skipping.")
            continue
            
        print(f"Processing environment: {env_name} with {len(samples)} samples")
        
        # Create environment subdirectory
        env_dir = os.path.join(output_base_dir, env_name)
        os.makedirs(env_dir, exist_ok=True)
        
        # Process samples for this environment
        processed_samples = process_group_data(env_name, samples)
        
        # Split data into train/validation/test sets
        if len(processed_samples) < 3:
            print(f"Warning: Only {len(processed_samples)} samples for {env_name}, using all for train")
            splits = {
                'train': processed_samples,
                'validation': processed_samples[:1],  # Use first sample for both val and test
                'test': processed_samples[:1]         # if there's only one or two samples
            }
        else:
            # Adjust split ratios for very small datasets
            if len(processed_samples) < 10:
                # For small datasets, ensure at least 1 sample in each split
                train_ratio = max(0.6, 1 - 2/len(processed_samples))
                val_ratio = test_ratio = (1 - train_ratio) / 2
                print(f"Adjusted split ratios for small dataset: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
            else:
                train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
                
            splits = split_data(
                processed_samples, 
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
        
        # Save each split
        for split_name, split_samples in splits.items():
            if not split_samples:
                print(f"Warning: No samples in {split_name} split for {env_name}")
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(split_samples)
            
            # Define output filename based on split
            if split_name == 'validation':
                output_file = os.path.join(env_dir, "val.parquet")
            else:
                output_file = os.path.join(env_dir, f"{split_name}.parquet")
                
            # Save to parquet
            df.to_parquet(output_file)
            print(f"Saved {len(split_samples)} samples to {output_file}")

def main():
    """
    Main function to process and save AgentGym dataset by environment.
    """
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("AgentGym/AgentTraj-L")
    data = dataset['train']
    
    # Group samples by environment using the ID mapping
    print("Grouping samples by environment...")
    env_groups = group_samples_by_environment(data, ENV_ID_MAPPING)
    
    # Print group statistics
    for env, samples in env_groups.items():
        print(f"Environment: {env}, Number of samples: {len(samples)}")
    
    # Save environment data to appropriate directories
    print("Saving environment data with train/val/test splits...")
    save_environment_data(env_groups, output_base_dir='./')
    
    print("Processing complete!")

if __name__ == "__main__":
    main()