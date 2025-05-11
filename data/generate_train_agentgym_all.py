from datasets import load_dataset
import re
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import json
import random

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
        conversations_text = json.dumps(conversations)
        
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
                "ground_truth": conversations_text
            },
            "extra_info": {
                'split': group_name,
                'index': idx,
                'conversations': conversations_text
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

def get_environment_from_item_id(item_id, env_mapping):
    """
    Determine the environment name from an item_id using the mapping.
    
    Args:
        item_id: The item identifier string
        env_mapping: Dictionary mapping environment names to potential ID prefixes
    
    Returns:
        Environment name or None if not matched
    """
    for env_name, prefixes in env_mapping.items():
        for prefix in prefixes:
            if prefix in item_id:
                return env_name
    return None

def create_id_to_sample_map(samples):
    """
    Create a mapping from item_id to sample data.
    
    Args:
        samples: List of sample data
        
    Returns:
        Dictionary mapping item_id to sample data
    """
    id_map = {}
    for sample in samples:
        id_map[sample['item_id']] = sample
    return id_map

def save_environment_data(train_env_groups, eval_item_ids, train_id_to_sample, output_base_dir):
    """
    Save environment data with train samples, random validation samples, and test samples.
    Test samples are from AgentTraj-L but their item_ids come from AgentEval.
    
    Args:
        train_env_groups: Dictionary with environment name as key and training samples as value
        eval_item_ids: Dictionary with environment name as key and list of item_ids from evaluation dataset
        train_id_to_sample: Dictionary mapping item_id to sample data from training dataset
        output_base_dir: Base directory where environment subdirectories will be created
    """
    # Ensure base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each environment group
    for env_name, samples in train_env_groups.items():
        if not samples:
            print(f"Warning: No samples found for environment '{env_name}'. Skipping.")
            continue
            
        print(f"Processing environment: {env_name} with {len(samples)} samples")
        
        # Create environment subdirectory
        env_dir = os.path.join(output_base_dir, env_name)
        os.makedirs(env_dir, exist_ok=True)
        
        # Process training samples for this environment
        processed_train_samples = process_group_data(env_name, samples)
        
        # Get test sample ids from eval dataset for this environment
        eval_ids = eval_item_ids.get(env_name, [])
        
        # Find matching samples in training data using eval item_ids
        test_sample_data = []
        matched_count = 0
        for item_id in eval_ids:
            if item_id in train_id_to_sample:
                test_sample_data.append(train_id_to_sample[item_id])
                matched_count += 1
        
        print(f"Found {matched_count} out of {len(eval_ids)} test samples for {env_name}")
        
        # Process test samples
        processed_test_samples = process_group_data(env_name, test_sample_data)
        
        # Get validation samples: randomly select 20 samples from training data
        # Avoid using samples that are in the test set
        test_item_ids = set(item['item_id'] for item in processed_test_samples)
        available_train_samples = [s for s in processed_train_samples if s['item_id'] not in test_item_ids]
        
        val_samples = []
        if len(available_train_samples) > 20:
            # Random sample without replacement
            val_indices = random.sample(range(len(available_train_samples)), 20)
            val_samples = [available_train_samples[i] for i in val_indices]
        else:
            # If less than 20 samples, use all available samples
            val_samples = available_train_samples.copy()
        
        # Save train samples
        if processed_train_samples:
            train_df = pd.DataFrame(processed_train_samples)
            train_file = os.path.join(env_dir, "train.parquet")
            train_df.to_parquet(train_file)
            print(f"Saved {len(processed_train_samples)} training samples to {train_file}")
        
        # Save validation samples
        if val_samples:
            val_df = pd.DataFrame(val_samples)
            val_file = os.path.join(env_dir, "val.parquet")
            val_df.to_parquet(val_file)
            print(f"Saved {len(val_samples)} validation samples to {val_file}")
        
        # Save test samples
        if processed_test_samples:
            test_df = pd.DataFrame(processed_test_samples)
            test_file = os.path.join(env_dir, "test.parquet")
            test_df.to_parquet(test_file)
            print(f"Saved {len(processed_test_samples)} test samples to {test_file}")
        else:
            print(f"Warning: No test samples found for environment '{env_name}'")

def main():
    """
    Main function to process and save AgentGym datasets by environment.
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load the training dataset
    print("Loading training dataset (AgentTraj-L)...")
    train_dataset = load_dataset("AgentGym/AgentTraj-L")
    train_data = train_dataset['train']
    
    # Load the evaluation dataset
    print("Loading evaluation dataset (AgentEval)...")
    eval_dataset = load_dataset("AgentGym/AgentEval")
    eval_data = eval_dataset['test']
    
    # Group training samples by environment
    print("Grouping training samples by environment...")
    train_env_groups = group_samples_by_environment(train_data, ENV_ID_MAPPING)
    
    # Create a mapping from item_id to sample data
    print("Creating item_id to sample mapping...")
    train_id_to_sample = create_id_to_sample_map(train_data)
    
    # Group evaluation sample IDs by environment
    print("Grouping evaluation sample IDs by environment...")
    eval_env_item_ids = defaultdict(list)
    
    for sample in eval_data:
        item_id = sample['item_id']
        env_name = get_environment_from_item_id(item_id, ENV_ID_MAPPING)
        
        if env_name:
            eval_env_item_ids[env_name].append(item_id)
        else:
            print(f"Warning: Could not match evaluation sample with item_id '{item_id}' to any environment")
    
    # Print statistics
    print("\n--- Training Data Statistics ---")
    for env, samples in train_env_groups.items():
        print(f"Environment: {env}, Number of training samples: {len(samples)}")
    
    print("\n--- Evaluation Data Statistics ---")
    for env, item_ids in eval_env_item_ids.items():
        print(f"Environment: {env}, Number of evaluation item_ids: {len(item_ids)}")
    
    # Save environment data
    print("\nSaving environment data...")
    save_environment_data(train_env_groups, eval_env_item_ids, train_id_to_sample, output_base_dir='./')
    
    print("Processing complete!")

if __name__ == "__main__":
    main()