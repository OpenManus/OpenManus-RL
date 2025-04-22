from datasets import load_dataset
import re
import os
import pandas as pd
from collections import defaultdict

# Define environments to extract
ENVIRONMENTS = [
    "alworld", "babyai", "lmrlgym_maze", "lmrlgum_wordle", 
    "sciworld", "sqlgym", "textcraft", "movie", 
    "todo", "weather", "webshop"
]

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

def group_samples_by_environment(data, environments):
    """
    Group data samples by their environment based on item_id.
    
    Args:
        data: Dataset samples
        environments: List of environment names to look for
        
    Returns:
        Dictionary with environment names as keys and sample lists as values
    """
    env_groups = defaultdict(list)
    prefix_pattern = re.compile(r'^([^\d]+)')  # Regex to extract prefix before numbers
    
    for sample in data:
        item_id = sample['item_id']
        
        # Extract item_id prefix until digits start
        match = prefix_pattern.match(item_id)
        if match:
            item_id_prefix = match.group(1)
        else:
            item_id_prefix = item_id
        
        # Check if item_id contains any of the specified environments
        for env in environments:
            if env in item_id:
                env_groups[env].append(sample)
                break  # If matched to one environment, don't check others
    
    return env_groups

def save_environment_data(env_groups, output_dir, txt_output_dir):
    """
    Save grouped data to parquet and txt files.
    
    Args:
        env_groups: Dictionary of environment groups
        output_dir: Directory to save parquet files
        txt_output_dir: Directory to save txt files
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(txt_output_dir, exist_ok=True)
    
    # Save each environment group as parquet and txt
    for env, samples in env_groups.items():
        print(f"Processing group: {env}")
        
        # Process samples for this environment
        processed_samples = process_group_data(env, samples)
        
        # Convert processed data to DataFrame
        df = pd.DataFrame(processed_samples)
        
        # Generate file paths
        parquet_file_path = os.path.join(output_dir, f"{env}.parquet")
        txt_file_path = os.path.join(txt_output_dir, f"{env}.txt")
        
        # Save as Parquet file
        df.to_parquet(parquet_file_path)
        print(f"Saved data for environment '{env}' to {parquet_file_path}")
        
        # Save as TXT file
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            for sample in processed_samples:
                txt_file.write(str(sample) + '\n')
        print(f"Saved data for environment '{env}' to {txt_file_path}")

def main():
    """
    Main function to process and save AgentGym dataset by environment.
    """
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("AgentGym/AgentTraj-L")
    train_data = dataset['train']
    
    # Group samples by environment
    print("Grouping samples by environment...")
    env_groups = group_samples_by_environment(train_data, ENVIRONMENTS)
    
    # Print group statistics
    for env, samples in env_groups.items():
        print(f"Environment: {env}, Number of samples: {len(samples)}")
    
    # Save grouped data
    print("Saving environment data...")
    save_environment_data(
        env_groups, 
        output_dir='output_env_groups',
        txt_output_dir='output_txt_files'
    )
    
    print("Processing complete!")

if __name__ == "__main__":
    main()