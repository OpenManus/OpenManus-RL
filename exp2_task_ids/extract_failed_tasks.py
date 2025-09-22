#!/usr/bin/env python3
"""
Script to extract failed task information from experiment results.

This script reads task_summary.json files from trajectories folder,
identifies failed tasks (won=False), and collects corresponding task
information from assignments folder to create a JSON list for reprocessing.
"""

import os
import json
import argparse
from typing import List, Dict, Any


def read_file_content(file_path: str) -> str:
    """
    Read the content of a text file and return it as a string.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Content of the file as string, or empty string if file doesn't exist
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def read_task_summary(summary_path: str) -> Dict[str, Any]:
    """
    Read and parse task_summary.json file.
    
    Args:
        summary_path: Path to task_summary.json file
        
    Returns:
        Dictionary containing parsed JSON data, or empty dict if error
    """
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Task summary not found: {summary_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {summary_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error reading task summary {summary_path}: {e}")
        return {}


def get_failed_env_ids(trajectories_dir: str) -> List[int]:
    """
    Scan all trajectory folders and collect env_ids where won=False.
    
    Args:
        trajectories_dir: Path to trajectories directory
        
    Returns:
        List of env_ids for failed tasks
    """
    failed_env_ids = []
    
    if not os.path.exists(trajectories_dir):
        print(f"Error: Trajectories directory not found: {trajectories_dir}")
        return failed_env_ids
    
    # Iterate through all subdirectories in trajectories
    for subdir in os.listdir(trajectories_dir):
        subdir_path = os.path.join(trajectories_dir, subdir)
        
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
            
        # Look for task_summary.json in this subdirectory
        summary_path = os.path.join(subdir_path, "task_summary.json")
        
        if not os.path.exists(summary_path):
            print(f"Warning: No task_summary.json found in {subdir_path}")
            continue
            
        # Read and parse the summary
        summary_data = read_task_summary(summary_path)
        
        if not summary_data:
            continue
            
        # Extract metadata
        metadata = summary_data.get("metadata", {})
        env_id = metadata.get("env_id")
        won = metadata.get("won", False)
        
        # Check if this is a failed task
        if env_id is not None and not won:
            failed_env_ids.append(env_id)
            print(f"Found failed task: env_id={env_id} in {subdir}")
    
    return failed_env_ids


def extract_task_info(assignments_dir: str, env_id: int) -> Dict[str, str]:
    """
    Extract task_ids and task_paths for a specific env_id from assignments folder.
    
    Args:
        assignments_dir: Path to assignments directory
        env_id: Environment ID to look for
        
    Returns:
        Dictionary with task_ids and task_paths content
    """
    # Format env_id as env_XXX (3-digit zero-padded)
    # Note: env_id=0 corresponds to env_001, env_id=1 to env_002, etc.
    env_folder = f"env_{env_id+1:03d}"
    env_path = os.path.join(assignments_dir, env_folder)
    
    result = {"task_ids": "", "task_paths": ""}
    
    if not os.path.exists(env_path):
        print(f"Warning: Assignment folder not found: {env_path}")
        return result
    
    # Read task_ids.txt
    task_ids_path = os.path.join(env_path, "task_ids.txt")
    result["task_ids"] = read_file_content(task_ids_path)
    
    # Read task_paths.txt
    task_paths_path = os.path.join(env_path, "task_paths.txt")
    result["task_paths"] = read_file_content(task_paths_path)
    
    return result


def main():
    """
    Main function to process failed tasks and generate JSON output.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract failed task information from experiment results"
    )
    parser.add_argument(
        "--input_path",
        default="/root/kunlun/OpenManus-RL/experiments/unified_debug_20250921_220222/alfworld",
        help="Path to experiment directory containing assignments, summaries, and trajectories folders"
    )
    parser.add_argument(
        "--output_file",
        default="alfworld_4o_mini_failed_tasks.json",
        help="Path to output JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_path):
        print(f"Error: Input directory does not exist: {args.input_path}")
        return 1
    
    # Define subdirectory paths
    trajectories_dir = os.path.join(args.input_path, "trajectories")
    assignments_dir = os.path.join(args.input_path, "assignments")
    
    # Check if required directories exist
    if not os.path.exists(trajectories_dir):
        print(f"Error: Trajectories directory not found: {trajectories_dir}")
        return 1
        
    if not os.path.exists(assignments_dir):
        print(f"Error: Assignments directory not found: {assignments_dir}")
        return 1
    
    print(f"Processing experiment data from: {args.input_path}")
    print(f"Scanning trajectories in: {trajectories_dir}")
    print(f"Looking for assignments in: {assignments_dir}")
    
    # Step 1: Get all failed env_ids
    print("\nStep 1: Identifying failed tasks...")
    failed_env_ids = get_failed_env_ids(trajectories_dir)
    
    if not failed_env_ids:
        print("No failed tasks found. All tasks were successful!")
        # Create empty JSON array
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        print(f"Empty JSON array saved to: {args.output_file}")
        return 0
    
    print(f"Found {len(failed_env_ids)} failed tasks: {sorted(failed_env_ids)}")
    
    # Step 2: Extract task information for failed env_ids
    print("\nStep 2: Extracting task information...")
    failed_tasks = []
    
    for env_id in sorted(failed_env_ids):
        env_folder = f"env_{env_id+1:03d}"
        print(f"Processing env_id={env_id} (mapping to folder {env_folder})...")
        task_info = extract_task_info(assignments_dir, env_id)
        
        if task_info["task_ids"] or task_info["task_paths"]:
            failed_tasks.append(task_info)
            print(f"  Successfully extracted info for env_id={env_id} from {env_folder}")
        else:
            print(f"  Warning: No task information found for env_id={env_id} in {env_folder}")
    
    # Step 3: Save results to JSON file
    print(f"\nStep 3: Saving results...")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(failed_tasks, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved {len(failed_tasks)} failed tasks to: {args.output_file}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total failed tasks: {len(failed_env_ids)}")
        print(f"  Tasks with extracted info: {len(failed_tasks)}")
        print(f"  Output file: {args.output_file}")
        
    except Exception as e:
        print(f"Error saving output file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
