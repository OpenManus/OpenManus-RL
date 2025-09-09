#!/usr/bin/env python3
"""
Data Processing Script for Tool Use Environment
Converts data.json to training format compatible with OpenManus RL system
"""

import json
import argparse
import os
from typing import List, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path: str) -> List[Dict]:
    """Load data from JSON file"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def filter_data_by_level(data: List[Dict], level: str = None) -> List[Dict]:
    """Filter data by difficulty level if specified"""
    if level is None:
        return data
    
    filtered_data = []
    for item in data:
        item_level = item.get('Level', '1')
        if item_level == str(level):
            filtered_data.append(item)
    
    return filtered_data


def convert_to_training_format(data: List[Dict]) -> List[Dict]:
    """
    Convert raw data to training format expected by the system.
    Each item contains the task and expected answer.
    """
    training_data = []
    
    for item in data:
        # Extract required fields
        training_item = {
            'pid': str(item.get('pid', len(training_data))),
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'task_id': item.get('task_id', ''),
            'level': item.get('Level', '1'),
            'split': item.get('split', 'train')  # Default to train if not specified
        }
        
        # Add optional metadata
        if 'Annotator Metadata' in item:
            metadata = item['Annotator Metadata']
            training_item['metadata'] = {
                'steps': metadata.get('Steps', ''),
                'num_steps': metadata.get('Number of steps', ''),
                'tools': metadata.get('Tools', ''),
                'num_tools': metadata.get('Number of tools', ''),
                'time_taken': metadata.get('How long did this take?', '')
            }
        
        training_data.append(training_item)
    
    return training_data


def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.2) -> Dict[str, List[Dict]]:
    """
    Split data into train/val sets if not already split.
    """
    # Check if data already has split field
    has_splits = any('split' in item for item in data)
    
    if has_splits:
        # Use existing splits
        train_data = [item for item in data if item.get('split', 'train') == 'train']
        val_data = [item for item in data if item.get('split', 'train') in ['val', 'validation']]
        test_data = [item for item in data if item.get('split', 'train') == 'test']
        
        print(f"Using existing splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    else:
        # Create new splits
        train_data, temp_data = train_test_split(data, test_size=(1-train_ratio), random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # Update split field
        for item in train_data:
            item['split'] = 'train'
        for item in val_data:
            item['split'] = 'val'
        for item in test_data:
            item['split'] = 'test'
        
        print(f"Created new splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }


def save_data(data_splits: Dict[str, List[Dict]], output_dir: str):
    """Save processed data to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in data_splits.items():
        if not split_data:  # Skip empty splits
            continue
            
        # Save as JSON
        json_path = os.path.join(output_dir, f"{split_name}.json")
        with open(json_path, 'w') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        # Save as Parquet (compatible with existing data format)
        parquet_path = os.path.join(output_dir, f"{split_name}.parquet")
        df = pd.DataFrame(split_data)
        df.to_parquet(parquet_path, index=False)
        
        print(f"Saved {len(split_data)} {split_name} samples to {json_path} and {parquet_path}")


def analyze_data(data: List[Dict]):
    """Print data analysis"""
    print("\n" + "="*50)
    print("DATA ANALYSIS")
    print("="*50)
    
    print(f"Total samples: {len(data)}")
    
    # Analyze by level
    levels = {}
    for item in data:
        level = item.get('level', '1')
        levels[level] = levels.get(level, 0) + 1
    
    print("\nBy difficulty level:")
    for level in sorted(levels.keys()):
        print(f"  Level {level}: {levels[level]} samples")
    
    # Analyze answer lengths
    answer_lengths = [len(str(item.get('answer', ''))) for item in data]
    print(f"\nAnswer length statistics:")
    print(f"  Min: {min(answer_lengths)}, Max: {max(answer_lengths)}")
    print(f"  Average: {sum(answer_lengths) / len(answer_lengths):.1f}")
    
    # Check for tools mentioned in metadata
    tools_mentioned = set()
    for item in data:
        metadata = item.get('metadata', {})
        tools = metadata.get('tools', '')
        if tools:
            # Simple extraction of tool names
            if 'search' in tools.lower():
                tools_mentioned.add('search')
            if 'browser' in tools.lower():
                tools_mentioned.add('browser')
            if 'image' in tools.lower():
                tools_mentioned.add('image_processing')
    
    print(f"\nTools mentioned in metadata: {list(tools_mentioned)}")


def main():
    parser = argparse.ArgumentParser(description="Process tool use data for training")
    parser.add_argument("--input", type=str, default="../data/gaia/val.json", 
                       help="Input data file path")
    parser.add_argument("--output", type=str, default="../data/gaia", 
                       help="Output directory for processed data")
    parser.add_argument("--level", type=str, default=None, 
                       help="Filter by difficulty level (1, 2, 3, etc.)")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                       help="Training data ratio (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.2, 
                       help="Validation data ratio (default: 0.2)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    print("Processing Tool Use Data")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Load data
    print("Loading data...")
    raw_data = load_data(args.input)
    print(f"Loaded {len(raw_data)} samples")
    
    # Filter by level if specified
    if args.level:
        raw_data = filter_data_by_level(raw_data, args.level)
        print(f"Filtered to {len(raw_data)} samples for level {args.level}")
    
    # Limit samples if specified
    if args.max_samples and len(raw_data) > args.max_samples:
        raw_data = raw_data[:args.max_samples]
        print(f"Limited to {len(raw_data)} samples")
    
    # Convert to training format
    print("Converting to training format...")
    training_data = convert_to_training_format(raw_data)
    
    # Analyze data
    analyze_data(training_data)
    
    # Split data
    print("\nSplitting data...")
    data_splits = split_data(training_data, args.train_ratio, args.val_ratio)
    
    # Save processed data
    print("\nSaving processed data...")
    save_data(data_splits, args.output)
    
    print(f"\nProcessing complete! Data saved to {args.output}")
    print("\nNext steps:")
    print(f"1. Update your config file to use env_name: 'tool_use'")
    print(f"2. Set data_path: '{args.output}/train.json' in config")
    print(f"3. Configure available_tools in your config")


if __name__ == "__main__":
    main()
