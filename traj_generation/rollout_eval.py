import os
import sys
import json
import jsonlines
import time
import argparse
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("offline_rollout")

# Import OpenManus and AgentGym components
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from agentenv.controller import Agent, Evaluator
    from agentenv.envs import (
        AlfWorldTask,
        BabyAITask,
        SciworldTask,
        TextCraftTask,
        WebarenaTask,
        WebshopTask,
        SqlGymTask,
        MazeTask,
        WordleTask,
        WeatherTask,
        TodoTask,
        MovieTask,
        SheetTask,
        AcademiaTask
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure AgentGym and its dependencies are installed")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline rollout trajectories for AgentGym environments")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration YAML file")
    parser.add_argument("--model_path", type=str, required="--config" not in sys.argv, 
                        help="Path to the model")
    parser.add_argument("--output_dir", type=str, default="./offline_rollout_results",
                        help="Directory to save results")
    parser.add_argument("--task_names", nargs="+", default=["webshop"],
                        help="Space-separated list of task names to evaluate")
    parser.add_argument("--inference_files", nargs="+", default=None,
                        help="Space-separated list of inference files for each task")
    parser.add_argument("--max_rounds", nargs="+", type=int, default=None,
                        help="Maximum interaction rounds for each task")
    parser.add_argument("--env_server_bases", nargs="+", default=None,
                        help="Environment server base URLs for each task")
    parser.add_argument("--data_len", type=int, default=200,
                        help="Number of data samples in environment")
    parser.add_argument("--timeout", type=int, default=2400,
                        help="Timeout value for environment interactions")
    parser.add_argument("--do_sample", type=str, default="False",
                        help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode")
    return parser.parse_args()

def read_config(config_path: str) -> Dict[str, Any]:
    """Read and process configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required_fields = ["model_path", "output_dir", "tasks"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Config file missing required field: {field}")
                
        # Process tasks configuration
        if not isinstance(config["tasks"], dict) or not config["tasks"]:
            raise ValueError("Config must contain at least one task under 'tasks' key")
            
        return config
    except Exception as e:
        logger.error(f"Error reading config file: {e}")
        raise

def get_task_class(task_name: str):
    """Get task class based on task name"""
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "babyai": BabyAITask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask,
        "webarena": WebarenaTask,
        "sqlgym": SqlGymTask,
        'maze': MazeTask,
        'wordle': WordleTask,
        "weather": WeatherTask,
        "todo": TodoTask,
        "movie": MovieTask,
        "sheet": SheetTask,
        "academia": AcademiaTask
    }
    
    task_class = task_classes.get(task_name.lower(), None)
    if task_class is None:
        raise ValueError(f"Unsupported task name: {task_name}")
    
    return task_class

def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load data from the specified file path"""
    logger.info(f"Loading data from {data_path}")
    
    try:
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        elif data_path.endswith('.jsonl'):
            data = []
            with jsonlines.open(data_path) as reader:
                for item in reader:
                    data.append(item)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded {len(data)} samples")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def save_summary(output_dir: str, task_name: str, success_rate: float, score: float, 
                 total_samples: int, success_count: int, process_time: float):
    """Save evaluation summary to file"""
    summary_path = os.path.join(output_dir, f"{task_name}_summary.json")
    
    summary = {
        "task_name": task_name,
        "success_rate": success_rate,
        "score": score,
        "total_samples": total_samples,
        "success_count": success_count,
        "process_time": process_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")

def extract_data_idxs(data: List[Dict[str, Any]], task_name: str) -> List[List[int]]:
    """Extract data indices from the data"""
    data_idxs = []
    
    for item in data:
        # Handle different data formats
        if "item_id" in item:
            # Format: "task_123" or just "123"
            item_id = item["item_id"]
            if isinstance(item_id, str):
                if task_name in item_id:
                    # Extract number after task_name_
                    idx = int(item_id.split("_")[-1])
                else:
                    # If no task prefix, just use the ID directly
                    try:
                        idx = int(item_id)
                    except ValueError:
                        idx = int(item_id.split("_")[-1])
            else:
                idx = item_id
        elif "id" in item:
            idx = item["id"]
        elif "index" in item:
            idx = item["index"]
        else:
            # Default to using the position in the list
            idx = data.index(item)
            
        data_idxs.append([idx])
    
    return data_idxs

def generate_trajectories(
    model_path: str,
    task_name: str,
    inference_file: str,
    output_file: str,
    max_rounds: int,
    env_server_base: str,
    data_len: int = 200,
    timeout: int = 2400,
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    """Generate trajectories for a specific task"""
    logger.info(f"Generating trajectories for {task_name}")
    logger.info(f"Using model: {model_path}")
    logger.info(f"Environment server: {env_server_base}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        trust_remote_code=True
    ).eval()
    
    # Get task class
    task_class = get_task_class(task_name)
    
    # Set environment parameters
    env_args = {
        "env_server_base": env_server_base,
        "data_len": data_len,
        "timeout": timeout,
    }
    
    # Set up evaluator
    evaluator = Evaluator(
        Agent(model, tokenizer),
        [task_class(client_args=env_args, n_clients=1)],
    )
    
    # Load data
    test_data = load_data(inference_file)
    data_idxs = extract_data_idxs(test_data, task_name)
    
    # Set generation config
    gen_config = GenerationConfig(
        max_length=4096,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
    )
    
    # Initialize output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)
        
    # Run evaluation
    total_score = 0.0
    total_success = 0.0
    start_time = time.time()
    
    for data_idx in tqdm(data_idxs, total=len(data_idxs), desc=f"[Evaluating {task_name}]"):
        try:
            exps = evaluator.eval(
                generation_config=gen_config,
                max_rounds=max_rounds,
                idxs=data_idx
            )
            
            total_score += exps.score
            total_success += exps.success
            
            cur_experiences = exps.experiences
            
            # Write results to file
            with jsonlines.open(output_file, mode="a") as f:
                for exp in cur_experiences:
                    conversation = exp.conversation
                    cur_reward = exp.reward
                    cur_success = 1 if exp.reward == 1 else 0
                    item_id = f"{task_name}_{data_idx[0]}"
                    
                    f.write({
                        "conversations": conversation,
                        "item_id": item_id,
                        "reward": cur_reward,
                        "success": cur_success,
                    })
                    
        except Exception as e:
            logger.error(f"Error evaluating sample {data_idx}: {e}")
            # Write error result to file
            with jsonlines.open(output_file, mode="a") as f:
                f.write({
                    "conversations": [{"error": str(e)}],
                    "item_id": f"{task_name}_{data_idx[0]}",
                    "reward": 0.0,
                    "success": 0,
                    "error": str(e)
                })
    
    process_time = time.time() - start_time
    
    score = total_score / len(data_idxs) if data_idxs else 0
    success_rate = total_success / len(data_idxs) if data_idxs else 0
    success_count = int(total_success)
    
    logger.info(f"Task: {task_name}")
    logger.info(f"Score: {score:.4f}")
    logger.info(f"Success rate: {success_rate:.4f} ({success_count}/{len(data_idxs)})")
    logger.info(f"Time: {process_time:.2f} seconds")
    
    # Return results
    return {
        "task_name": task_name,
        "score": score,
        "success_rate": success_rate,
        "total_samples": len(data_idxs),
        "success_count": success_count,
        "process_time": process_time
    }

def create_output_filepaths(output_dir: str, task_names: List[str]) -> Dict[str, str]:
    """Create output file paths for each task"""
    output_files = {}
    
    for task_name in task_names:
        task_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        output_files[task_name] = os.path.join(task_dir, f"{task_name}_trajectories.jsonl")
    
    return output_files

def main():
    args = parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process configuration
    if args.config:
        config = read_config(args.config)
        model_path = args.model_path or config["model_path"]
        output_dir = args.output_dir or config["output_dir"]
        task_names = args.task_names or list(config["tasks"].keys())
        
        # Get task-specific configurations
        inference_files = {}
        max_rounds = {}
        env_server_bases = {}
        
        for task_name in task_names:
            if task_name in config["tasks"]:
                task_config = config["tasks"][task_name]
                inference_files[task_name] = task_config.get("inference_file")
                max_rounds[task_name] = task_config.get("max_rounds")
                env_server_bases[task_name] = task_config.get("env_server_base")
    else:
        model_path = args.model_path
        output_dir = args.output_dir
        task_names = args.task_names
        
        # Process lists from command line
        inference_files = {}
        max_rounds = {}
        env_server_bases = {}
        
        if args.inference_files:
            for i, task_name in enumerate(task_names):
                if i < len(args.inference_files):
                    inference_files[task_name] = args.inference_files[i]
        
        if args.max_rounds:
            for i, task_name in enumerate(task_names):
                if i < len(args.max_rounds):
                    max_rounds[task_name] = args.max_rounds[i]
        
        if args.env_server_bases:
            for i, task_name in enumerate(task_names):
                if i < len(args.env_server_bases):
                    env_server_bases[task_name] = args.env_server_bases[i]
    
    # Set default server bases if not specified
    for task_name in task_names:
        if task_name not in env_server_bases or not env_server_bases[task_name]:
            env_server_bases[task_name] = "http://localhost:8000"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output files for each task
    output_files = create_output_filepaths(output_dir, task_names)
    
    # Run evaluation for each task
    all_results = {}
    
    for task_name in task_names:
        if task_name not in inference_files or not inference_files[task_name]:
            logger.error(f"No inference file specified for {task_name}, skipping")
            continue
        
        try:
            logger.info(f"Processing task: {task_name}")
            
            # Set default max_rounds if not specified
            task_max_rounds = max_rounds.get(task_name, 10)
            
            # Generate trajectories
            task_results = generate_trajectories(
                model_path=model_path,
                task_name=task_name,
                inference_file=inference_files[task_name],
                output_file=output_files[task_name],
                max_rounds=task_max_rounds,
                env_server_base=env_server_bases[task_name],
                data_len=args.data_len,
                timeout=args.timeout,
                do_sample=args.do_sample.lower() == "true",
                temperature=args.temperature,
                seed=args.seed
            )
            
            all_results[task_name] = task_results
            
            # Save summary
            save_summary(
                output_dir=os.path.join(output_dir, task_name),
                task_name=task_name,
                success_rate=task_results["success_rate"],
                score=task_results["score"],
                total_samples=task_results["total_samples"],
                success_count=task_results["success_count"],
                process_time=task_results["process_time"]
            )
            
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Save overall results
    overall_results_path = os.path.join(output_dir, "overall_results.json")
    with open(overall_results_path, 'w') as f:
        json.dump({
            "model_path": model_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": all_results
        }, f, indent=2)
    
    # Print summary table
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Task':<15} | {'Success Rate':<15} | {'Score':<10} | {'Success/Total':<15}")
    logger.info("-"*60)
    
    for task_name, results in all_results.items():
        logger.info(f"{task_name:<15} | {results['success_rate']:.2%:<15} | {results['score']:.4f}   | {results['success_count']}/{results['total_samples']:<15}")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()