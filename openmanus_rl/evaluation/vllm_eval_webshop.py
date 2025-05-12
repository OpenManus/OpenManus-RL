import sys
import os
import json
import concurrent.futures
import argparse
import traceback  # Import traceback for detailed error logging

from transformers import AutoTokenizer, GenerationConfig

from agentgym.agentenv.agentenv.controller import Agent, Evaluator
from agentgym.agentenv.agentenv.envs.webshop import WebshopTask
from tqdm import tqdm


def evaluate_single_task(model_path, env_server_base, max_rounds, idx):
    """
    Initializes necessary components (tokenizer, agent, task, evaluator)
    and evaluates a single WebShop task by index.
    Returns the experience data on success, None otherwise.
    """
    try:
        # --- Initialize resources within the worker ---
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            return None  # Cannot proceed without tokenizer

        # Define generation config
        generation_config = GenerationConfig(
            max_length=4096,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        )

        # Initialize Agent (model is None as per original script)
        agent = Agent(model=None, tokenizer=tokenizer)

        # Initialize WebshopTask
        webshop_task = WebshopTask(
            client_args={
                "env_server_base": env_server_base,
                "data_len": 200,  # Often unused, can be omitted if causing issues
                "timeout": 300,
            },
            n_clients=1,  # Evaluate one task index at a time
        )

        # Initialize Evaluator
        evaluator = Evaluator(agent, [webshop_task])
        
        # Call evaluator.eval for a single index
        result = evaluator.eval(
            generation_config=generation_config,
            max_rounds=max_rounds,
            idxs=[idx],  # Evaluate only this specific index
        )

        # Extract experience data if successful
        if result and result.experiences:
            experience = result.experiences[0]
            # Return entire experience object including conversation, reward, and success
            return {
                "conversation": getattr(experience, 'conversation', None),
                "reward": getattr(experience, 'reward', 0.0),
                "success": 1 if getattr(experience, 'reward', 0.0) == 1 else 0
            }
        else:
            return None

    except Exception as e:
        traceback.print_exc()
        return None


def main():
    print(f"Current working directory: {os.getcwd()}")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run WebShop evaluation concurrently, initialize evaluator per worker, and save results to JSONL.')
    parser.add_argument('--model_name', type=str, default='Qwen3-8B', help='Name of the model being evaluated (e.g., AgentLM-7B)')
    parser.add_argument('--sector', type=str, default='eval', help='Sector or domain of the evaluation (e.g., WebShop)')
    parser.add_argument('--num_tasks', type=int, default=100, help='Number of tasks to process (default: 100)')
    parser.add_argument('--max_workers', type=int, default=20, help='Maximum number of concurrent workers (default: 20)')
    parser.add_argument('--model_path', type=str, default="/data1/models/Qwen/Qwen3-8B-FP8", help='Path to the model directory')
    parser.add_argument('--env_server_base', type=str, default="http://127.0.0.1:36001", help='Base URL for the environment server')
    parser.add_argument('--max_rounds', type=int, default=7, help='Maximum interaction rounds per task (default: 7)')
    parser.add_argument('--output_file', type=str, default="", help='Output file path (default: {model_name}_{sector}.jsonl)')

    args = parser.parse_args()

    # Use arguments
    model_name = args.model_name
    sector = args.sector
    num_tasks_to_process = args.num_tasks
    max_workers = args.max_workers
    model_path = args.model_path
    env_server_base = args.env_server_base
    max_rounds = args.max_rounds
    output_filename = args.output_file if args.output_file else f"{model_name}_{sector}.jsonl"

    # --- Concurrency Logic ---
    all_experiences = []  # Store all experience data, not just conversations

    print(f"Starting concurrent evaluation of the first {num_tasks_to_process} tasks with {max_workers} workers.")
    print(f"Each worker will initialize its own evaluator.")
    print(f"Results will be saved to: {output_filename}")
    print(f"Model path: {model_path}")
    print(f"Env server base: {env_server_base}")
    print(f"Max rounds per task: {max_rounds}")

    # Track success metrics
    total_score = 0.0
    total_success = 0.0
    total_completed = 0

    # Use ThreadPoolExecutor for concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_idx = {
            executor.submit(evaluate_single_task, model_path, env_server_base, max_rounds, i): i
            for i in range(num_tasks_to_process)
        }

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc="Evaluating tasks"):
            idx = future_to_idx[future]
            try:
                experience_data = future.result()  # This should be the dictionary with conversation, reward, success
                if experience_data is not None and experience_data["conversation"] is not None:
                    # Add task_id to experience data
                    experience_data["item_id"] = f"webshop_{idx}"
                    all_experiences.append(experience_data)
                    
                    # Update metrics
                    total_score += experience_data["reward"]
                    total_success += experience_data["success"]
                    total_completed += 1
                else:
                    print(f"Task {idx} completed but returned no valid data.")
            except Exception as exc:
                print(f'Task {idx} generated an exception during future processing: {exc}')
                traceback.print_exc()

    print(f"\n==== CONCURRENT EVALUATION COMPLETE (Collected {len(all_experiences)} experiences) ====\n")

    # --- Save Results to JSONL ---
    if all_experiences:
        print(f"Saving {len(all_experiences)} experiences to {output_filename}")
        try:
            with open(output_filename, 'w') as f:
                for exp in all_experiences:
                    f.write(json.dumps(exp) + '\n')
            print(f"Successfully saved experiences to {output_filename}")
        except Exception as e:
            print(f"Error saving results to {output_filename}: {e}")
            traceback.print_exc()
    else:
        print("No experiences were collected to save.")

    # Calculate and print evaluation metrics
    if total_completed > 0:
        score_average = total_score / total_completed
        success_rate = total_success / total_completed
        print("\n\n==== EVALUATION ====\n")
        print(f"Score: {score_average:.4f}")
        print(f"Success: {success_rate:.4f}")
        print(f"Completed Tasks: {total_completed}/{num_tasks_to_process}")


if __name__ == "__main__":
    main()