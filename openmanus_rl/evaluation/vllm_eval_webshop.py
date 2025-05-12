import sys
import os
import json
import concurrent.futures
import argparse
import traceback # Import traceback for detailed error logging

from transformers import AutoTokenizer, GenerationConfig

from agentgym.agentenv.agentenv.controller import Agent, Evaluator
from agentgym.agentenv.agentenv.envs.webshop import WebshopTask
from tqdm import tqdm


def evaluate_single_task(model_path, env_server_base, max_rounds, idx):
    """
    Initializes necessary components (tokenizer, agent, task, evaluator)
    and evaluates a single WebShop task by index.
    Returns only the conversation list on success, None otherwise.
    """
    # print(f"[Task {idx}] Starting evaluation...")
    try:
        # --- Initialize resources within the worker ---
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # print(f"[Task {idx}] Tokenizer loaded successfully.")
        except Exception as e:
            # print(f"[Task {idx}] Error loading tokenizer from {model_path}: {e}")
            # print(f"[Task {idx}] Please ensure the model_path is correct and the model files are accessible.")
            return None # Cannot proceed without tokenizer

        # Define generation config
        generation_config = GenerationConfig(
            max_length=4096,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        )
        # print(f"[Task {idx}] Generation config created.")

        # Initialize Agent (model is None as per original script)
        agent = Agent(model=None, tokenizer=tokenizer)
        # print(f"[Task {idx}] Agent initialized.")

        # Initialize WebshopTask
        webshop_task = WebshopTask(
            client_args={
                "env_server_base": env_server_base,
                "data_len": 200, # Often unused, can be omitted if causing issues
                "timeout": 300,
            },
            n_clients=1, # Evaluate one task index at a time
        )
        # print(f"[Task {idx}] WebshopTask initialized.")

        # Initialize Evaluator
        evaluator = Evaluator(agent, [webshop_task])
        # print(f"[Task {idx}] Evaluator initialized.")
        # --- End Initialization ---

        # Call evaluator.eval for a single index.
        # print(f"[Task {idx}] Calling evaluator.eval...")
        result = evaluator.eval(
            generation_config=generation_config,
            max_rounds=max_rounds,
            idxs=[idx], # Evaluate only this specific index
        )
        # print(f"[Task {idx}] Evaluator.eval finished.")

        # Extract conversation if successful
        if result and result.experiences:
            experience = result.experiences[0]
            conversation = getattr(experience, 'conversation', None)
            if conversation is not None:
                # print(f"[Task {idx}] Evaluation successful, returning conversation.")
                return conversation
            else:
                #  print(f"[Task {idx}] Evaluation finished, but no conversation found in experience.")
                 return None
        else:
            # print(f"[Task {idx}] Evaluation finished, but no experiences returned.")
            return None

    except Exception as e:
        # print(f"[Task {idx}] Error during evaluation: {e}")
        # Print detailed traceback for debugging
        traceback.print_exc()
        return None
    finally:
        # Optional: Clean up resources if necessary, though Python's GC might handle it
        # for thread-local objects when the thread finishes.
        # print(f"[Task {idx}] Evaluation attempt complete.")
        pass


def main():
    print(f"Current working directory: {os.getcwd()}")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run WebShop evaluation concurrently, initialize evaluator per worker, and save only conversations to JSONL.')
    parser.add_argument('--model_name', type=str, default='Qwen3-8B', help='Name of the model being evaluated (e.g., AgentLM-7B)')
    parser.add_argument('--sector', type=str, default='Train', help='Sector or domain of the evaluation (e.g., WebShop)')
    parser.add_argument('--num_tasks', type=int, default=100, help='Number of tasks to process (default: 120)')
    parser.add_argument('--max_workers', type=int, default=20, help='Maximum number of concurrent workers (default: 120)')
    parser.add_argument('--model_path', type=str, default="/data1/models/Qwen/Qwen3-8B-FP8", help='Path to the model directory')
    parser.add_argument('--env_server_base', type=str, default="http://127.0.0.1:36001", help='Base URL for the environment server')
    parser.add_argument('--max_rounds', type=int, default=7, help='Maximum interaction rounds per task (default: 7)')

    args = parser.parse_args()

    # Use arguments
    model_name = args.model_name
    sector = args.sector
    num_tasks_to_process = args.num_tasks
    max_workers = args.max_workers
    model_path = args.model_path
    env_server_base = args.env_server_base
    max_rounds = args.max_rounds # Use parsed max_rounds
    output_filename = f"{model_name}_{sector}.jsonl" # Added _conversations to filename

    # --- Concurrency Logic ---
    all_conversations = [] # Store only the conversations

    print(f"Starting concurrent evaluation of the first {num_tasks_to_process} tasks with {max_workers} workers.")
    print(f"Each worker will initialize its own evaluator.")
    print(f"Results (conversations only) will be saved to: {output_filename}")
    print(f"Model path: {model_path}")
    print(f"Env server base: {env_server_base}")
    print(f"Max rounds per task: {max_rounds}")


    # Use ThreadPoolExecutor for concurrency
    # Consider ProcessPoolExecutor if thread-safety issues arise with underlying libraries
    # or true process isolation is needed (beware of serialization overhead).
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_idx = {
            # Pass necessary arguments for worker initialization
            executor.submit(evaluate_single_task, model_path, env_server_base, max_rounds, i): i
            for i in range(num_tasks_to_process)
        }

        # Process results as they complete
        # Using tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc="Evaluating tasks"):
            idx = future_to_idx[future]
            try:
                conversation = future.result() # This should be the conversation list or None
                if conversation is not None:
                    # Append the conversation directly (no need for the full experience object)
                    all_conversations.append(conversation)
                    # Optional: Add task_id if needed for context, though not strictly requested
                    # all_conversations.append({"task_id": idx, "conversation": conversation})
                else:
                    # Task failed or returned no conversation, already logged in the function
                    print(f"Task {idx} completed but returned no conversation data.")
            except Exception as exc:
                # This catches errors during future.result() itself, though evaluate_single_task has internal try-except
                print(f'Task {idx} generated an exception during future processing: {exc}')
                traceback.print_exc()


    print(f"\n==== CONCURRENT EVALUATION COMPLETE (Collected {len(all_conversations)} conversations) ====\n")

    # --- Save Results to JSONL ---
    if all_conversations:
        print(f"Saving {len(all_conversations)} conversations to {output_filename}")
        try:
            with open(output_filename, 'w') as f:
                for i, conv in enumerate(all_conversations):
                    # Create a dictionary containing only the conversation for each line
                    # Adding an index might be helpful for reference, but not strictly required
                    line_data = {
                        # "original_task_index": i, # Example if you want to track original submission order index
                        "conversation": conv
                    }
                    f.write(json.dumps(line_data) + '\n')
            print(f"Successfully saved conversations to {output_filename}")
        except Exception as e:
            print(f"Error saving results to {output_filename}: {e}")
            traceback.print_exc()
    else:
        print("No conversations were collected to save.")

    # Example: Print summary
    total_tasks_attempted = num_tasks_to_process
    total_conversations_collected = len(all_conversations)
    print(f"\nSuccessfully collected conversations for {total_conversations_collected} tasks out of {total_tasks_attempted} attempted.")

    # No need to print example conversation as per requirements


if __name__ == "__main__":
    main()