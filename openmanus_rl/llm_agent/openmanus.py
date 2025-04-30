import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from transformers import GenerationConfig
import importlib # Added import
import traceback # For error logging
from concurrent.futures import ThreadPoolExecutor, as_completed # For parallel rollout
from openmanus_rl.utils.visualization import (
    save_trajectory_to_output,
    parse_llm_output
)
from verl.utils.tracking import Tracking
from omegaconf import DictConfig # Import DictConfig for type hint
import numpy as np

@dataclass
class AgentConfig:
    """
    Configuration class for OpenManusAgent.
    
    Attributes:
        max_turns: Maximum number of turns in a conversation
        max_start_length: Maximum length of initial input
        max_prompt_length: Maximum length of prompt
        max_response_length: Maximum length of response
        max_obs_length: Maximum length of observation
        num_gpus: Number of GPUs to use
        env_name: Name of the environment (e.g., "webshop")
        env_ports: List of ports for parallel servers
        env_server_base: Base URL for environment server
        react_format: Whether to use ReAct format
        env_data_len: Number of data samples in the environment (used for client init)
        rollout_strategy: Strategy to use for rollout (StandardReAct/ToT/MCTS)
        max_workers: Maximum number of worker threads
        logging: dict = None  # Contains log_images, log_n_image_per_batch, log_image_step_size, etc.
        algorithm_config: DictConfig = None # Pass relevant part of algorithm config
    """
    # All required fields without default values
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    env_name: str 
    env_ports: List[int] # List of ports for parallel servers
    env_server_base: str
    
    # All optional fields with default values
    react_format: bool = True
    env_data_len: int = 200 # Default, might need adjustment
    rollout_strategy: str = "StandardReAct" # Strategy is now internal logic
    # storage_backend: str = "mongodb" # Storage handled elsewhere or not needed here
    max_workers: int = 10 # For parallelizing rollouts within the agent
    
    # Add visualization-related configuration
    logging: dict = None  # Contains log_images, log_n_image_per_batch, log_image_step_size, etc.

    # Add algorithm config relevant to reward allocation
    algorithm_config: DictConfig = None # Pass relevant part of algorithm config

class OpenManusAgent:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg, # This is the Verl component for generation
        config: AgentConfig,
        is_validation: bool = False,
        logger: Tracking = None,  # Add logger parameter for trajectory saving
    ):
        """
        Initialize OpenManusAgent with rollout controller integration.
        
        Args:
            tokenizer: Tokenizer for text processing
            actor_rollout_wg: Actor rollout wrapper for generation
            config: Agent configuration including env details and algorithm config
            is_validation: Whether in validation mode
            logger: Logger for tracking and visualization
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config # AgentConfig now holds algorithm_config
        self.is_validation = is_validation
        self.logger = logger

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        # Initialize multiple environment clients
        self.clients = self._init_env_clients() # Changed method name

        # Adjust thread pool size based on number of clients, up to max_workers
        num_clients = len(self.clients)
        actual_workers = min(num_clients, self.config.max_workers)
        if actual_workers < num_clients:
             print(f"[Warning] Number of clients ({num_clients}) exceeds max_workers ({self.config.max_workers}). Using {actual_workers} workers.")
        print(f"[Info] Initializing ThreadPoolExecutor with {actual_workers} workers for {num_clients} clients.")
        self.executor = ThreadPoolExecutor(max_workers=actual_workers)

    def _init_env_clients(self) -> List[Any]: # Renamed and return type changed
        """
        Initialize and return a list of specific AgentGym environment clients
        based on the ports provided in the config.
        """
        clients = []
        env_name_lower = self.config.env_name.lower()

        # Mapping from env_name (lowercase) to Task class name
        ENV_TO_TASK_CLASS = {
            "academia": "AcademiaTask", "alfworld": "AlfWorldTask", "babyai": "BabyAITask",
            "maze": "MazeTask", "wordle": "WordleTask", "movie": "MovieTask",
            "sciworld": "SciworldTask", "sheet": "SheetTask", "sqlgym": "SqlGymTask",
            "textcraft": "TextCraftTask", "todo": "TodoTask", "weather": "WeatherTask",
            "webarena": "WebarenaTask", "webshop": "WebshopTask",
        }

        if env_name_lower not in ENV_TO_TASK_CLASS:
            raise ValueError(f"Unsupported environment name: {self.config.env_name}. Supported: {list(ENV_TO_TASK_CLASS.keys())}")

        task_class_name = ENV_TO_TASK_CLASS[env_name_lower]
        print(f"[Info] Initializing {len(self.config.env_ports)} Env Client(s) for: {self.config.env_name} (via Task: {task_class_name})")

        # Dynamically import the Task class
        try:
            envs_module = importlib.import_module("agentenv.envs")
            TaskClass = getattr(envs_module, task_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import Task class {task_class_name} from agentenv.envs: {e}")

        for i, port in enumerate(self.config.env_ports):
            server_url = f"{self.config.env_server_base}:{port}"
            print(f"  - Client {i+1}: Connecting to {server_url}")

            client_args={
                "env_server_base": server_url,
                "data_len": self.config.env_data_len,
                "timeout": 300,
            }

            try:
                # Instantiate the task to get the client.
                # We need one client per specified port.
                # Assuming TaskClass handles client creation correctly when n_clients=1.
                # If TaskClass itself manages multiple internal clients, this might need adjustment.
                task_instance = TaskClass(client_args=client_args, n_clients=1)
                if hasattr(task_instance, 'clients') and task_instance.clients:
                    client = task_instance.clients[0]
                    print(f"  - Client {i+1}: Successfully obtained client: {type(client)}")
                    clients.append(client)
                else:
                     print(f"  - Client {i+1}: Error - Task class {task_class_name} did not provide a client for port {port}.")
                     # Decide how to handle failure: raise error or skip this client? Skipping for now.
                     # raise ValueError(f"Task class {task_class_name} did not provide a client for port {port}.")
            except Exception as e:
                 print(f"  - Client {i+1}: Error initializing Task or getting client for port {port}: {e}")
                 print(traceback.format_exc()) # Print detailed traceback
                 # Decide how to handle failure: raise error or skip? Skipping for now.
                 # raise

        if not clients:
            raise RuntimeError("Failed to initialize any environment clients.")

        print(f"[Info] Successfully initialized {len(clients)} environment clients.")
        return clients

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at tool call or final response."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # In ReAct format, we look for <action> or <response> tags
        processed_responses = []
        for resp in responses_str:
            if '</action>' in resp:
                # Stop at end of action
                processed = resp.split('</action>')[0] + '</action>'
            elif '</response>' in resp:
                # Stop at end of response
                processed = resp.split('</response>')[0] + '</response>'
            else:
                # No recognized end tag, keep as is
                processed = resp
            processed_responses.append(processed)

        responses = self._batch_tokenize(processed_responses)
        return responses, processed_responses

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _setup_visualization(self) -> List[Dict]:
        """Setup visualization tracking if enabled."""
        # If config.logging is not set or log_images is False, return None
        if not self.config.logging or not self.config.logging.get('log_images', False):
            return None
        # Create n_image_per_batch defaultdict(list) instances
        return [defaultdict(list) for _ in range(self.config.logging.get('log_n_image_per_batch', 1))]

    def _update_trajectory(self, trajectory: List[Dict], 
                         envs: List[Any], responses: List[str], active_mask: torch.Tensor):
        """Update visualization trajectory if enabled."""
        if not trajectory:
            return
        # Get the number of environments to visualize
        n_visualize = self.config.logging.get('log_n_image_per_batch', 1)
        # Update environment states
        for idx, (env, active) in enumerate(zip(envs[:n_visualize], active_mask[:n_visualize])):
            if active:
                trajectory[idx]['state'].append(env.render('rgb_array'))
        
        # Update responses
        for idx, (response, env, active) in enumerate(zip(responses[:n_visualize], 
                                                envs[:n_visualize],
                                                active_mask[:n_visualize])):
            if active:
                parsed = parse_llm_output(response, strategy="raw")
                
                trajectory[idx]['answer'].append(response)
                trajectory[idx]['parsed_response'].append(parsed)

    def _save_trajectory(self, trajectory: List[Dict], 
                        output_dir: str, global_steps: int):
        """Save trajectory visualization if enabled."""
        if not trajectory:
            return
            
        # Determine save frequency based on configuration
        save_step_size = self.config.logging.get('log_image_step_size', 100)
        if not global_steps % save_step_size or self.is_validation:
            os.makedirs(output_dir, exist_ok=True)
            filenames = save_trajectory_to_output(trajectory, save_dir=output_dir)
            # If using wandb for logging, save the files
            if self.logger and 'wandb' in self.logger.logger:
                for filename in filenames:
                    self.logger.logger['wandb'].save(filename)

    def _run_single_rollout(self, initial_prompt_ids: torch.Tensor, task_idx: int, client: Any) -> Dict[str, Any]:
        """
        Runs the interaction loop for a single environment instance using the provided client.
        Now includes the final computed reward from the environment step in the result.

        Args:
            initial_prompt_ids: Token IDs for the initial prompt/observation.
            task_idx: The index for resetting the environment.
            client: The specific environment client instance to use for this rollout.

        Returns:
            A dictionary containing the trajectory, step rewards, final reward,
            final env score, turns, and original task index.
        """
        trajectory = []
        step_rewards = []  # Store rewards per step
        final_reward = 0.0 # Reward from the *last step*
        final_env_score = 0.0 # Final score from env info
        done = False
        turns = 0
        current_input_ids = None

        try:
            # Reset environment using the provided client
            # Some envs might need a specific seed or config reset
            # print(f"[Agent._run_single_rollout][{task_idx}] Resetting env...")
            reset_info = client.reset(task_idx) # Capture potential info from reset
            initial_obs_text = client.observe()
            # print(f"[Agent._run_single_rollout][{task_idx}] Initial Obs: {initial_obs_text[:100]}...")

            # Handle initial observation
            if not initial_obs_text:
                # print(f"[Agent._run_single_rollout][{task_idx} @ {client.env_server_base}] Warning: Received empty initial observation. Using initial prompt from batch.")
                # Use the initial prompt text passed in
                initial_prompt_text = self.tokenizer.decode(initial_prompt_ids[0], skip_special_tokens=True)
                trajectory.append({"from": "human", "value": initial_prompt_text})
                current_input_ids = initial_prompt_ids
            else:
                trajectory.append({"from": "human", "value": initial_obs_text})
                current_input_ids = self.tokenizer(initial_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']

            # --- Interaction Loop --- 
            for t in range(self.config.max_turns):
                turns = t + 1
                if current_input_ids is None:
                    # print(f"[Agent._run_single_rollout][{task_idx}] Breaking loop: current_input_ids is None")
                    break

                # Handle input that exceeds max length
                if current_input_ids.shape[1] > self.config.max_prompt_length:
                    # print(f"[Agent._run_single_rollout][{task_idx} @ {client.env_server_base}] Warning: Truncating input {current_input_ids.shape} > {self.config.max_prompt_length}.")
                    current_input_ids = current_input_ids[:, -self.config.max_prompt_length:]

                # Prepare input
                current_attention_mask = self.tensor_fn.create_attention_mask(current_input_ids)
                current_position_ids = self.tensor_fn.create_position_ids(current_attention_mask)
                # device = 'cuda' # Assume target device is cuda; worker group handles internal placement
                gen_input_proto = DataProto.from_dict({
                    'input_ids': current_input_ids, # Pass tensor directly (likely CPU)
                    'attention_mask': current_attention_mask,
                    'position_ids': current_position_ids
                }) # may need to put this on the correct device

                # <<< FIX: Pad batch to match world size if needed >>>
                world_size = self.actor_rollout_wg.world_size
                original_size = 1 # We know batch size is 1 here
                padded_gen_input_proto = gen_input_proto
                padding_size = 0
                if world_size > 1 and original_size % world_size != 0:
                    padding_size = world_size - (original_size % world_size)
                    padded_batch = {}
                    for k, v in gen_input_proto.batch.items():
                        # Use the single sequence as padding template
                        pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
                        padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
                    padded_gen_input_proto = DataProto.from_dict(padded_batch)
                    # Copy meta_info if needed
                    if hasattr(gen_input_proto, 'meta_info'):
                         padded_gen_input_proto.meta_info = gen_input_proto.meta_info.copy()
                # <<< END FIX >>>

                # --- Prepare Generation Config --- 
                generation_config = GenerationConfig(
                    max_new_tokens=self.config.max_response_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=1.0, # Consider adjusting temperature/sampling based on validation vs training
                    do_sample=True
                )

                # <<< FIX: Move generation_config into meta_info >>>
                if not hasattr(padded_gen_input_proto, 'meta_info'):
                    padded_gen_input_proto.meta_info = {}
                padded_gen_input_proto.meta_info['generation_config'] = generation_config
                # <<< END FIX >>>

                # Generation happens on the actor worker group's device
                gen_output_proto = self.actor_rollout_wg.generate_sequences(padded_gen_input_proto)
                # response_ids = gen_output_proto.batch['response_ids'] # Original line causing KeyError
                response_ids = gen_output_proto.batch['responses'] # Use the correct key ('responses') assuming it holds IDs

                # <<< FIX: Remove padding from response_ids if padding was added >>>
                if padding_size > 0:
                     response_ids = response_ids[:-padding_size]
                # <<< END FIX >>>

                # Decode the response IDs to get the text for the trajectory
                response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

                # print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Response: {response_text[:100]}...")
                trajectory.append({"from": "gpt", "value": response_text})

                # Post-process response to get action
                action_types, action_contents = self.postprocess_predictions([response_text])
                action_text = action_contents[0]

                # Execute environment step using the provided client
                if action_text is None: action_text = ""
                # print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Action: {action_text}")
                # next_obs_text, reward, done, info = client.step(action_text) # Original unpacking (fails)
                step_output = client.step(action_text)
                next_obs_text = step_output.state
                reward = step_output.reward
                done = step_output.done
                info = {} # Initialize info as empty dict, as StepOutput doesn't explicitly return it
                # print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Env Step Result: Reward={reward}, Done={done}, Info={info}")

                # Store the reward from this specific step
                step_rewards.append(reward)
                final_reward = reward # Keep track of the reward from the last executed step
                final_env_score = info.get('score', 0.0) # Use .get for safety

                # Add reward and info to the trajectory for this agent step
                # This helps the RewardComposer access step-specific info if needed
                trajectory[-1]['reward'] = reward
                trajectory[-1]['info'] = info

                # Process next observation
                if not done:
                    # print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Next Obs: {next_obs_text[:100]}...")
                    trajectory.append({"from": "human", "value": next_obs_text})
                    next_obs_ids = self.tokenizer(next_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']
                    # Ensure tensors are concatenated on the same device (e.g., CPU or model's device if needed later)
                    current_input_ids = torch.cat([
                        current_input_ids.to(response_ids.device), # Move to same device as response_ids
                        response_ids,
                        next_obs_ids.to(response_ids.device) # Move to same device
                    ], dim=1)
                else:
                    # print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Done received.")
                    break

        except Exception as e:
            print(f"[Agent._run_single_rollout][{task_idx} @ {getattr(client, 'env_server_base', 'unknown_client')}] Error during rollout: {e}")
            print(traceback.format_exc())
            # Reset results on error
            trajectory = trajectory # Keep partial trajectory for debugging?
            step_rewards = []
            final_reward = 0.0
            final_env_score = 0.0
            done = True # Mark as done on error

        # Return the collected information
        return {
            'trajectory': trajectory,        # Full interaction history
            'step_rewards': step_rewards,    # List of rewards from each env.step call
            'reward': final_reward,          # Reward from the *last* env.step call
            'env_score': final_env_score,    # Final score reported by env info
            'turns': turns,
            'task_idx': task_idx,
            'done': done                   # Whether the episode finished naturally or via error
        }

    def run_llm_loop(self, gen_batch: DataProto, output_dir: str = None, global_steps: int = 0) -> DataProto:
        """
        Run the LLM interaction loop for a batch of initial prompts using multiple clients.

        Args:
            gen_batch: DataProto containing initial prompts
            output_dir: Directory to save visualizations
            global_steps: Current training step

        Returns:
            DataProto containing processed results
        """
        initial_prompts_ids = gen_batch.batch['input_ids']
        batch_size = initial_prompts_ids.shape[0]
        num_clients = len(self.clients)
        if num_clients == 0:
             raise RuntimeError("No environment clients available for rollout.")

        print(f"[Agent.run_llm_loop] Starting rollout for batch size: {batch_size} using {num_clients} clients.")

        # --- Setup Visualization ---
        trajectory = self._setup_visualization()

        # --- Extract Task Indices ---
        if 'idx' in gen_batch.meta_info:
            task_idxs = gen_batch.meta_info['idx']
            if isinstance(task_idxs, torch.Tensor):
                task_idxs = task_idxs.tolist()
            if len(task_idxs) != batch_size:
                 print(f"[Agent.run_llm_loop] Warning: Mismatch between batch size ({batch_size}) and provided indices ({len(task_idxs)}). Using range(batch_size)." )
                 task_idxs = list(range(batch_size))
        else:
            print("[Agent.run_llm_loop] Warning: 'idx' not found in gen_batch.meta_info. Using range(batch_size)." )
            task_idxs = list(range(batch_size))

        # --- Parallel Rollout Execution ---
        futures = {}
        rollout_results_list = [None] * batch_size  # Preallocate list to store results in order

        # Submit tasks to the thread pool, distributing across clients
        for i in range(batch_size):
            task_idx = task_idxs[i]
            initial_prompt = initial_prompts_ids[i:i+1]  # Keep batch dim

            # Select a client for this task (round-robin)
            client_index = i % num_clients
            selected_client = self.clients[client_index]

            # Submit the rollout task with the selected client
            future = self.executor.submit(self._run_single_rollout, initial_prompt, task_idx, selected_client)
            futures[future] = i  # Store original batch index

        print(f"[Agent.run_llm_loop] Submitted {batch_size} rollout tasks to {self.executor._max_workers} workers.")

        # Collect results
        completed_count = 0
        for future in as_completed(futures):
            original_index = futures[future]
            try:
                result_dict = future.result()
                rollout_results_list[original_index] = result_dict
                completed_count += 1
                # print(f"Completed task {original_index + 1}/{batch_size}") # Optional progress logging

                # If visualization is enabled, update trajectory
                # Note: Visualization logic might need adjustment if envs are not easily accessible
                # or if you want per-client visualization. Current logic assumes a single env list.
                # Consider passing necessary env state back from _run_single_rollout if needed.
                # if trajectory and original_index < len(trajectory):
                #     # This part might be tricky with multiple clients unless you manage env state carefully
                #     pass # Placeholder for potential visualization update logic

            except Exception as e:
                print(f"[Agent.run_llm_loop] Error collecting result for batch index {original_index} (task_idx {task_idxs[original_index]}): {e}")
                print(traceback.format_exc())
                # Store a placeholder or error indicator
                rollout_results_list[original_index] = {
                    'trajectory': [], 'step_rewards': [], 'reward': 0.0,
                    'turns': 0, 'env_score': 0.0, 'task_idx': task_idxs[original_index],
                    'error': str(e)
                }

        print(f"[Agent.run_llm_loop] Collected results from {completed_count}/{batch_size} rollouts.")

        # Save trajectory visualizations (if implemented and needed)
        # if output_dir and trajectory:
        #     self._save_trajectory(trajectory, output_dir, global_steps)

        # Filter out potential None entries if some tasks failed critically
        valid_results = [res for res in rollout_results_list if res is not None]

        if not valid_results:
            print("[Agent.run_llm_loop] Error: No valid rollout results collected.")
            # Return empty DataProto but with correct structure if possible
            return DataProto.from_dict({
                "input_ids": torch.empty((0,0), dtype=torch.long),
                "attention_mask": torch.empty((0,0), dtype=torch.long),
                "position_ids": torch.empty((0,0), dtype=torch.long),
                "info_mask": torch.empty((0,0), dtype=torch.long),
                "token_level_rewards": torch.empty((0,0), dtype=torch.float)
            })

        # --- Format Results into DataProto ---
        processed_data = self._convert_rollout_results_to_dataproto(valid_results, gen_batch)

        print(f"[Agent.run_llm_loop] Finished processing rollout results.")
        return processed_data

    def _convert_rollout_results_to_dataproto(self, results: List[Dict], original_batch: DataProto) -> DataProto:
        """
        Convert the list of dictionaries (each containing trajectory, step_rewards, env_score)
        from the internal rollout loop into a DataProto suitable for PPO training.
        Creates 'token_level_rewards' based on the chosen reward allocation strategy.

        Args:
            results: List of result dictionaries from _run_single_rollout.
            original_batch: Original batch DataProto with metadata.

        Returns:
            DataProto: Processed data with token-level rewards and metadata.
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_info_mask = []
        batch_token_level_rewards = [] # Store final token-level rewards for PPO
        batch_meta_info = defaultdict(list)
        batch_responses = [] # Initialize batch_responses

        # Get reward allocation strategy from config
        reward_allocation = "last_token" # Default
        if self.config.algorithm_config:
            reward_allocation = self.config.algorithm_config.get('reward_allocation', 'last_token')
        print(f"[Agent._convert_rollout] Using reward allocation strategy: {reward_allocation}")

        # Get the index mapping from the original batch
        original_indices = original_batch.meta_info.get('idx', list(range(original_batch.batch['input_ids'].shape[0])))
        if isinstance(original_indices, torch.Tensor):
            original_indices = original_indices.tolist()
        original_indices_map = {idx_val: i for i, idx_val in enumerate(original_indices)}

        print(f"[Agent._convert_rollout] Formatting {len(results)} trajectories.")
        for result_dict in results:
            # Extract trajectory and other info
            trajectory = result_dict.get('trajectory', [])
            # IMPORTANT: Decide which reward signal to use for allocation.
            # Option 1: Use the final env score (often 0/1 for success)
            # reward_to_distribute = result_dict.get('env_score', 0.0)
            # Option 2: Use the reward from the last step
            # reward_to_distribute = result_dict.get('reward', 0.0)
            # Option 3: Use the sum/average of step_rewards (less common for final goal tasks)
            step_rewards_list = result_dict.get('step_rewards', [])
            # Let's use final_env_score as the primary signal for allocation, as it usually reflects task completion.
            reward_to_distribute = result_dict.get('env_score', 0.0)

            turns = result_dict.get('turns', 0)
            task_idx = result_dict.get('task_idx', -1)

            # Get the original batch index
            original_batch_idx = original_indices_map.get(task_idx, -1)
            if original_batch_idx == -1:
                print(f"[Agent._convert_rollout] Warning: Task idx {task_idx} not found in original batch. Skipping.")
                continue

            # --- Concatenate conversation and identify agent segments --- 
            conversation_ids_list = []
            info_mask_parts = []
            segment_lengths = [] # Store length of each segment (human/gpt)
            agent_response_indices = [] # Store indices of agent responses (in the segment list)
            valid_actions = 0 # Count of agent turns

            if not trajectory:
                 # ... (handle empty trajectory) ...
                 initial_prompt_ids = original_batch.batch['input_ids'][original_batch_idx:original_batch_idx+1]
                 conversation_ids_list.append(initial_prompt_ids)
                 info_mask_parts.append(torch.ones_like(initial_prompt_ids))
                 segment_lengths.append(initial_prompt_ids.shape[1])
            else:
                for turn_idx, msg in enumerate(trajectory):
                    msg_text = msg.get("value", "")
                    msg_from = msg.get("from", "")
                    if not msg_text: continue

                    msg_ids = self.tokenizer(msg_text, add_special_tokens=False, return_tensors='pt')['input_ids']
                    conversation_ids_list.append(msg_ids)
                    segment_lengths.append(msg_ids.shape[1])

                    if msg_from == "gpt":
                        info_mask_parts.append(torch.ones_like(msg_ids))
                        valid_actions += 1
                        agent_response_indices.append(len(conversation_ids_list) - 1)
                    else: # human or other
                        info_mask_parts.append(torch.ones_like(msg_ids))

            if not conversation_ids_list:
                print(f"[Agent._convert_rollout] Warning: No valid conversation segments for task_idx {task_idx}. Skipping.")
                continue

            # --- Pad and Truncate --- 
            # ... (Padding and truncation logic remains the same) ...
            full_input_ids = torch.cat(conversation_ids_list, dim=1)
            full_info_mask = torch.cat(info_mask_parts, dim=1)
            seq_len = full_input_ids.shape[1]
            target_len = self.config.max_prompt_length # Or another max len? Check this
            padding_len = max(0, target_len - seq_len)
            agent_indices_in_padded = [] # List of (start, end) indices for agent tokens in the final padded tensor

            if seq_len > target_len:
                # Truncate left
                removed_len = seq_len - target_len
                current_removed = 0
                first_segment_idx = 0
                while current_removed < removed_len and first_segment_idx < len(segment_lengths):
                    len_to_remove = min(segment_lengths[first_segment_idx], removed_len - current_removed)
                    segment_lengths[first_segment_idx] -= len_to_remove
                    current_removed += len_to_remove
                    if segment_lengths[first_segment_idx] == 0:
                        first_segment_idx += 1
                
                # Adjust agent response indices
                adjusted_agent_response_indices = [idx - first_segment_idx for idx in agent_response_indices if idx >= first_segment_idx]
                segment_lengths = segment_lengths[first_segment_idx:]
                
                full_input_ids = full_input_ids[:, -target_len:]
                full_info_mask = full_info_mask[:, -target_len:]
                seq_len = target_len
                padding_len = 0 # No padding needed after truncation
            elif seq_len < target_len:
                # Pad left
                pad_tensor = torch.full((1, padding_len), self.tokenizer.pad_token_id, dtype=torch.long, device=full_input_ids.device)
                full_input_ids = torch.cat([pad_tensor, full_input_ids], dim=1)
                info_pad = torch.zeros_like(pad_tensor) # Padding is masked in info
                full_info_mask = torch.cat([info_pad, full_info_mask], dim=1)
                adjusted_agent_response_indices = agent_response_indices # Indices remain the same relative to segments

            # Calculate agent token indices in the *final* padded/truncated tensor
            current_token_idx_in_padded = padding_len
            for segment_idx, length in enumerate(segment_lengths):
                 is_agent_response = segment_idx in adjusted_agent_response_indices
                 start_idx = current_token_idx_in_padded
                 end_idx = current_token_idx_in_padded + length - 1
                 if is_agent_response and length > 0:
                      agent_indices_in_padded.append((start_idx, end_idx))
                 current_token_idx_in_padded += length

            # --- Create Token Level Rewards Tensor based on Allocation Strategy --- 
            token_level_rewards = torch.zeros_like(full_input_ids, dtype=torch.float32)

            if agent_indices_in_padded: # Only allocate if there are agent responses
                if reward_allocation == "last_token":
                    # Assign reward only to the last token of the last agent segment
                    last_segment_start, last_segment_end = agent_indices_in_padded[-1]
                    if last_segment_end < target_len: # Ensure index is within bounds
                        token_level_rewards[0, last_segment_end] = reward_to_distribute

                elif reward_allocation == "uniform_positive":
                    # Distribute positive rewards evenly across all agent tokens
                    if reward_to_distribute > 0:
                        total_agent_tokens = sum(end - start + 1 for start, end in agent_indices_in_padded)
                        reward_per_token = reward_to_distribute / max(1, total_agent_tokens)
                        for start, end in agent_indices_in_padded:
                            token_level_rewards[0, start : end + 1] = reward_per_token
                    # Negative rewards are assigned to the last token (or ignored)
                    elif reward_to_distribute < 0:
                         last_segment_start, last_segment_end = agent_indices_in_padded[-1]
                         if last_segment_end < target_len:
                              token_level_rewards[0, last_segment_end] = reward_to_distribute

                elif reward_allocation == "discounted":
                    # Distribute reward starting from the last agent segment, discounted backward
                    gamma = self.config.algorithm_config.get('gamma', 1.0) if self.config.algorithm_config else 1.0
                    current_reward = reward_to_distribute
                    # Iterate segments backward
                    for start, end in reversed(agent_indices_in_padded):
                        segment_len = end - start + 1
                        # Simple example: distribute reward uniformly within the segment
                        reward_for_segment = current_reward / segment_len
                        token_level_rewards[0, start : end + 1] = reward_for_segment
                        # Apply discount for the *next* (earlier) segment
                        current_reward *= (gamma ** segment_len)
                else:
                     print(f"[Agent._convert_rollout] Warning: Unknown reward_allocation strategy '{reward_allocation}'. Defaulting to last_token.")
                     last_segment_start, last_segment_end = agent_indices_in_padded[-1]
                     if last_segment_end < target_len:
                         token_level_rewards[0, last_segment_end] = reward_to_distribute

            # --- Create Attention Mask and Position IDs --- 
            full_attention_mask = self.tensor_fn.create_attention_mask(full_input_ids)
            full_position_ids = self.tensor_fn.create_position_ids(full_attention_mask)

            # --- Store Processed Data --- 
            batch_input_ids.append(full_input_ids)
            batch_attention_mask.append(full_attention_mask)
            batch_position_ids.append(full_position_ids)
            batch_info_mask.append(full_info_mask) # Store the info mask
            batch_token_level_rewards.append(token_level_rewards) # Store calculated rewards

            # --- FIX: Extract and pad response-only tokens ---
            response_segments = []
            total_response_len = 0
            for r_start, r_end in agent_indices_in_padded:
                segment = full_input_ids[0, r_start : r_end + 1]
                response_segments.append(segment)
                total_response_len += segment.shape[0]
            
            if response_segments:
                response_only_ids_cat = torch.cat(response_segments, dim=0).unsqueeze(0) # Shape (1, total_response_len)
                resp_pad_len = max(0, self.config.max_response_length - total_response_len)
                if resp_pad_len > 0:
                    resp_pad = torch.full((1, resp_pad_len), self.tokenizer.pad_token_id, dtype=torch.long, device=response_only_ids_cat.device)
                    response_only_ids_padded = torch.cat([response_only_ids_cat, resp_pad], dim=1)
                else:
                    # Truncate if response is too long
                    response_only_ids_padded = response_only_ids_cat[:, :self.config.max_response_length]
            else:
                # Handle case with no agent responses (e.g., empty trajectory)
                response_only_ids_padded = torch.full((1, self.config.max_response_length), self.tokenizer.pad_token_id, dtype=torch.long, device=full_input_ids.device)
            
            # Append to batch list (this will be concatenated later)
            # Ensure batch_responses list is initialized outside the loop
            if 'batch_responses' not in locals() and 'batch_responses' not in globals():
                 batch_responses = [] # Initialize here if first iteration
            batch_responses.append(response_only_ids_padded)
            # --- END FIX ---

            # Add metadata (ensure reward/env_score reflect the values used for distribution if needed)
            batch_meta_info["task_idx"].append(task_idx)
            batch_meta_info["turns_stats"].append(turns)
            batch_meta_info["valid_action_stats"].append(valid_actions)
            batch_meta_info["reward"].append(result_dict.get('reward', 0.0)) # Last step reward
            batch_meta_info["env_score"].append(result_dict.get('env_score', 0.0)) # Final env score
            batch_meta_info["rollout_trajectory"].append(trajectory)
            # Add other relevant metadata from original_batch
            # ... (metadata copying logic as before) ...
            for key, value in original_batch.meta_info.items():
                 if key not in ['idx', 'reward', 'env_score']: # Avoid duplication
                      if isinstance(value, list) and len(value) > original_batch_idx:
                           batch_meta_info[key].append(value[original_batch_idx])
                      elif not isinstance(value, list): # Keep non-list metadata
                           if task_idx == original_indices[0]: # Add only once per batch
                                batch_meta_info[key] = value

        # --- Stack Tensors --- 
        if not batch_input_ids:
            print("[Agent._convert_rollout] No valid trajectories formatted. Returning empty DataProto.")
            # Return structure matching trainer expectations, even if empty
            return DataProto.from_dict({
                "input_ids": torch.empty((0,0), dtype=torch.long),
                "attention_mask": torch.empty((0,0), dtype=torch.long),
                "position_ids": torch.empty((0,0), dtype=torch.long),
                "info_mask": torch.empty((0,0), dtype=torch.long),
                "token_level_rewards": torch.empty((0,0), dtype=torch.float)
            })

        # Create final batch data
        final_batch = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "position_ids": torch.cat(batch_position_ids, dim=0),
            "info_mask": torch.cat(batch_info_mask, dim=0),
            "token_level_rewards": torch.cat(batch_token_level_rewards, dim=0), # Crucial output for PPO
            "responses": torch.cat(batch_responses, dim=0) # FIX: Add the collected responses
        }

        # Create DataProto and add metadata
        data_proto = DataProto.from_dict(final_batch)
        # ... (metadata handling as before, converting lists to tensors where appropriate) ...
        for key, value in batch_meta_info.items():
            try:
                if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
                    data_proto.meta_info[key] = torch.tensor(value)
                # Handle numpy arrays if they appear
                elif isinstance(value, np.ndarray):
                     data_proto.meta_info[key] = torch.from_numpy(value)
                else:
                    # Keep as list for non-numeric types (like trajectories)
                    data_proto.meta_info[key] = value
            except (ValueError, TypeError, RuntimeError) as e:
                 # Fallback: keep as list if tensor conversion fails
                 print(f"[Agent._convert_rollout] Warning: Could not convert metadata '{key}' to tensor: {e}. Keeping as list.")
                 data_proto.meta_info[key] = value

        # Explicitly add final env scores as a tensor if possible
        if "env_score" in batch_meta_info:
            try:
                data_proto.meta_info["env_scores"] = torch.tensor(batch_meta_info["env_score"], dtype=torch.float32)
            except (ValueError, TypeError):
                # This case should be less likely now, but keep fallback
                print("[Agent._convert_rollout] Could not convert env_scores to tensor, keeping original list.")
                data_proto.meta_info["env_scores"] = batch_meta_info["env_score"]

        print(f"[Agent._convert_rollout] Final batch shapes: input_ids={final_batch['input_ids'].shape}, token_level_rewards={final_batch['token_level_rewards'].shape}")
        return data_proto


    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
        """
        Process predictions into actions and content based on XML-like tags.
        Does not require tool_manager.

        Args:
            predictions: List of raw predictions (strings from LLM)

        Returns:
            Tuple of (action types list ['action' or 'response' or None],
                    action contents list [text inside tags or empty string])
        """
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str):
                # Extract action or response tags
                action_pattern = r'<action>(.*?)</action>'
                response_pattern = r'<response>(.*?)</response>'

                action_match = re.search(action_pattern, prediction, re.DOTALL)
                response_match = re.search(response_pattern, prediction, re.DOTALL)

                if action_match:
                    actions.append('action')
                    contents.append(action_match.group(1).strip())
                elif response_match:
                    actions.append('response')
                    contents.append(response_match.group(1).strip())
                else:
                    # If no recognized tag, assume it's neither a specific action nor response
                    actions.append(None)
                    contents.append('') # Return empty content if no tag found
            else:
                # Handle non-string predictions if necessary, e.g., raise error or log warning
                print(f"[Warning] Received non-string prediction: {type(prediction)}. Cannot process.")
                actions.append(None)
                contents.append('')
                # Or raise ValueError(f"Invalid prediction type: {type(prediction)}")

        return actions, contents