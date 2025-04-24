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
from ragen.utils.plot import (
    save_trajectory_to_output,
    parse_llm_output
)
from verl.utils.tracking import Tracking

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
        react_format: Whether to use ReAct format
        env_name: Name of the environment (e.g., "webshop")
        env_ports: List of ports for parallel servers
        env_server_base: Base URL for environment server
        env_data_len: Number of data samples in the environment (used for client init)
        rollout_strategy: Strategy to use for rollout (StandardReAct/ToT/MCTS)
        storage_backend: Backend for storing trajectories (mongodb/file)
        max_workers: Maximum number of worker threads
        logging: dict = None  # Contains log_images, log_n_image_per_batch, log_image_step_size, etc.
    """
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    react_format: bool = True
    
    # Environment configuration (Now passed from trainer)
    env_name: str 
    env_ports: List[int] # List of ports for parallel servers
    env_server_base: str
    env_data_len: int = 200 # Default, might need adjustment
    rollout_strategy: str = "StandardReAct" # Strategy is now internal logic
    # storage_backend: str = "mongodb" # Storage handled elsewhere or not needed here
    max_workers: int = 10 # For parallelizing rollouts within the agent
    
    # Add visualization-related configuration
    logging: dict = None  # Contains log_images, log_n_image_per_batch, log_image_step_size, etc.

def create_react_prompt(task_description, tool_manager):
    """
    Create a prompt for the agent using ReAct format.
    
    Args:
        task_description: Description of the specific task
        tool_manager: ToolManager instance with registered tools
        
    Returns:
        Formatted prompt string
    """
    tools_instructions = tool_manager.get_prompt_instructions()
    
    prompt = f"""# Task
{task_description}

# Instructions
{tools_instructions}

Let's solve this step by step.

"""
    return prompt

class OpenManusAgent:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg, # This is the Verl component for generation
        config: AgentConfig,
        tool_manager, # Keep for potential parsing, but execution is via env
        is_validation: bool = False,
        logger: Tracking = None,  # Add logger parameter for trajectory saving
    ):
        """
        Initialize OpenManusAgent with rollout controller integration.
        
        Args:
            tokenizer: Tokenizer for text processing
            actor_rollout_wg: Actor rollout wrapper for generation
            config: Agent configuration including env details
            tool_manager: Manager for tool operations (potentially unused)
            is_validation: Whether in validation mode
            logger: Logger for tracking and visualization
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.tool_manager = tool_manager
        self.is_validation = is_validation
        self.logger = logger  # Add logger attribute

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

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate tensors and handle padding with info mask."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

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
        
        Args:
            initial_prompt_ids: Token IDs for the initial prompt/observation.
            task_idx: The index for resetting the environment.
            client: The specific environment client instance to use for this rollout.
            
        Returns:
            A dictionary containing the trajectory, step rewards, final reward, turns,
            final env score, and original task index.
        """
        trajectory = []
        step_rewards = []  # Store rewards per step
        final_reward = 0.0 
        final_env_score = 0.0
        done = False
        turns = 0
        current_input_ids = None 

        try:
            # Reset environment using the provided client
            client.reset(task_idx)
            initial_obs_text = client.observe()
            
            # Handle initial observation
            if not initial_obs_text:
                print(f"[Agent._run_single_rollout][{task_idx} @ {client.env_server_base}] Warning: Received empty initial observation. Using initial prompt from batch.")
                initial_prompt_text = self.tokenizer.decode(initial_prompt_ids[0], skip_special_tokens=True)
                trajectory.append({"from": "human", "value": initial_prompt_text})
                current_input_ids = initial_prompt_ids
            else:
                trajectory.append({"from": "human", "value": initial_obs_text})
                current_input_ids = self.tokenizer(initial_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']
            
            # --- Interaction Loop --- 
            for t in range(self.config.max_turns):
                turns = t + 1
                if current_input_ids is None: break 
                
                # Handle input that exceeds max length
                if current_input_ids.shape[1] > self.config.max_prompt_length:
                    current_input_ids = current_input_ids[:, -self.config.max_prompt_length:]
                    print(f"[Agent._run_single_rollout][{task_idx} @ {client.env_server_base}] Warning: Truncating input {current_input_ids.shape} > {self.config.max_prompt_length}.")

                # Prepare input
                current_attention_mask = self.tensor_fn.create_attention_mask(current_input_ids)
                current_position_ids = self.tensor_fn.create_position_ids(current_attention_mask)
                # Ensure input tensors are on the correct device for the actor model
                device = next(self.actor_rollout_wg.actor_model.parameters()).device # Get model's device
                gen_input_proto = DataProto.from_dict({
                    'input_ids': current_input_ids.to(device),
                    'attention_mask': current_attention_mask.to(device),
                    'position_ids': current_position_ids.to(device)
                })
                
                # Generate response
                generation_config = GenerationConfig(
                    max_new_tokens=self.config.max_response_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=1.0, 
                    do_sample=True   
                )
                # Generation happens on the actor worker group's device
                gen_output_proto = self.actor_rollout_wg.generate_sequences(gen_input_proto, generation_config=generation_config)
                response_ids = gen_output_proto.batch['response_ids'] 
                response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
                trajectory.append({"from": "gpt", "value": response_text})

                # Post-process response to get action
                action_types, action_contents = self.postprocess_predictions([response_text])
                action_text = action_contents[0] 
                
                # Execute environment step using the provided client
                if action_text is None: action_text = "" 
                next_obs_text, reward, done, info = client.step(action_text)
                
                # Record rewards
                step_rewards.append(reward)
                final_reward = reward 
                final_env_score = info.get('score', 0.0) # Use .get for safety

                # Process next observation
                if not done:
                    trajectory.append({"from": "human", "value": next_obs_text})
                    next_obs_ids = self.tokenizer(next_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']
                    # Ensure tensors are concatenated on the same device (e.g., CPU or model's device if needed later)
                    current_input_ids = torch.cat([
                        current_input_ids.to(response_ids.device), # Move to same device as response_ids
                        response_ids, 
                        next_obs_ids.to(response_ids.device) # Move to same device
                    ], dim=1)
                else:
                    break 

        except Exception as e:
            print(f"[Agent._run_single_rollout][{task_idx} @ {getattr(client, 'env_server_base', 'unknown_client')}] Error during rollout: {e}")
            print(traceback.format_exc())
            step_rewards = []
            final_reward = 0.0 
            final_env_score = 0.0
            done = True

        return {
            'trajectory': trajectory, 
            'step_rewards': step_rewards,
            'reward': final_reward, 
            'turns': turns, 
            'env_score': final_env_score,
            'task_idx': task_idx
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
        Creates 'token_level_rewards' based on step_rewards.
        
        Args:
            results: List of result dictionaries from rollout
            original_batch: Original batch DataProto with metadata
            
        Returns:
            DataProto: Processed data with rewards and metadata
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_info_mask = [] 
        batch_rewards = [] # Store final rewards
        batch_token_level_rewards = [] # Store step rewards aligned with tokens
        batch_meta_info = defaultdict(list)

        # Get the index mapping from the original batch
        original_indices = original_batch.meta_info.get('idx', list(range(original_batch.batch['input_ids'].shape[0])))
        if isinstance(original_indices, torch.Tensor):
            original_indices = original_indices.tolist()
        original_indices_map = {idx_val: i for i, idx_val in enumerate(original_indices)}

        print(f"[Agent._convert_rollout] Formatting {len(results)} trajectories.")
        for result_dict in results:
            # Extract trajectory and reward information
            trajectory = result_dict.get('trajectory', [])
            step_rewards_list = result_dict.get('step_rewards', [])
            final_reward = result_dict.get('reward', 0.0)
            turns = result_dict.get('turns', 0)
            env_score = result_dict.get('env_score', 0.0)
            task_idx = result_dict.get('task_idx', -1)
            
            # Get the original batch index
            original_batch_idx = original_indices_map.get(task_idx, -1)
            if original_batch_idx == -1: 
                print(f"[Agent._convert_rollout] Warning: Task idx {task_idx} not found in original batch. Skipping.")
                continue
                
            # --- Concatenate conversation and align rewards --- 
            conversation_ids_list = []
            info_mask_parts = []
            segment_lengths = [] # Store length of each segment (human/gpt)
            agent_response_indices = [] # Store indices of agent responses
            valid_actions = 0

            if not trajectory:
                 print(f"[Agent._convert_rollout] Warning: Empty trajectory for task_idx {task_idx}. Using initial prompt only.")
                 # If trajectory is empty, use original prompt
                 initial_prompt_ids = original_batch.batch['input_ids'][original_batch_idx:original_batch_idx+1]
                 conversation_ids_list.append(initial_prompt_ids)
                 info_mask_parts.append(torch.ones_like(initial_prompt_ids))
                 segment_lengths.append(initial_prompt_ids.shape[1])
            else:
                # Process each turn in the trajectory
                for turn_idx, msg in enumerate(trajectory):
                    msg_text = msg.get("value", "")
                    msg_from = msg.get("from", "")
                    if not msg_text: continue
                    
                    # Convert text to token ids
                    msg_ids = self.tokenizer(msg_text, add_special_tokens=False, return_tensors='pt')['input_ids']
                    conversation_ids_list.append(msg_ids)
                    segment_lengths.append(msg_ids.shape[1])
                    
                    # Distinguish between agent responses and environment observations
                    if msg_from == "gpt":
                        info_mask_parts.append(torch.ones_like(msg_ids)) 
                        valid_actions += 1 
                        agent_response_indices.append(len(conversation_ids_list) - 1) # Store index of this segment
                    else: 
                        info_mask_parts.append(torch.ones_like(msg_ids)) 
            
            # Concatenate, Pad, Truncate (Input IDs, Info Mask)
            if not conversation_ids_list:
                print(f"[Agent._convert_rollout] Warning: No valid conversation segments for task_idx {task_idx}. Skipping.")
                continue
                
            # Concatenate all conversation segments
            full_input_ids = torch.cat(conversation_ids_list, dim=1)
            full_info_mask = torch.cat(info_mask_parts, dim=1)
            seq_len = full_input_ids.shape[1]
            target_len = self.config.max_prompt_length 
            padding_len = max(0, target_len - seq_len)

            if seq_len > target_len:
                # Truncate from left - need to adjust segment_lengths and indices
                removed_len = seq_len - target_len
                current_removed = 0
                first_segment_idx = 0
                while current_removed < removed_len and first_segment_idx < len(segment_lengths):
                    len_to_remove = min(segment_lengths[first_segment_idx], removed_len - current_removed)
                    segment_lengths[first_segment_idx] -= len_to_remove
                    current_removed += len_to_remove
                    if segment_lengths[first_segment_idx] == 0:
                        first_segment_idx += 1
                
                # Adjust agent response indices if segments were removed
                agent_response_indices = [idx for idx in agent_response_indices if idx >= first_segment_idx]
                # Recalculate indices relative to the truncated start
                agent_response_indices = [idx - first_segment_idx for idx in agent_response_indices]
                # Update segment_lengths list
                segment_lengths = segment_lengths[first_segment_idx:]
                
                # Truncate input_ids and info_mask
                full_input_ids = full_input_ids[:, -target_len:]
                full_info_mask = full_info_mask[:, -target_len:]
                seq_len = target_len # Update sequence length

            elif seq_len < target_len:
                # Pad left (Input IDs)
                pad_tensor = torch.full((1, padding_len), self.tokenizer.pad_token_id, dtype=torch.long, device=full_input_ids.device)
                full_input_ids = torch.cat([pad_tensor, full_input_ids], dim=1) 
                # Pad left (Info Mask)
                info_pad = torch.zeros_like(pad_tensor) # Padding is masked
                full_info_mask = torch.cat([info_pad, full_info_mask], dim=1)
            
            # --- Create Token Level Rewards Tensor --- 
            token_level_rewards = torch.zeros_like(full_input_ids, dtype=torch.float32)
            
            # If there are step rewards, assign them to appropriate tokens
            if step_rewards_list:
                current_token_idx_in_unpadded = 0 
                agent_turn_reward_idx = 0
                for segment_idx, length in enumerate(segment_lengths):
                    if length == 0: continue # Skip segments that were fully truncated
                    
                    # Check if this segment corresponds to an agent response
                    is_agent_response = segment_idx in agent_response_indices
                    
                    if is_agent_response and agent_turn_reward_idx < len(step_rewards_list):
                        # Assign reward for this step
                        reward_for_this_step = step_rewards_list[agent_turn_reward_idx]
                        # Assign reward to the last token of this agent segment
                        end_idx_in_unpadded = current_token_idx_in_unpadded + length - 1
                        actual_end_idx_in_padded = padding_len + end_idx_in_unpadded # Adjust for padding
                        if actual_end_idx_in_padded < target_len:
                            token_level_rewards[0, actual_end_idx_in_padded] = reward_for_this_step
                        agent_turn_reward_idx += 1
                    
                    current_token_idx_in_unpadded += length
            
            # --- Add reward shaping variations, supporting multiple reward distribution methods ---
            # 1. If there's only one reward, distribute it across all agent response tokens
            if len(step_rewards_list) == 1 and valid_actions > 0:
                # Distribute reward across all agent response tokens
                reward_value = step_rewards_list[0] / max(1, valid_actions)
                # Identify agent response tokens where info_mask is 1
                agent_token_mask = (full_info_mask == 1)
                token_level_rewards = torch.where(agent_token_mask, 
                                                torch.full_like(token_level_rewards, reward_value), 
                                                token_level_rewards)

            # --- Create Attention Mask and Position IDs --- 
            full_attention_mask = self.tensor_fn.create_attention_mask(full_input_ids)
            full_position_ids = self.tensor_fn.create_position_ids(full_attention_mask)

            # --- Store Processed Data --- 
            batch_input_ids.append(full_input_ids)
            batch_attention_mask.append(full_attention_mask)
            batch_position_ids.append(full_position_ids)
            batch_info_mask.append(full_info_mask)
            batch_token_level_rewards.append(token_level_rewards) # Store rewards tensor
            batch_rewards.append(final_reward) # Store final reward
            
            # Add metadata
            batch_meta_info["task_idx"].append(task_idx)
            batch_meta_info["turns_stats"].append(turns)
            batch_meta_info["valid_action_stats"].append(valid_actions)
            batch_meta_info["reward"].append(final_reward) # Last step reward
            batch_meta_info["env_score"].append(env_score) 
            batch_meta_info["rollout_trajectory"].append(trajectory) # Add trajectory list
            
            # --- Add reward_model information ---
            if 'reward_model' in original_batch.meta_info:
                if isinstance(original_batch.meta_info['reward_model'], list) and len(original_batch.meta_info['reward_model']) > original_batch_idx:
                    # Assume reward_model is a list in the batch metadata
                    batch_meta_info["reward_model"].append(original_batch.meta_info['reward_model'][original_batch_idx])
                elif isinstance(original_batch.meta_info['reward_model'], dict):
                    # If reward_model is a single dict passed for the whole batch (less likely)
                    if original_batch_idx == 0: # Add only once
                        batch_meta_info["reward_model"] = original_batch.meta_info['reward_model']
            
            # Copy other relevant metadata from the original batch
            for key, value in original_batch.meta_info.items():
                # Avoid duplicating keys already handled (idx, reward, reward_model)
                if key not in ['idx', 'reward', 'reward_model']:
                    if isinstance(value, list) and len(value) > original_batch_idx:
                        batch_meta_info[key].append(value[original_batch_idx])
                    elif not isinstance(value, list): # Keep non-list metadata (add only once)
                        if original_batch_idx == 0: # Add only once
                            batch_meta_info[key] = value 

        # --- Stack Tensors --- 
        if not batch_input_ids: 
            print("[Agent._convert_rollout] No valid trajectories formatted. Returning empty DataProto.")
            return DataProto.from_dict({}) 
            
        # Create final batch data
        final_batch = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "position_ids": torch.cat(batch_position_ids, dim=0),
            "info_mask": torch.cat(batch_info_mask, dim=0), 
            "token_level_rewards": torch.cat(batch_token_level_rewards, dim=0) # Add stacked rewards tensor
        }
        
        # Create DataProto and add metadata
        data_proto = DataProto.from_dict(final_batch)
        for key, value in batch_meta_info.items():
            try:
                # Try to convert values to tensors
                if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
                    data_proto.meta_info[key] = torch.tensor(value)
                else:
                    data_proto.meta_info[key] = value
            except (ValueError, TypeError):
                data_proto.meta_info[key] = value 
        
        # Add rewards tensor
        data_proto.meta_info["rewards"] = torch.tensor(batch_rewards, dtype=torch.float32)
        
        # Explicitly add environment scores
        if "env_score" in batch_meta_info:
            try:
                data_proto.meta_info["env_scores"] = torch.tensor(batch_meta_info["env_score"], dtype=torch.float32)
            except (ValueError, TypeError):
                print("[Agent._convert_rollout] Could not convert env_scores to tensor, keeping as list.")
                data_proto.meta_info["env_scores"] = batch_meta_info["env_score"]
                 
        print(f"[Agent._convert_rollout] Final batch shapes: input_ids={final_batch['input_ids'].shape}, token_level_rewards={final_batch['token_level_rewards'].shape}")
        return data_proto

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, execute_tools=True) -> Tuple[List[str], List[bool], List[bool], List[bool]]:
        """
        Execute predictions (Placeholder - Actual execution handled by AgentGym Task via controller).
        
        This method is likely called by the RolloutController's strategy.
        In the AgentGym setup, the controller gets the prediction string from the agent 
        (via model generation) and passes it to the task's step function.
        This method here might only be needed for parsing or returning flags, not execution.
        
        Args:
            predictions: List of action predictions (strings from model)
            pad_token: Padding token
            active_mask: Mask for active sequences
            execute_tools: Whether to execute tools (Likely ignored)
            
        Returns:
            Placeholder tuple. The actual next_obs, dones, etc., come from the Task's step method.
        """
        # The original implementation had a recursive call to rollout_controller._rollout_one, which is incorrect.
        # The RolloutController orchestrates the flow: agent predicts -> controller passes prediction to task.step -> task executes -> task returns obs/done.
        # Therefore, this method in the agent should likely *not* execute tools or interact with the environment directly.
        
        # For now, return placeholder values. The actual values are determined by the Task environment.
        # We need to understand how the chosen Strategy uses this method, if at all.
        num_preds = len(predictions)
        dummy_obs = ["" for _ in range(num_preds)]
        dummy_dones = [False for _ in range(num_preds)] # Assume not done unless Task says otherwise
        dummy_valid = [True for _ in range(num_preds)] # Assume valid unless parsing fails
        dummy_tool_use = [False for _ in range(num_preds)] # Determine based on prediction parsing
        
        # Basic check if prediction looks like a tool call based on common patterns
        actions, _ = self.postprocess_predictions(predictions)
        for i, action_type in enumerate(actions):
            if action_type == 'action':
                dummy_tool_use[i] = True
                
        # If not using tools (e.g., final response), these flags might be different.
        # This part might need refinement based on how the strategy/trainer uses these return values.

        print(f"[Agent.execute_predictions] Received {num_preds} predictions. Returning placeholder env state.")
        return dummy_obs, dummy_dones, dummy_valid, dummy_tool_use

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
        """
        Process predictions into actions and content.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (action types list, action contents list)
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
                    actions.append(None)
                    contents.append('')
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
        return actions, contents