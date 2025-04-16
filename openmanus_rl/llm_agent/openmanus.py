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
        env_port: Port number for environment server
        env_server_base: Base URL for environment server
        env_data_len: Number of data samples in the environment (used for client init)
        rollout_strategy: Strategy to use for rollout (StandardReAct/ToT/MCTS)
        storage_backend: Backend for storing trajectories (mongodb/file)
        max_workers: Maximum number of worker threads
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
    env_port: int
    env_server_base: str
    env_data_len: int = 200 # Default, might need adjustment
    rollout_strategy: str = "StandardReAct" # Strategy is now internal logic
    # storage_backend: str = "mongodb" # Storage handled elsewhere or not needed here
    max_workers: int = 10 # For parallelizing rollouts within the agent

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
    ):
        """
        Initialize OpenManusAgent with rollout controller integration.
        
        Args:
            tokenizer: Tokenizer for text processing
            actor_rollout_wg: Actor rollout wrapper for generation
            config: Agent configuration including env details
            tool_manager: Manager for tool operations (potentially unused)
            is_validation: Whether in validation mode
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.tool_manager = tool_manager
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        # Initialize the environment client directly
        self.client = self._init_env_client()
        # Initialize thread pool for parallel rollouts
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

    def _init_env_client(self):
        """
        Initialize and return the specific AgentGym environment client based on config.
        """
        # Mapping from env_name (lowercase) to Task class name
        # We need the Task class to potentially get the right client or initial setup
        ENV_TO_TASK_CLASS = {
            "academia": "AcademiaTask",
            "alfworld": "AlfWorldTask",
            "babyai": "BabyAITask",
            "maze": "MazeTask", 
            "wordle": "WordleTask",
            "movie": "MovieTask",
            "sciworld": "SciworldTask",
            "sheet": "SheetTask",
            "sqlgym": "SqlGymTask",
            "textcraft": "TextCraftTask",
            "todo": "TodoTask",
            "weather": "WeatherTask",
            "webarena": "WebarenaTask",
            "webshop": "WebshopTask",
        }
        
        env_name_lower = self.config.env_name.lower()
        if env_name_lower not in ENV_TO_TASK_CLASS:
            raise ValueError(f"Unsupported environment name: {self.config.env_name}. Supported: {list(ENV_TO_TASK_CLASS.keys())}")

        task_class_name = ENV_TO_TASK_CLASS[env_name_lower]
        print(f"Initializing Env Client for: {self.config.env_name} (via Task: {task_class_name})")
        print(f"Connecting to AgentGym server at: {self.config.env_server_base}:{self.config.env_port}")

        # Dynamically import the Task class
        try:
            envs_module = importlib.import_module("agentenv.envs")
            TaskClass = getattr(envs_module, task_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import Task class {task_class_name} from agentenv.envs: {e}")

        client_args={
            "env_server_base": f"{self.config.env_server_base}:{self.config.env_port}",
            "data_len": self.config.env_data_len, 
            "timeout": 300, 
        }
        
        # Instantiate the task to get the client. 
        # Assuming Task object creates and holds the client(s) in a list `clients`.
        # This might need adjustment based on actual Task implementation.
        try:
            # We only need one client instance per agent worker typically.
            task_instance = TaskClass(client_args=client_args, n_clients=1)
            if hasattr(task_instance, 'clients') and task_instance.clients:
                client = task_instance.clients[0] 
                print(f"Successfully obtained client: {type(client)}")
                return client
            else:
                raise ValueError(f"Task class {task_class_name} did not provide a client in 'clients' attribute.")
        except Exception as e:
             print(f"Error initializing Task or getting client for {task_class_name}: {e}")
             print(traceback.format_exc()) # Print detailed traceback
             raise

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

    def _run_single_rollout(self, initial_prompt_ids: torch.Tensor, task_idx: int) -> Dict[str, Any]:
        """
        Runs the interaction loop for a single environment instance.

        Args:
            initial_prompt_ids: Token IDs for the initial prompt/observation.
            task_idx: The index for resetting the environment.

        Returns:
            A dictionary containing the trajectory, final reward, turns, final env score,
            and original task index.
            e.g., {'trajectory': [...], 'reward': r, 'turns': N, 'env_score': s, 'task_idx': idx}
        """
        trajectory = []
        final_reward = 0.0 # Accumulate reward? Or just final step reward?
        final_env_score = 0.0 # Capture the score from the environment info dict
        done = False
        turns = 0
        current_input_ids = None # Initialize here

        try:
            # 1. Reset environment and get initial observation
            # print(f"[Agent._run_single_rollout][{task_idx}] Resetting environment...") # Debug
            self.client.reset(task_idx) # Use the provided task index
            initial_obs_text = self.client.observe()
            # print(f"[Agent._run_single_rollout][{task_idx}] Initial Obs: {initial_obs_text[:100]}...") # Debug
            
            if not initial_obs_text:
                 print(f"[Agent._run_single_rollout][{task_idx}] Warning: Received empty initial observation. Using initial prompt from batch.")
                 initial_prompt_text = self.tokenizer.decode(initial_prompt_ids[0], skip_special_tokens=True)
                 trajectory.append({"from": "human", "value": initial_prompt_text})
                 current_input_ids = initial_prompt_ids
            else:
                trajectory.append({"from": "human", "value": initial_obs_text})
                current_input_ids = self.tokenizer(initial_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']
            
            # --- Interaction Loop --- 
            for t in range(self.config.max_turns):
                turns = t + 1
                if current_input_ids is None: # Should not happen after initialization
                    print(f"[Agent._run_single_rollout][{task_idx}] Error: current_input_ids is None before generation.")
                    break 
                    
                if current_input_ids.shape[1] > self.config.max_prompt_length:
                    current_input_ids = current_input_ids[:, -self.config.max_prompt_length:]
                    print(f"[Agent._run_single_rollout][{task_idx}] Warning: Truncating input {current_input_ids.shape} > {self.config.max_prompt_length}.")

                # 2. Prepare input DataProto for generation
                current_attention_mask = self.tensor_fn.create_attention_mask(current_input_ids)
                current_position_ids = self.tensor_fn.create_position_ids(current_attention_mask)
                gen_input_proto = DataProto.from_dict({
                    'input_ids': current_input_ids.to(self.actor_rollout_wg.device), # Ensure device match
                    'attention_mask': current_attention_mask.to(self.actor_rollout_wg.device),
                    'position_ids': current_position_ids.to(self.actor_rollout_wg.device)
                })
                
                # 3. Generate response using actor_rollout_wg
                generation_config = GenerationConfig(
                    max_new_tokens=self.config.max_response_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=1.0, 
                    do_sample=True   
                )
                gen_output_proto = self.actor_rollout_wg.generate_sequences(gen_input_proto, generation_config=generation_config)
                response_ids = gen_output_proto.batch['response_ids'] 
                response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
                trajectory.append({"from": "gpt", "value": response_text})

                # 4. Postprocess response to get action
                action_types, action_contents = self.postprocess_predictions([response_text])
                action_text = action_contents[0] 
                
                # 5. Step the environment
                if action_text is None: action_text = "" # Ensure action is not None
                # print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Action: {action_text[:100]}...") # Debug
                next_obs_text, reward, done, info = self.client.step(action_text)
                # print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Reward: {reward}, Done: {done}, Info: {info}") # Debug
                
                final_reward = reward # Store the reward from the last step
                final_env_score = info.get('score', 0.0) # Get score from info dict

                if not done:
                    trajectory.append({"from": "human", "value": next_obs_text})
                    next_obs_ids = self.tokenizer(next_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']
                    # Ensure response_ids has the same device as current_input_ids before cat
                    current_input_ids = torch.cat([current_input_ids, response_ids.to(current_input_ids.device), next_obs_ids.to(current_input_ids.device)], dim=1)
                else:
                    break 

            # --- End of Loop --- 
            # print(f"[Agent._run_single_rollout][{task_idx}] Episode finished. Turns: {turns}, Reward: {final_reward}, Env Score: {final_env_score}") # Debug
            
        except Exception as e:
            print(f"[Agent._run_single_rollout][{task_idx}] Error during rollout: {e}")
            print(traceback.format_exc())
            final_reward = 0.0 
            final_env_score = 0.0 # Score is 0 if error occurred
            done = True

        return {
            'trajectory': trajectory, 
            'reward': final_reward, 
            'turns': turns, 
            'env_score': final_env_score, # Include env score
            'task_idx': task_idx
            }

    def run_llm_loop(self, gen_batch: DataProto) -> DataProto:
        """
        Run the LLM interaction loop for a batch of initial prompts using a ThreadPoolExecutor.
        Replaces the logic that previously used RolloutController.
        """
        initial_prompts_ids = gen_batch.batch['input_ids']
        batch_size = initial_prompts_ids.shape[0]
        print(f"[Agent.run_llm_loop] Starting parallel rollout for batch size: {batch_size}")

        # --- Extract Task Indices --- 
        # Use original indices from the batch if available
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
        rollout_results_list = [None] * batch_size # Preallocate list to store results in order

        for i in range(batch_size):
            task_idx = task_idxs[i]
            initial_prompt = initial_prompts_ids[i:i+1] # Keep batch dim
            future = self.executor.submit(self._run_single_rollout, initial_prompt, task_idx)
            futures[future] = i # Store original index

        for future in as_completed(futures):
            original_index = futures[future]
            try:
                result_dict = future.result()
                rollout_results_list[original_index] = result_dict
            except Exception as e:
                print(f"[Agent.run_llm_loop] Error collecting result for index {original_index}: {e}")
                # Store a placeholder or error indicator if needed
                rollout_results_list[original_index] = {'trajectory': [], 'reward': 0.0, 'turns': 0, 'env_score': 0.0, 'task_idx': task_idxs[original_index], 'error': str(e)}

        print(f"[Agent.run_llm_loop] Collected results from {len(futures)} rollouts.")
        
        # Filter out potential None entries if some tasks failed critically before returning dict
        valid_results = [res for res in rollout_results_list if res is not None]
        
        if not valid_results:
            print("[Agent.run_llm_loop] Error: No valid rollout results collected.")
            return DataProto.from_dict({}) # Return empty DataProto
            
        # --- Format Results into DataProto --- 
        # Reuse the conversion logic, passing the collected trajectories and rewards
        processed_data = self._convert_rollout_results_to_dataproto(valid_results, gen_batch)
        
        print(f"[Agent.run_llm_loop] Finished processing rollout results.")
        return processed_data

    def _convert_rollout_results_to_dataproto(self, results: List[Dict], original_batch: DataProto) -> DataProto:
        """
        Convert the list of dictionaries (each containing trajectory, reward, env_score)
        from the internal rollout loop into a DataProto suitable for PPO training.
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_info_mask = [] 
        batch_rewards = []
        batch_meta_info = defaultdict(list)

        # Need to map results back to original batch indices if results list is filtered/shorter
        # Use 'task_idx' from result dict and find corresponding index in original_batch.meta_info['idx']
        original_indices_map = {idx_val: i for i, idx_val in enumerate(original_batch.meta_info.get('idx', list(range(original_batch.batch['input_ids'].shape[0]))).tolist())}

        print(f"[Agent._convert_rollout] Formatting {len(results)} trajectories.")
        for result_dict in results:
            trajectory = result_dict.get('trajectory', [])
            reward = result_dict.get('reward', 0.0)
            turns = result_dict.get('turns', 0)
            env_score = result_dict.get('env_score', 0.0)
            task_idx = result_dict.get('task_idx', -1)
            
            # Find corresponding index in the original batch
            original_batch_idx = original_indices_map.get(task_idx, -1)
            if original_batch_idx == -1:
                print(f"[Agent._convert_rollout] Warning: Could not map task_idx {task_idx} back to original batch index. Skipping.")
                continue
                
            # --- Concatenate Conversation --- 
            conversation_ids = []
            info_mask_parts = []
            valid_actions = 0
            tool_uses = 0 # Need logic to determine tool use if required

            if not trajectory:
                print(f"[Agent._convert_rollout] Warning: Empty trajectory for task_idx {task_idx}. Using initial prompt only.")
                # Use initial prompt from original batch
                initial_prompt_ids = original_batch.batch['input_ids'][original_batch_idx:original_batch_idx+1]
                conversation_ids.append(initial_prompt_ids)
                info_mask_parts.append(torch.ones_like(initial_prompt_ids))
            else:
                for turn_idx, msg in enumerate(trajectory):
                    msg_text = msg.get("value", "")
                    msg_from = msg.get("from", "")

                    if not msg_text:
                        continue
                        
                    msg_ids = self.tokenizer(msg_text, add_special_tokens=False, return_tensors='pt')['input_ids']
                    conversation_ids.append(msg_ids)
                    
                    # Simplified info mask: Keep human prompts and gpt responses, mask others?
                    # Or align with PPO: Keep everything that influences value function.
                    # Let's keep prompt and response by default.
                    if msg_from == "gpt":
                        info_mask_parts.append(torch.ones_like(msg_ids)) 
                        valid_actions +=1 
                        # Add tool use detection logic here if needed based on response content
                        # actions, _ = self.postprocess_predictions([msg_text])
                        # if actions[0] == 'action': tool_uses += 1
                    else: # human, system, env
                         info_mask_parts.append(torch.ones_like(msg_ids)) # Also keep observations/prompts for value? Check PPO logic.
            
            # Concatenate, Pad, Truncate (same logic as before)
            full_input_ids = torch.cat(conversation_ids, dim=1)
            full_info_mask = torch.cat(info_mask_parts, dim=1)
            seq_len = full_input_ids.shape[1]
            target_len = self.config.max_prompt_length 
            if seq_len > target_len:
                full_input_ids = full_input_ids[:, -target_len:]
                full_info_mask = full_info_mask[:, -target_len:]
            elif seq_len < target_len:
                padding_len = target_len - seq_len
                pad_tensor = torch.full((1, padding_len), self.tokenizer.pad_token_id, dtype=torch.long, device=full_input_ids.device)
                full_input_ids = torch.cat([pad_tensor, full_input_ids], dim=1) # Pad left
                info_pad = torch.zeros_like(pad_tensor) # Padding is masked
                full_info_mask = torch.cat([info_pad, full_info_mask], dim=1)
            
            full_attention_mask = self.tensor_fn.create_attention_mask(full_input_ids)
            full_position_ids = self.tensor_fn.create_position_ids(full_attention_mask)

            batch_input_ids.append(full_input_ids)
            batch_attention_mask.append(full_attention_mask)
            batch_position_ids.append(full_position_ids)
            batch_info_mask.append(full_info_mask)
            batch_rewards.append(reward)
            
            # Add metadata
            batch_meta_info["task_idx"].append(task_idx)
            batch_meta_info["turns_stats"].append(turns)
            batch_meta_info["valid_action_stats"].append(valid_actions)
            batch_meta_info["reward"].append(reward)
            batch_meta_info["env_score"].append(env_score)
            
            # --- >>> Add reward_model info <<< ---
            if 'reward_model' in original_batch.meta_info and isinstance(original_batch.meta_info['reward_model'], list) and len(original_batch.meta_info['reward_model']) > original_batch_idx:
                # Assuming reward_model was a list in the batched meta_info
                batch_meta_info["reward_model"].append(original_batch.meta_info['reward_model'][original_batch_idx])
            elif 'reward_model' in original_batch.meta_info and isinstance(original_batch.meta_info['reward_model'], dict):
                 # If reward_model was somehow passed as a single dict for the whole batch (less likely)
                 if original_batch_idx == 0: # Add only once
                     batch_meta_info["reward_model"] = original_batch.meta_info['reward_model']
            # else: reward_model info not found or in unexpected format
                
            # Copy other relevant meta info from original batch
            for key, value in original_batch.meta_info.items():
                # Avoid duplicating keys already handled (idx, reward, reward_model)
                if key not in ['idx', 'reward', 'reward_model']:
                    if isinstance(value, list) and len(value) > original_batch_idx:
                         batch_meta_info[key].append(value[original_batch_idx])
                    elif not isinstance(value, list): # Keep non-list meta info (add only once)
                        if original_batch_idx == 0: # Add only once
                            batch_meta_info[key] = value 

        # --- Stack Tensors --- (same logic as before)
        if not batch_input_ids: 
             print("[Agent._convert_rollout] No valid trajectories formatted. Returning empty DataProto.")
             return DataProto.from_dict({}) 
        final_batch = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "position_ids": torch.cat(batch_position_ids, dim=0),
            "info_mask": torch.cat(batch_info_mask, dim=0), 
        }
        data_proto = DataProto.from_dict(final_batch)
        for key, value in batch_meta_info.items():
            try:
                data_proto.meta_info[key] = torch.tensor(value)
            except (ValueError, TypeError):
                data_proto.meta_info[key] = value 
        data_proto.meta_info["rewards"] = torch.tensor(batch_rewards, dtype=torch.float32)
        
        # Add specific tensors explicitly if needed by trainer (rewards already handled)
        if "env_score" in batch_meta_info:
             try:
                 data_proto.meta_info["env_scores"] = torch.tensor(batch_meta_info["env_score"], dtype=torch.float32)
             except (ValueError, TypeError):
                 print("[Agent._convert_rollout] Could not convert env_scores to tensor, keeping as list.")
                 data_proto.meta_info["env_scores"] = batch_meta_info["env_score"]
                 
        print(f"[Agent._convert_rollout] Final batch shapes: input_ids={final_batch['input_ids'].shape}")
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