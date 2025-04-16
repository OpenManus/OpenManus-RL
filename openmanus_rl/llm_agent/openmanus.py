import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from transformers import GenerationConfig

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
    rollout_strategy: str = "StandardReAct"
    storage_backend: str = "mongodb" # Or None if not saving via controller
    max_workers: int = 10

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
        actor_rollout_wg,
        config: AgentConfig,
        tool_manager,
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

        # Initialize rollout controller to connect to external AgentGym service
        self._init_rollout_controller()

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _init_rollout_controller(self):
        """
        Initialize the rollout controller to connect to the external AgentGym service.
        Dynamically loads the correct Task based on self.config.env_name.
        """
        import importlib
        from agentenv.rollout.rollout_controller import RolloutController
        from agentenv.rollout.rollout_strategy import StandardReActStrategy, ToTStrategy, MCTSStrategy
        from agentenv.rollout.rollout_db import MongoDBTrajectoryStorage, FileTrajectoryStorage
        
        # Mapping from env_name (lowercase) to Task class name
        # Ensure these names match the classes imported in agentenv.envs.__init__.py
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
        print(f"Initializing RolloutController for env: {self.config.env_name} using Task: {task_class_name}")
        print(f"Connecting to AgentGym server at: {self.config.env_server_base}:{self.config.env_port}")

        # Dynamically import the Task class
        try:
            envs_module = importlib.import_module("agentenv.envs")
            TaskClass = getattr(envs_module, task_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import Task class {task_class_name} from agentenv.envs: {e}")

        # Create environment task connected to the external server
        # client_args might need adjustment for specific environments, but start with common ones.
        # Especially lmrlgym (maze/wordle) might need the path in env_server_base adjusted (handled in train_ppo.sh now)
        client_args={
            "env_server_base": f"{self.config.env_server_base}:{self.config.env_port}",
            "data_len": self.config.env_data_len, # Ensure the server is initialized with enough data
            "timeout": 300, 
        }
        task = TaskClass(
            client_args=client_args,
            n_clients=1 # Assuming one client connection per agent instance for now
        )

        # Select rollout strategy based on config
        if self.config.rollout_strategy == "StandardReAct":
            strategy = StandardReActStrategy()
        elif self.config.rollout_strategy == "ToT":
            strategy = ToTStrategy(num_branches=3, depth=2)
        elif self.config.rollout_strategy == "MCTS":
            strategy = MCTSStrategy(num_simulations=50, exploration_weight=1.0)
        else:
            raise ValueError(f"Unknown strategy: {self.config.rollout_strategy}")

        # Configure storage backend (Optional, might be handled by trainer elsewhere)
        storage = None # Disable controller's storage by default
        if self.config.storage_backend == "mongodb":
            storage = MongoDBTrajectoryStorage()
        elif self.config.storage_backend == "file":
            storage = FileTrajectoryStorage()
        # else: storage remains None

        # Initialize controller
        self.rollout_controller = RolloutController(
            agent=self, # Pass self as the agent
            tasks=[task],
            strategy=strategy,
            storage=storage, # Pass configured storage
            max_workers=self.config.max_workers
        )
        print("RolloutController initialized successfully.")

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

    def run_llm_loop(self, gen_batch: DataProto) -> DataProto:
        """
        Run the LLM generation loop using the configured RolloutController.

        This method orchestrates the interaction with the AgentGym environment 
        via the RolloutController, collects the trajectories, and formats them 
        into a DataProto suitable for PPO training.

        Args:
            gen_batch: Batch containing initial prompts and metadata.
                       Expects gen_batch.meta_info['idx'] to contain task indices.

        Returns:
            DataProto containing the full rollout trajectories and associated info.
        """
        print(f"[Agent.run_llm_loop] Starting rollout for batch size: {gen_batch.batch['input_ids'].shape[0]}")
        
        # --- 1. Configure Generation --- 
        # Create a basic GenerationConfig. Specific strategies might override this.
        # Ensure EOS token is correctly set for the model.
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_response_length, # Use max_new_tokens
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            # Add other relevant generation parameters if needed (e.g., temperature, top_k)
            temperature=1.0, # Example: Add temperature if needed by strategy
            do_sample=True   # Example: Enable sampling if needed
        )

        # --- 2. Extract Task Indices --- 
        # Assume indices are passed via gen_batch meta_info
        # If not present, fallback or raise error
        if 'idx' not in gen_batch.meta_info:
            # Fallback: Use range if indices not provided (might not be correct)
            print("[Agent.run_llm_loop] WARNING: 'idx' not found in gen_batch.meta_info. Using range(batch_size). This might be incorrect.")
            task_idxs = list(range(gen_batch.batch['input_ids'].shape[0]))
        else:
            task_idxs = gen_batch.meta_info['idx']
            if isinstance(task_idxs, torch.Tensor):
                task_idxs = task_idxs.tolist()
            print(f"[Agent.run_llm_loop] Using task indices: {task_idxs}")
            
        # --- 3. Execute Rollout --- 
        # Call the controller's rollout method
        # Pass necessary arguments like indices, generation config, max turns.
        try:
            rollout_results: List[ExperienceOutput] = self.rollout_controller.rollout(
                generation_config=generation_config,
                max_rounds=self.config.max_turns,
                idxs=task_idxs, 
                save_to_storage=False, # Saving handled by trainer/elsewhere if needed
                parallel=True, # Use parallel execution
                batch_size=self.config.max_workers, # Adjust batch size based on workers
                metadata={
                    # "model_name": self.actor_rollout_wg.model_name, # Get model name if available
                    "validation": self.is_validation,
                    "prompt_source": "gen_batch" # Indicate source of initial prompt
                }
            )
            print(f"[Agent.run_llm_loop] Rollout completed. Received {len(rollout_results)} trajectories.")
        except Exception as e:
            print(f"[Agent.run_llm_loop] Error during rollout: {e}")
            # Handle error appropriately, maybe return an empty DataProto or re-raise
            return DataProto.from_dict({}) # Return empty
            
        # Check if results match expected size
        if len(rollout_results) != len(task_idxs):
            print(f"[Agent.run_llm_loop] WARNING: Rollout returned {len(rollout_results)} results, but expected {len(task_idxs)}.")
            # Decide how to handle mismatch - pad, filter, or error

        # --- 4. Process Results --- 
        # Convert the list of ExperienceOutput trajectories into a DataProto
        processed_data = self._convert_rollout_results_to_dataproto(rollout_results, gen_batch)
        
        print(f"[Agent.run_llm_loop] Finished processing rollout results.")
        return processed_data

    def _convert_rollout_results_to_dataproto(self, results: List[ExperienceOutput], original_batch: DataProto) -> DataProto:
        """
        Convert the list of ExperienceOutput from rollout into a DataProto 
        suitable for PPO training.

        This involves concatenating the conversation history (prompt, response, obs)
        into sequences and creating corresponding masks and metadata.

        Args:
            results: List of ExperienceOutput objects from RolloutController.rollout.
            original_batch: The initial batch passed to run_llm_loop, used for reference.

        Returns:
            A DataProto object containing the processed trajectories.
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_info_mask = [] # Mask for value function (masks out observations/prompts)
        batch_rewards = []
        batch_meta_info = defaultdict(list)

        # Assuming original_batch contains initial prompts in 'input_ids'
        initial_prompts_ids = original_batch.batch['input_ids']
        initial_prompts_attn = original_batch.batch['attention_mask']
        
        # Map result index back to original batch index if necessary (assuming order is preserved for now)
        # If rollout results might be out of order or filtered, need a mapping based on task_idx
        
        print(f"[Agent._convert_rollout] Processing {len(results)} trajectories.")
        for i, result in enumerate(results):
            # --- Concatenate Conversation --- 
            # Start with the initial prompt from the original batch
            # Note: RolloutController might have its own way of getting the initial prompt,
            # ensure consistency. Here we assume original_batch has the correct initial prompt.
            # We need the original index `orig_idx` if results are not in order. Assume order for now.
            orig_idx = i # Assuming results are ordered same as task_idxs
            current_input_ids = initial_prompts_ids[orig_idx:orig_idx+1] # Get the specific prompt
            current_info_mask_parts = [torch.ones_like(current_input_ids)] # Prompt is part of value input
            
            conversation_ids = [current_input_ids]
            
            turns = 0
            valid_actions = 0
            tool_uses = 0

            if result and result.conversation: # Check if result and conversation exist
                for turn_idx, msg in enumerate(result.conversation):
                    msg_text = msg.get("value", "")
                    msg_from = msg.get("from", "")
                    is_loss_turn = msg.get("loss", False) # Used by some strategies

                    if not msg_text: # Skip empty messages
                        continue
                        
                    # Tokenize the message
                    # Use add_special_tokens=False for intermediate parts
                    msg_ids = self.tokenizer(msg_text, add_special_tokens=False, return_tensors='pt')['input_ids']
                    
                    # Add to the sequence
                    conversation_ids.append(msg_ids)
                    
                    # Update info mask: mask out observations (from env/user), keep prompts/responses
                    if msg_from == "user" or msg_from == "system" or msg_from == "env": # Typically observations or new instructions
                        current_info_mask_parts.append(torch.zeros_like(msg_ids)) # Mask out
                    else: # Agent's response/action
                        current_info_mask_parts.append(torch.ones_like(msg_ids)) # Keep
                        if msg_from == "gpt": # Count agent turns
                            valid_actions += 1
                            if is_loss_turn: # Check if it was a tool use based on loss flag
                                tool_uses += 1
                turns = len(result.conversation) // 2 # Approximate turns
            else:
                 print(f"[Agent._convert_rollout] Warning: Empty or invalid trajectory for index {i}")
                 # Handle empty trajectory: skip or add placeholder? Add placeholder for now
                 pass # Fallback to just the initial prompt below

            # Concatenate all parts for this trajectory
            full_input_ids = torch.cat(conversation_ids, dim=1)
            full_info_mask = torch.cat(current_info_mask_parts, dim=1)

            # --- Pad and Truncate --- 
            # Pad to max_prompt_length (or another suitable length)
            # Truncate from the left if too long
            seq_len = full_input_ids.shape[1]
            target_len = self.config.max_prompt_length # Use max_prompt_length for combined sequence
            
            if seq_len > target_len:
                full_input_ids = full_input_ids[:, -target_len:]
                full_info_mask = full_info_mask[:, -target_len:]
            elif seq_len < target_len:
                padding_len = target_len - seq_len
                pad_tensor = torch.full((1, padding_len), self.tokenizer.pad_token_id, dtype=torch.long, device=full_input_ids.device)
                full_input_ids = torch.cat([pad_tensor, full_input_ids], dim=1) # Pad left
                info_pad = torch.zeros_like(pad_tensor) # Padding is masked out in info mask
                full_info_mask = torch.cat([info_pad, full_info_mask], dim=1)
            
            # --- Create Attention Mask and Position IDs --- 
            full_attention_mask = self.tensor_fn.create_attention_mask(full_input_ids)
            full_position_ids = self.tensor_fn.create_position_ids(full_attention_mask)

            # --- Store Processed Data --- 
            batch_input_ids.append(full_input_ids)
            batch_attention_mask.append(full_attention_mask)
            batch_position_ids.append(full_position_ids)
            batch_info_mask.append(full_info_mask)
            batch_rewards.append(result.reward if result else 0.0) # Add reward
            
            # Add metadata
            batch_meta_info["turns_stats"].append(turns)
            batch_meta_info["valid_action_stats"].append(valid_actions)
            batch_meta_info["tool_use_stats"].append(tool_uses)
            batch_meta_info["reward"].append(result.reward if result else 0.0)
            # Copy relevant meta info from original batch if needed
            for key, value in original_batch.meta_info.items():
                if key != 'idx' and isinstance(value, list) and len(value) > orig_idx:
                     batch_meta_info[key].append(value[orig_idx])
                elif key != 'idx': # Keep non-list meta info as is for all
                     if i == 0: # Add only once
                         batch_meta_info[key] = value 
                         
        # --- Stack Tensors --- 
        if not batch_input_ids: # Handle case with no valid results
             print("[Agent._convert_rollout] No valid trajectories processed. Returning empty DataProto.")
             return DataProto.from_dict({}) 
             
        final_batch = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "position_ids": torch.cat(batch_position_ids, dim=0),
            "info_mask": torch.cat(batch_info_mask, dim=0), 
            # Note: 'prompts' and 'responses' fields from old format are not directly created here.
            # The full sequence is in 'input_ids'. The trainer needs to handle this.
        }

        # --- Create Final DataProto --- 
        data_proto = DataProto.from_dict(final_batch)
        # Convert lists in meta_info to tensors if appropriate, keep as lists otherwise
        for key, value in batch_meta_info.items():
            try:
                # Attempt to convert to tensor if items are numerical
                data_proto.meta_info[key] = torch.tensor(value)
            except (ValueError, TypeError):
                # Keep as list if conversion fails
                data_proto.meta_info[key] = value 
                
        # Add rewards tensor explicitly if needed by trainer
        data_proto.meta_info["rewards"] = torch.tensor(batch_rewards, dtype=torch.float32)
        
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