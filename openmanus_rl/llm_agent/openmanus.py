import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto

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
    
    # Environment configuration
    env_name: str = "webshop"
    env_port: int = 36001
    env_server_base: str = "http://127.0.0.1"
    rollout_strategy: str = "StandardReAct"
    storage_backend: str = "mongodb"
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
            config: Agent configuration
            tool_manager: Manager for tool operations
            is_validation: Whether in validation mode
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.tool_manager = tool_manager
        self.is_validation = is_validation

        # Initialize rollout controller
        self._init_rollout_controller()

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _init_rollout_controller(self):
        """
        Initialize the rollout controller with specified strategy and storage.
        
        This method:
        1. Creates environment task
        2. Selects rollout strategy
        3. Configures storage backend
        4. Initializes the controller
        """
        from agentenv.rollout.rollout_controller import RolloutController
        from agentenv.rollout.rollout_strategy import StandardReActStrategy, ToTStrategy, MCTSStrategy
        from agentenv.rollout.rollout_db import MongoDBTrajectoryStorage, FileTrajectoryStorage
        from agentenv.envs import WebshopTask

        # Create environment task
        task = WebshopTask(
            client_args={
                "env_server_base": f"{self.config.env_server_base}:{self.config.env_port}",
                "data_len": 200,
                "timeout": 300,
            },
            n_clients=1
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

        # Configure storage backend
        if self.config.storage_backend == "mongodb":
            storage = MongoDBTrajectoryStorage()
        elif self.config.storage_backend == "file":
            storage = FileTrajectoryStorage()
        else:
            raise ValueError(f"Unknown storage backend: {self.config.storage_backend}")

        # Initialize controller
        self.rollout_controller = RolloutController(
            agent=self,
            tasks=[task],
            strategy=strategy,
            storage=storage,
            max_workers=self.config.max_workers
        )

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

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Run the LLM generation loop using rollout controller.
        
        This method:
        1. Configures generation parameters
        2. Executes rollout using controller
        3. Processes and returns results
        
        Args:
            gen_batch: Batch of generation inputs
            initial_input_ids: Initial input token IDs
            
        Returns:
            Tuple containing final output and metadata
        """
        # Configure generation parameters
        generation_config = GenerationConfig(
            max_length=self.config.max_response_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Execute rollout using controller
        results = self.rollout_controller.rollout(
            generation_config=generation_config,
            max_rounds=self.config.max_turns,
            idxs=list(range(gen_batch.batch['input_ids'].shape[0])),
            save_to_storage=True,
            parallel=True,
            batch_size=1,
            metadata={
                "model_name": self.actor_rollout_wg.model_name,
                "validation": self.is_validation
            }
        )

        # Process results
        active_mask = torch.ones(len(results), dtype=torch.bool)
        turns_stats = torch.ones(len(results), dtype=torch.int)
        valid_action_stats = torch.zeros(len(results), dtype=torch.int)
        tool_use_stats = torch.zeros(len(results), dtype=torch.int)
        
        # Collect statistics
        for i, result in enumerate(results):
            turns_stats[i] = len(result.conversation) // 2
            valid_action_stats[i] = sum(1 for msg in result.conversation if msg["from"] == "gpt")
            tool_use_stats[i] = sum(1 for msg in result.conversation if msg.get("loss") is True)

        # Build final output
        meta_info = {
            "turns_stats": turns_stats.tolist(),
            "active_mask": active_mask.tolist(),
            "valid_action_stats": valid_action_stats.tolist(),
            "tool_use_stats": tool_use_stats.tolist()
        }

        # Convert results to DataProto format
        final_output = self._compose_final_output(
            {"input_ids": initial_input_ids},
            {"responses": self._batch_tokenize([r.text for r in results])},
            meta_info
        )

        return final_output

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> DataProto:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, execute_tools=True) -> Tuple[List[str], List[bool], List[bool], List[bool]]:
        """
        Execute predictions using rollout controller.
        
        Args:
            predictions: List of action predictions
            pad_token: Padding token
            active_mask: Mask for active sequences
            execute_tools: Whether to execute tools
            
        Returns:
            Tuple of (next observations, done flags, valid action flags, tool use flags)
        """
        if not execute_tools:
            return super().execute_predictions(predictions, pad_token, active_mask, execute_tools)

        # Execute tool calls using rollout controller
        results = self.rollout_controller._rollout_one(
            task=self.rollout_controller.tasks[0],
            idx=0,
            generation_config=None,
            max_rounds=1,
            save_to_storage=False,
            metadata=None
        )

        # Process results
        next_obs = []
        dones = []
        valid_action = []
        is_tool_use = []

        for result in results:
            next_obs.append(result.conversation[-1]["value"] if result.conversation else "")
            dones.append(True)
            valid_action.append(True)
            is_tool_use.append(True)

        return next_obs, dones, valid_action, is_tool_use

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