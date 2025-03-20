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
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    react_format: bool = True  # Whether to use ReAct format

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
        tool_manager,  # Tool manager instead of search functionality
        is_validation: bool = False,
    ):
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
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        tool_use_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
                
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # Generate next responses for active sequences
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute tool calls and process observations
            next_obs, dones, valid_action, is_tool_use = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            tool_use_stats += torch.tensor(is_tool_use, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # Final LLM rollout for any remaining active sequences
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute final responses, but don't process tools
            _, dones, valid_action, is_tool_use = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, execute_tools=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            tool_use_stats += torch.tensor(is_tool_use, dtype=torch.int)
            
            # Update metadata
            meta_info['turns_stats'] = turns_stats.tolist()
            meta_info['active_mask'] = active_mask.tolist()
            meta_info['valid_action_stats'] = valid_action_stats.tolist()
            meta_info['tool_use_stats'] = tool_use_stats.tolist()

            # Update right side with final responses
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

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
        Execute tool calls based on predictions.
        
        Args:
            predictions: List of action predictions
            pad_token: Token to use for padding
            active_mask: Boolean mask indicating active sequences
            execute_tools: Whether to actually execute tool calls or just process them
            
        Returns:
            Tuple of (next observations, done flags, valid action flags, tool use flags)
        """
        actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_tool_use = [], [], [], []
        
        # Process tools in batch if possible
        tool_results = {}
        if execute_tools:
            for i, (action, content) in enumerate(zip(actions, contents)):
                if not active_mask[i] or action != 'action':
                    continue
                    
                try:
                    # Parse tool call (tool_name and args)
                    tool_match = re.match(r'^(\w+)\((.*)\)$', content.strip())
                    if tool_match:
                        tool_name = tool_match.group(1)
                        args_str = tool_match.group(2)
                        
                        # Parse args - simple implementation
                        args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                        
                        # Call the tool
                        if tool_name in self.tool_manager.get_tool_names():
                            result = self.tool_manager.call_tool(tool_name, *args)
                            tool_results[i] = result
                        else:
                            tool_results[i] = f"Error: Unknown tool '{tool_name}'"
                    else:
                        tool_results[i] = "Error: Invalid tool call format. Use: tool_name(arg1, arg2, ...)"
                except Exception as e:
                    tool_results[i] = f"Error: {str(e)}"
        
        # Process all predictions
        for i, (action, content) in enumerate(zip(actions, contents)):
            if not active_mask[i]:
                next_obs.append('')
                dones.append(True)
                valid_action.append(False)
                is_tool_use.append(False)
                continue
                
            if action == 'response':
                # Final response - sequence is done
                next_obs.append('')
                dones.append(True)
                valid_action.append(True)
                is_tool_use.append(False)
            elif action == 'action':
                if execute_tools:
                    if i in tool_results:
                        result = tool_results[i]
                        next_obs.append(f'\n\n<observation>{result}</observation>\n\n')
                        dones.append(False)  # Continue the sequence
                        valid_action.append(True)
                        is_tool_use.append(True)
                    else:
                        next_obs.append('\nError processing the tool call. Please try again.\n')
                        dones.append(False)
                        valid_action.append(False)
                        is_tool_use.append(True)
                else:
                    # Just acknowledge the tool call but don't execute it
                    next_obs.append('\nTool call received but not executed in final turn.\n')
                    dones.append(False)
                    valid_action.append(False)
                    is_tool_use.append(True)
            else:
                # Invalid action format
                next_obs.append(f'\nMy previous action is invalid. \
If I want to use a tool, I should put the tool call between <action> and </action> tags. \
If I want to give the final response, I should put it between <response> and </response> tags. Let me try again.\n')
                dones.append(False)
                valid_action.append(False)
                is_tool_use.append(False)
                
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