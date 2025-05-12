import torch
import re
from collections import defaultdict, deque
import heapq
import os
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from openmanus_rl.llm_agent.openmanus import OpenManusAgent, AgentConfig
from verl import DataProto
from transformers import GenerationConfig


@dataclass
class ToTConfig(AgentConfig):
    """
    Configuration for ToT (Tree of Thought) exploration and selection.
    
    Additional ToT-specific attributes:
        tot_beam_width: Number of parallel paths to explore at each step
        tot_exploration_factor: How many variations to generate at each exploration step
        tot_max_branches: Maximum total explored branches per episode
        tot_value_guidance: Whether to use critic for value estimation
        tot_temperature: Temperature for exploration vs exploitation
        tot_search_strategy: "BFS" (Breadth-First Search) or "DFS" (Depth-First Search)
    """
    # All ToT specific settings with default values
    tot_beam_width: int = 3  # Number of paths to keep at each step
    tot_exploration_factor: int = 2  # How many variations to generate per node
    tot_max_branches: int = 15  # Maximum total branches explored per episode
    tot_value_guidance: bool = False  # Whether to use critic for value estimation
    tot_temperature: float = 1.0  # Temperature for exploration vs exploitation
    tot_search_strategy: str = "BFS"  # BFS or DFS
    tot_value_threshold: float = 0.0  # Minimum value to consider a branch for further exploration
    tot_reward_cutoff: float = 0.8  # Early terminate exploration if we hit this reward


@dataclass
class TrajectoryNode:
    """
    Represents a node in the tree of thought, containing a trajectory up to this point.
    """
    trajectory: List[Dict[str, Any]]  # The interaction history 
    reward: float = 0.0  # Accumulated reward
    value: float = 0.0  # Value estimate (if available)
    state: Any = None  # Current environment state (if needed)
    parent: Optional['TrajectoryNode'] = None
    children: List['TrajectoryNode'] = field(default_factory=list)
    depth: int = 0  # Depth in the tree
    exploration_score: float = 0.0  # For UCB or similar exploration strategies
    terminal: bool = False  # Whether this node is terminal
    
    def __lt__(self, other):
        # For priority queue
        if self.terminal and not other.terminal:
            return False
        elif not self.terminal and other.terminal:
            return True
        
        # First by reward
        if self.reward != other.reward:
            return self.reward > other.reward  # Higher reward is better
        
        # Then by value
        if self.value != other.value:
            return self.value > other.value  # Higher value is better
            
        # Then by depth, preferring shorter paths
        return self.depth < other.depth


class OpenManusToTAgent(OpenManusAgent):
    """
    OpenManus Agent that uses Tree of Thought for exploration and selection.
    Extends the standard OpenManusAgent with ToT capabilities.
    """
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: ToTConfig,
        critic_wg=None,  # Optional critic for value estimation
        is_validation: bool = False,
        logger=None,
    ):
        """
        Initialize ToT Agent with rollout controller integration.
        
        Args:
            tokenizer: Tokenizer for text processing
            actor_rollout_wg: Actor rollout wrapper for generation
            config: ToT configuration
            critic_wg: Optional critic for value estimation
            is_validation: Whether in validation mode
            logger: Logger for tracking and visualization
        """
        super().__init__(
            tokenizer=tokenizer,
            actor_rollout_wg=actor_rollout_wg, 
            config=config,
            is_validation=is_validation,
            logger=logger
        )
        
        self.critic_wg = critic_wg
        self.tot_config = config
        
        # Setup critic for value estimation if enabled
        if self.tot_config.tot_value_guidance and self.critic_wg is None:
            print("[WARNING] Value guidance enabled but no critic provided. Value guidance will be disabled.")
            self.tot_config.tot_value_guidance = False
            
    def _calculate_ucb_score(self, node: TrajectoryNode, parent_visits: int, exploration_weight: float = 1.0) -> float:
        """
        Calculate UCB (Upper Confidence Bound) score for a node.
        Used for balancing exploration vs exploitation.
        
        Args:
            node: TrajectoryNode to calculate score for
            parent_visits: Number of visits to the parent node
            exploration_weight: Weight for the exploration term
            
        Returns:
            UCB score
        """
        # Use reward as exploitation term
        exploitation = node.reward
        
        # Visits are tracked implicitly by node.depth for now
        # This could be enhanced with explicit visit tracking
        node_visits = max(1, node.depth)
        
        # Add exploration bonus based on UCB formula
        exploration = exploration_weight * np.sqrt(2 * np.log(parent_visits) / node_visits)
        
        return exploitation + exploration
        
    def _estimate_value(self, node: TrajectoryNode) -> float:
        """
        Estimate value of a trajectory node using critic if available.
        
        Args:
            node: TrajectoryNode to estimate value for
            
        Returns:
            Estimated value
        """
        if not self.tot_config.tot_value_guidance or self.critic_wg is None:
            return 0.0
            
        try:
            # Construct input for critic
            # We'll need to convert the trajectory to a format the critic can understand
            # This depends on the specific critic implementation
            # For now, let's assume the critic takes a sequence of tokens
            
            # Combine the trajectory into a single string
            trajectory_text = ""
            for item in node.trajectory:
                role = item.get("from", "unknown")
                content = item.get("value", "")
                if role == "human":
                    trajectory_text += f"Human: {content}\n"
                elif role == "gpt":
                    trajectory_text += f"Assistant: {content}\n"
                elif role == "env":
                    trajectory_text += f"Environment: {content}\n"
                    
            # Tokenize the trajectory
            tokens = self.tokenizer(trajectory_text, return_tensors="pt", truncation=True, 
                                    max_length=self.config.max_prompt_length)
                                    
            # Call critic to get value estimate
            # This is a simplified version - actual implementation depends on critic interface
            value_output = self.critic_wg.compute_values(DataProto.from_dict(tokens))
            
            # Extract value from output
            if hasattr(value_output, 'batch') and 'values' in value_output.batch:
                # Take the last value as the trajectory value
                value = value_output.batch['values'][0, -1].item()
                return value
            
            return 0.0
        except Exception as e:
            print(f"Error estimating value: {e}")
            return 0.0
            
    def _generate_explorations(self, node: TrajectoryNode, num_variations: int) -> List[TrajectoryNode]:
        """
        Generate multiple variations from a given node for exploration.
        
        Args:
            node: Source node to explore from
            num_variations: Number of variations to generate
            
        Returns:
            List of new TrajectoryNode objects with variations
        """
        # Check if this is a terminal node
        if node.terminal:
            return []
            
        # Extract the current state of the trajectory
        current_input_ids = None
        
        # Get the client for this task
        task_idx = 0  # Default task index
        
        # Extract task_idx if available in the trajectory
        for item in node.trajectory:
            if "task_idx" in item:
                task_idx = item["task_idx"]
                break
                
        client_index = task_idx % len(self.clients)
        client = self.clients[client_index]
        
        # Recreate the input for continuation
        current_state = ""
        
        for entry in node.trajectory:
            if entry.get("from") == "human":
                current_state += entry.get("value", "")
            elif entry.get("from") == "gpt":
                current_state += entry.get("value", "")
            elif entry.get("from") == "env":
                current_state += entry.get("value", "")
                
        # Tokenize the current state
        if current_state:
            current_input_ids = self.tokenizer(current_state, return_tensors='pt', add_special_tokens=False)['input_ids']
        
        # If we don't have input_ids, we can't continue
        if current_input_ids is None:
            print(f"[ERROR] Cannot generate explorations: current_input_ids is None")
            return []
            
        # Handle input that exceeds max length
        if current_input_ids.shape[1] > self.config.max_prompt_length:
            current_input_ids = current_input_ids[:, -self.config.max_prompt_length:]
                
        # Prepare input tensors
        current_attention_mask = self.tensor_fn.create_attention_mask(current_input_ids)
        current_position_ids = self.tensor_fn.create_position_ids(current_attention_mask)
        
        # Create DataProto for generation
        gen_input_proto = DataProto.from_dict({
            'input_ids': current_input_ids,
            'attention_mask': current_attention_mask,
            'position_ids': current_position_ids
        })
        
        # Pad for tensor parallelism if needed
        world_size = self.actor_rollout_wg.world_size
        original_size = 1  # We know batch size is 1 here
        padded_gen_input_proto = gen_input_proto
        padding_size = 0
        
        if world_size > 1 and original_size % world_size != 0:
            padding_size = world_size - (original_size % world_size)
            padded_batch = {}
            for k, v in gen_input_proto.batch.items():
                pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
                padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
            padded_gen_input_proto = DataProto.from_dict(padded_batch)
            if hasattr(gen_input_proto, 'meta_info'):
                padded_gen_input_proto.meta_info = gen_input_proto.meta_info.copy()
                
        # Prepare generation config - increase temperature for exploration
        tot_temperature = self.tot_config.tot_temperature
        diversity_weight = max(1.0, self.tot_config.tot_temperature * 1.5)  # Increase diversity for explorations
        
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_response_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=tot_temperature * diversity_weight,
            do_sample=True,
            num_return_sequences=num_variations,
            top_p=0.92,  # Slightly higher top_p for more diversity
            top_k=50,    # Keep reasonable top_k
        )
        
        if not hasattr(padded_gen_input_proto, 'meta_info'):
            padded_gen_input_proto.meta_info = {}
        padded_gen_input_proto.meta_info['generation_config'] = generation_config
        
        # Generate variations
        try:
            gen_output_proto = self.actor_rollout_wg.generate_sequences(padded_gen_input_proto)
            response_ids = gen_output_proto.batch['responses']  # Use the correct key for response IDs
            
            if padding_size > 0:
                response_ids = response_ids[:-padding_size]
                
            # Create exploration nodes
            exploration_nodes = []
            
            for i in range(min(num_variations, response_ids.shape[0])):
                # Decode the response
                response_text = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)
                
                # Extract action using postprocessing
                action_types, action_contents = self.postprocess_predictions([response_text])
                action_text = action_contents[0] if action_contents else ""
                
                # Execute action in environment
                if action_text is None:
                    action_text = ""
                
                # Make a copy of node's trajectory
                new_trajectory = node.trajectory.copy()
                
                # Add the agent's action to the trajectory
                new_trajectory.append({"from": "gpt", "value": response_text})
                
                # Execute environment step using client
                try:
                    step_output = client.step(action_text)
                    next_obs_text = step_output.state
                    reward = step_output.reward
                    done = step_output.done
                    
                    # Add environment response to trajectory
                    if not done:
                        new_trajectory.append({"from": "env", "value": next_obs_text})
                    
                    # Create new exploration node
                    new_node = TrajectoryNode(
                        trajectory=new_trajectory,
                        reward=node.reward + reward,  # Accumulate reward
                        parent=node,
                        depth=node.depth + 1,
                        terminal=done,
                    )
                    
                    # If value guidance is enabled, estimate value
                    if self.tot_config.tot_value_guidance:
                        new_node.value = self._estimate_value(new_node)
                    
                    # Only add nodes that meet the value threshold
                    if new_node.value >= self.tot_config.tot_value_threshold:
                        exploration_nodes.append(new_node)
                        node.children.append(new_node)
                        
                except Exception as e:
                    print(f"Error during environment step: {e}")
                    traceback.print_exc()
                    continue
            
            return exploration_nodes
            
        except Exception as e:
            print(f"Error generating explorations: {e}")
            traceback.print_exc()
            return []
            
    def _select_best_trajectory(self, root_node: TrajectoryNode) -> TrajectoryNode:
        """
        Select the best trajectory from the explored tree.
        
        Args:
            root_node: Root node of the exploration tree
            
        Returns:
            Best TrajectoryNode according to reward and other criteria
        """
        # Start with a breadth-first traversal to find all leaf nodes
        queue = deque([root_node])
        leaf_nodes = []
        
        while queue:
            node = queue.popleft()
            
            # If node is terminal or has no children, it's a leaf
            if node.terminal or not node.children:
                leaf_nodes.append(node)
            else:
                # Add children to queue
                queue.extend(node.children)
                
        # If no leaf nodes, return root
        if not leaf_nodes:
            return root_node
            
        # Sort leaf nodes by reward, value, and depth
        sorted_nodes = sorted(leaf_nodes, key=lambda n: (n.reward, n.value, -n.depth), reverse=True)
        
        # Return the best node
        return sorted_nodes[0]
            
    def _run_tot_exploration(self, initial_prompt_ids: torch.Tensor, task_idx: int, client: Any) -> Dict[str, Any]:
        """
        Runs the Tree of Thought exploration for a single environment instance.
        
        Args:
            initial_prompt_ids: Token IDs for the initial prompt
            task_idx: The index for resetting the environment
            client: The specific environment client instance to use

        Returns:
            A dictionary containing the best trajectory, rewards, etc.
        """
        # Reset environment
        reset_info = client.reset(task_idx)
        initial_obs_text = client.observe()
        
        # Create root node
        trajectory = []
        
        # Handle initial observation
        if not initial_obs_text:
            initial_prompt_text = self.tokenizer.decode(initial_prompt_ids[0], skip_special_tokens=True)
            trajectory.append({"from": "human", "value": initial_prompt_text})
        else:
            trajectory.append({"from": "human", "value": initial_obs_text})
            
        root_node = TrajectoryNode(
            trajectory=trajectory,
            reward=0.0,
            depth=0,
        )
        
        # Exploration strategy selection
        if self.tot_config.tot_search_strategy == "BFS":
            return self._run_bfs_exploration(root_node, task_idx, client)
        elif self.tot_config.tot_search_strategy == "DFS":
            return self._run_dfs_exploration(root_node, task_idx, client)
        else:
            print(f"[Warning] Unknown search strategy: {self.tot_config.tot_search_strategy}. Using BFS.")
            return self._run_bfs_exploration(root_node, task_idx, client)
            
    def _run_bfs_exploration(self, root_node: TrajectoryNode, task_idx: int, client: Any) -> Dict[str, Any]:
        """
        Run breadth-first search exploration of the decision tree.
        
        Args:
            root_node: Root node of the exploration tree
            task_idx: The index for resetting the environment
            client: The specific environment client instance
            
        Returns:
            Dictionary with exploration results
        """
        # Queue for BFS
        exploration_queue = deque([root_node])
        
        # Keep track of explored branches
        explored_branches = 0
        max_branches = self.tot_config.tot_max_branches
        
        # Exploration loop
        while exploration_queue and explored_branches < max_branches:
            # Get next node to explore
            current_node = exploration_queue.popleft()
            
            # Generate explorations from this node
            exploration_factor = self.tot_config.tot_exploration_factor
            new_nodes = self._generate_explorations(current_node, exploration_factor)
            explored_branches += len(new_nodes)
            
            # Check for early termination if we found a good reward
            found_good_reward = any(node.reward >= self.tot_config.tot_reward_cutoff for node in new_nodes)
            if found_good_reward:
                print(f"[Info] Found good reward, terminating exploration early after {explored_branches} branches.")
                break
                
            # Add new nodes to queue (up to beam width)
            # Sort by reward and value before adding to maintain beam
            sorted_nodes = sorted(new_nodes, key=lambda n: (n.reward, n.value), reverse=True)
            beam_width = min(len(sorted_nodes), self.tot_config.tot_beam_width)
            exploration_queue.extend(sorted_nodes[:beam_width])
            
        # Select best trajectory
        best_node = self._select_best_trajectory(root_node)
        
        # Compile results
        return {
            'trajectory': best_node.trajectory,
            'step_rewards': [item.get("reward", 0.0) for item in best_node.trajectory if "reward" in item],
            'reward': best_node.reward,
            'env_score': best_node.reward,  # Use cumulative reward as env score for now
            'turns': best_node.depth,
            'valid_actions': len([msg for msg in best_node.trajectory if msg.get("from") == "gpt"]),
            'task_idx': task_idx,
            'done': best_node.terminal,
            'tot_metrics': {
                'explored_branches': explored_branches,
                'max_depth': best_node.depth,
            }
        }
            
    def _run_dfs_exploration(self, root_node: TrajectoryNode, task_idx: int, client: Any) -> Dict[str, Any]:
        """
        Run depth-first search exploration of the decision tree.
        
        Args:
            root_node: Root node of the exploration tree
            task_idx: The index for resetting the environment
            client: The specific environment client instance
            
        Returns:
            Dictionary with exploration results
        """
        # Stack for DFS
        exploration_stack = [root_node]
        
        # Keep track of explored branches
        explored_branches = 0
        max_branches = self.tot_config.tot_max_branches
        
        # Keep track of best node so far
        best_node = root_node
        best_reward = 0.0
        
        # Exploration loop
        while exploration_stack and explored_branches < max_branches:
            # Get next node to explore (priority by UCB score)
            current_node = exploration_stack.pop()
            
            # Update best node if this one is better
            if current_node.reward > best_reward or (current_node.reward == best_reward and current_node.depth < best_node.depth):
                best_node = current_node
                best_reward = current_node.reward
                
            # Check for early termination if we found a good reward
            if current_node.reward >= self.tot_config.tot_reward_cutoff:
                print(f"[Info] Found good reward, terminating exploration early after {explored_branches} branches.")
                break
                
            # Generate explorations from this node
            exploration_factor = self.tot_config.tot_exploration_factor
            new_nodes = self._generate_explorations(current_node, exploration_factor)
            explored_branches += len(new_nodes)
            
            # Calculate UCB scores for prioritization
            for node in new_nodes:
                # Use parent visits as exploration factor
                parent_visits = max(1, len(current_node.children))
                node.exploration_score = self._calculate_ucb_score(node, parent_visits)
                
            # Sort nodes by UCB score
            sorted_nodes = sorted(new_nodes, key=lambda n: n.exploration_score, reverse=True)
            
            # Add new nodes to stack (up to beam width)
            beam_width = min(len(sorted_nodes), self.tot_config.tot_beam_width)
            exploration_stack.extend(sorted_nodes[:beam_width])
            
        # Compile results
        return {
            'trajectory': best_node.trajectory,
            'step_rewards': [item.get("reward", 0.0) for item in best_node.trajectory if "reward" in item],
            'reward': best_node.reward,
            'env_score': best_node.reward,  # Use cumulative reward as env score for now
            'turns': best_node.depth,
            'valid_actions': len([msg for msg in best_node.trajectory if msg.get("from") == "gpt"]),
            'task_idx': task_idx,
            'done': best_node.terminal,
            'tot_metrics': {
                'explored_branches': explored_branches,
                'max_depth': best_node.depth,
            }
        }
        
    def run_llm_loop(self, gen_batch: DataProto, output_dir: str = None, global_steps: int = 0) -> DataProto:
        """
        Run the LLM interaction loop with Tree of Thought exploration for a batch of initial prompts.
        Overrides the base run_llm_loop method to use ToT exploration.

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

        print(f"[ToT Agent.run_llm_loop] Starting rollout with ToT exploration for batch size: {batch_size} using {num_clients} clients.")

        # Setup initial state tracking
        original_left_side = {'input_ids': initial_prompts_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_prompts_ids[:, []], 
            'responses_with_info_mask': initial_prompts_ids[:, []]
        }
        
        # Initialize active mask and tracking statistics
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Parallel ToT Exploration
        futures = {}
        rollout_results_list = [None] * batch_size  # Preallocate list

        # Submit tasks to the thread pool, distributing across clients
        for i in range(batch_size):
            task_idx = i
            initial_prompt = initial_prompts_ids[i:i+1]  # Keep batch dim

            # Select a client for this task (round-robin)
            client_index = i % num_clients
            selected_client = self.clients[client_index]

            # Submit the ToT exploration task with the selected client
            # Replace standard rollout with ToT exploration
            future = self.executor.submit(self._run_tot_exploration, initial_prompt, task_idx, selected_client)
            futures[future] = i  # Store original batch index

        print(f"[ToT Agent.run_llm_loop] Submitted {batch_size} ToT exploration tasks.")

        # Collect results
        completed_count = 0
        for future in as_completed(futures):
            original_index = futures[future]
            try:
                result_dict = future.result()
                rollout_results_list[original_index] = result_dict
                completed_count += 1

            except Exception as e:
                print(f"[ToT Agent.run_llm_loop] Error collecting result for batch index {original_index}: {e}")
                print(traceback.format_exc())
                # Store a placeholder or error indicator
                rollout_results_list[original_index] = {
                    'trajectory': [], 'step_rewards': [], 'reward': 0.0,
                    'turns': 0, 'env_score': 0.0, 'task_idx': original_index,
                    'error': str(e)
                }

        print(f"[ToT Agent.run_llm_loop] Collected results from {completed_count}/{batch_size} ToT explorations.")

        # Filter out potential None entries if some tasks failed critically
        valid_results = [res for res in rollout_results_list if res is not None]

        if not valid_results:
            print("[ToT Agent.run_llm_loop] Error: No valid rollout results collected.")
            # Return empty DataProto but with correct structure if possible
            empty_proto = DataProto.from_dict({
                "input_ids": torch.empty((0,0), dtype=torch.long),
                "attention_mask": torch.empty((0,0), dtype=torch.long),
                "position_ids": torch.empty((0,0), dtype=torch.long),
                "info_mask": torch.empty((0,0), dtype=torch.long),
                "token_level_rewards": torch.empty((0,0), dtype=torch.float)
            })
            # Add necessary meta_info for downstream compute_log_prob call
            empty_proto.meta_info = {'micro_batch_size': 1}
            return empty_proto

        # Format Results into DataProto - reuse base class implementation
        processed_data = self._convert_rollout_results_to_dataproto(valid_results, gen_batch)

        # Add ToT-specific metrics if available
        tot_metrics = {}
        for result in valid_results:
            if 'tot_metrics' in result:
                for metric_name, metric_value in result['tot_metrics'].items():
                    if metric_name not in tot_metrics:
                        tot_metrics[metric_name] = []
                    tot_metrics[metric_name].append(metric_value)
                    
        # Average ToT metrics
        if tot_metrics:
            avg_tot_metrics = {f"tot_{k}": np.mean(v) for k, v in tot_metrics.items()}
            if 'metrics' not in processed_data.meta_info:
                processed_data.meta_info['metrics'] = {}
            processed_data.meta_info['metrics'].update(avg_tot_metrics)

        # Add necessary meta_info for downstream compute_log_prob
        log_prob_micro_batch_size = getattr(self.actor_rollout_wg, 'log_prob_micro_batch_size', 128)
        if hasattr(self.config, 'actor_rollout_ref') and hasattr(self.config.actor_rollout_ref, 'rollout'):
            log_prob_micro_batch_size = getattr(self.config.actor_rollout_ref.rollout, 'log_prob_micro_batch_size', log_prob_micro_batch_size)
        
        # Ensure these keys exist and have reasonable default values
        if 'micro_batch_size' not in processed_data.meta_info:
            processed_data.meta_info['micro_batch_size'] = log_prob_micro_batch_size
        
        if 'temperature' not in processed_data.meta_info:
            processed_data.meta_info['temperature'] = getattr(self.config, 'temperature', 1.0)
        
        if 'use_dynamic_bsz' not in processed_data.meta_info:
            processed_data.meta_info['use_dynamic_bsz'] = getattr(self.config, 'log_prob_use_dynamic_bsz', False)
        
        # If dynamic batch size is used, also set max_token_len
        if processed_data.meta_info.get('use_dynamic_bsz', False):
            max_token_len = getattr(self.config, 'log_prob_max_token_len_per_gpu', 2048)
            processed_data.meta_info['max_token_len'] = max_token_len

        print(f"[ToT Agent.run_llm_loop] Finished processing ToT exploration results.")
        return processed_data