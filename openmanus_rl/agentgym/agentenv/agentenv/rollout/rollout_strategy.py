from typing import List, Dict, Any, Optional, TypedDict, Union
from abc import ABC, abstractmethod
import copy
import random

from agentenv.controller import BaseEnvClient, StepOutput
from agentenv.controller.task import ConversationMessage, ExperienceOutput
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig


class TrajectoryNode(TypedDict):
    """Node in a trajectory tree/graph"""
    observation: str
    thought: str
    action: str
    reward: float
    done: bool
    children: List['TrajectoryNode']
    

class IRolloutStrategy(ABC):
    """Interface for rollout strategies"""
    @abstractmethod
    def execute(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase, 
        client: BaseEnvClient, 
        initial_observation: str,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None
    ) -> List[ExperienceOutput]:
        """Execute the strategy and return trajectories"""
        pass


class BaseRolloutStrategy(IRolloutStrategy):
    """Base class for rollout strategies with common functionality"""
    
    def __init__(self, name: str):
        self.name = name
    
    def _create_experience_output(
        self, 
        conversation: List[ConversationMessage],
        reward: float,
        text: str,
        tokenizer: PreTrainedTokenizerBase
    ) -> ExperienceOutput:
        """Convert conversation to ExperienceOutput format"""
        tokenized = tokenizer.encode(text, add_special_tokens=False)
        return ExperienceOutput(
            conversation=conversation,
            reward=reward,
            text=text,
            seq_ids=tokenized,
            attention_mask=[1] * len(tokenized),
            action_mask=[0] * len(tokenized)  # Simplified; actual mask would be more complex
        )


class StandardReActStrategy(BaseRolloutStrategy):
    """Standard ReAct strategy used in original AgentGym"""
    
    def __init__(self):
        super().__init__("StandardReAct")
    
    def execute(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase, 
        client: BaseEnvClient, 
        initial_observation: str,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None
    ) -> List[ExperienceOutput]:
        """Execute standard ReAct strategy"""
        # This is basically the same as BaseTask._generate_experience_one
        # but adapted to our interface
        
        conversation = list(client.conversation_start)
        conversation.append(
            ConversationMessage({"from": "human", "loss": None, "value": initial_observation})
        )
        
        text = ""
        for msg in conversation:
            if msg["from"] == "human":
                text += f"\nHuman: {msg['value']}"
            else:
                text += f"\nAssistant: {msg['value']}"
        
        rounds = 0
        reward = 0.0
        done = False
        
        while not done:
            if max_rounds is not None and rounds >= max_rounds:
                break
                
            # Generate agent's response
            input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
            output = model.generate(
                input_ids,
                generation_config=generation_config
            )
            
            # Decode response
            generated_text = tokenizer.decode(
                output[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Add to conversation
            conversation.append(
                ConversationMessage({"from": "gpt", "loss": True, "value": generated_text})
            )
            text += f"\nAssistant: {generated_text}"
            
            # Take action in environment
            step_output = client.step(generated_text)
            state, reward, done = step_output.state, step_output.reward, step_output.done
            
            # Add environment feedback to conversation
            conversation.append(
                ConversationMessage({"from": "human", "loss": None, "value": state})
            )
            text += f"\nHuman: {state}"
            
            rounds += 1
        
        # Create and return a single trajectory
        experience = self._create_experience_output(
            conversation=conversation,
            reward=reward,
            text=text,
            tokenizer=tokenizer
        )
        
        return [experience]


class ToTStrategy(BaseRolloutStrategy):
    """Tree of Thoughts strategy"""
    
    def __init__(self, num_branches: int = 3, depth: int = 2):
        super().__init__("ToT")
        self.num_branches = num_branches
        self.depth = depth
    
    def execute(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase, 
        client: BaseEnvClient, 
        initial_observation: str,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None
    ) -> List[ExperienceOutput]:
        """Execute Tree of Thoughts strategy"""
        # Generate trajectory tree
        tree = self._generate_tree(
            model=model,
            tokenizer=tokenizer,
            client=client,
            observation=initial_observation,
            conversation=[
                *list(client.conversation_start),
                ConversationMessage({"from": "human", "loss": None, "value": initial_observation})
            ],
            depth=self.depth,
            generation_config=generation_config,
            max_rounds=max_rounds
        )
        
        # Extract all paths from tree
        trajectories = self._extract_paths(tree, tokenizer)
        
        return trajectories
    
    def _generate_tree(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        client: BaseEnvClient,
        observation: str,
        conversation: List[ConversationMessage],
        depth: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        current_round: int = 0
    ) -> TrajectoryNode:
        """Generate a trajectory tree node and its children recursively"""
        # Clone the client to avoid altering the original
        client_copy = copy.deepcopy(client)
        
        # Build text from conversation
        text = ""
        for msg in conversation:
            if msg["from"] == "human":
                text += f"\nHuman: {msg['value']}"
            else:
                text += f"\nAssistant: {msg['value']}"
        
        # Generate agent's response
        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
        
        # Create the current node
        node = {
            "observation": observation,
            "thought": "",
            "action": "",
            "reward": 0.0,
            "done": False,
            "children": []
        }
        
        # Check if we've reached max rounds
        if max_rounds is not None and current_round >= max_rounds:
            return node
        
        # Generate multiple branches
        for _ in range(self.num_branches):
            # Add temperature to promote diversity
            branch_generation_config = copy.deepcopy(generation_config)
            branch_generation_config.temperature = 0.7  # Adjust as needed
            
            output = model.generate(
                input_ids,
                generation_config=branch_generation_config
            )
            
            # Decode response
            generated_text = tokenizer.decode(
                output[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Create branch conversation
            branch_conversation = copy.deepcopy(conversation)
            branch_conversation.append(
                ConversationMessage({"from": "gpt", "loss": True, "value": generated_text})
            )
            
            # Take action in environment
            branch_client = copy.deepcopy(client_copy)
            step_output = branch_client.step(generated_text)
            state, reward, done = step_output.state, step_output.reward, step_output.done
            
            # Add environment feedback
            branch_conversation.append(
                ConversationMessage({"from": "human", "loss": None, "value": state})
            )
            
            # Create child node
            child_node = {
                "observation": observation,
                "thought": "",  # Could extract a thought if using a specific format
                "action": generated_text,
                "reward": reward,
                "done": done,
                "children": []
            }
            
            # Recursively generate children if not at max depth and not done
            if depth > 1 and not done:
                child_children = self._generate_tree(
                    model=model,
                    tokenizer=tokenizer,
                    client=branch_client,
                    observation=state,
                    conversation=branch_conversation,
                    depth=depth-1,
                    generation_config=generation_config,
                    max_rounds=max_rounds,
                    current_round=current_round+1
                )
                child_node["children"] = [child_children]
            
            node["children"].append(child_node)
        
        return node
    
    def _extract_paths(
        self, 
        tree: TrajectoryNode, 
        tokenizer: PreTrainedTokenizerBase,
        path: Optional[List[Dict[str, Any]]] = None
    ) -> List[ExperienceOutput]:
        """Extract all paths from tree and convert to ExperienceOutput"""
        if path is None:
            path = []
        
        # Add current node to path
        current_path = path + [{
            "observation": tree["observation"],
            "action": tree["action"]
        }]
        
        # If leaf node, convert path to ExperienceOutput
        if not tree["children"]:
            # Build conversation from path
            conversation = []
            for step in current_path:
                if step["observation"]:
                    conversation.append(
                        ConversationMessage({"from": "human", "loss": None, "value": step["observation"]})
                    )
                if step["action"]:
                    conversation.append(
                        ConversationMessage({"from": "gpt", "loss": True, "value": step["action"]})
                    )
            
            # Build text from conversation
            text = ""
            for msg in conversation:
                if msg["from"] == "human":
                    text += f"\nHuman: {msg['value']}"
                else:
                    text += f"\nAssistant: {msg['value']}"
            
            # Create ExperienceOutput
            experience = self._create_experience_output(
                conversation=conversation,
                reward=tree["reward"],
                text=text,
                tokenizer=tokenizer
            )
            
            return [experience]
        
        # Recursively extract paths from children
        trajectories = []
        for child in tree["children"]:
            trajectories.extend(self._extract_paths(child, tokenizer, current_path))
        
        return trajectories


class MCTSStrategy(BaseRolloutStrategy):
    """Monte Carlo Tree Search strategy"""
    
    # TODO