# verl/utils/reward_score/reward_components.py
import re
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class RewardComponent(ABC):
    """Abstract base class for all reward components."""
    def __init__(self, weight: float = 1.0, name: str = ""):
        """
        Initializes the reward component.

        Args:
            weight (float): The weight to apply to the computed reward score.
            name (str): An optional name for the component. Defaults to the class name.
        """
        self.weight = weight
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, trajectory: List[Dict[str, Any]], **kwargs) -> float:
        """
        Computes the reward score based on the trajectory and other context.
        Must be implemented by subclasses.

        Args:
            trajectory (List[Dict[str, Any]]): A list of dictionaries representing the
                                               conversation or rollout steps. Each dict
                                               typically contains 'from' (e.g., 'human', 'gpt')
                                               and 'value' (the text content). Agent steps
                                               might also include 'reward', 'info', etc.
            **kwargs: Additional context that might be needed, such as original prompt,
                      environment configuration, reward model info, etc.

        Returns:
            float: The computed reward score (before applying the weight).
        """
        pass

    def __call__(self, trajectory: List[Dict[str, Any]], **kwargs) -> float:
        """
        Computes and returns the weighted reward score.

        Args:
            trajectory (List[Dict[str, Any]]): The rollout trajectory.
            **kwargs: Additional context passed to the compute method.

        Returns:
            float: The final weighted reward score for this component.
        """
        # Apply the component's weight to the computed score
        return self.weight * self.compute(trajectory, **kwargs)

# --- Example Concrete Reward Components ---

class GoalReward(RewardComponent):
    """Provides reward based on task success/failure indicated in the final step info."""
    def compute(self, trajectory: List[Dict[str, Any]], **kwargs) -> float:
        """
        Checks the 'info' dictionary of the last step in the trajectory for success indicators.
        Assumes AgentGym-like 'info' structure or relevant keys in 'reward_model_info'.

        Args:
            trajectory: The rollout trajectory.
            **kwargs: May contain 'reward_model_info' for non-AgentGym scenarios.

        Returns:
            1.0 for success, -1.0 for failure (optional), or score value, otherwise 0.0.
        """
        if not trajectory:
            return 0.0

        # Check the last step first, assuming it contains final env info
        last_step = trajectory[-1]
        if isinstance(last_step.get('info'), dict):
            info = last_step['info']
            if info.get('success') is True:
                return 1.0
            if info.get('fail') is True:
                return -1.0  # Optional penalty for failure
            # Use 'score' if success/fail flags are not present
            return float(info.get('score', 0.0))

        # Fallback: Check reward_model_info if provided (for non-AgentGym?)
        reward_model_info = kwargs.get('reward_model_info', {})
        if reward_model_info.get('success') is True: # Example key
             return 1.0

        # Default to 0 if no clear success/failure/score indicator is found
        return 0.0

class LengthPenalty(RewardComponent):
    """Applies a penalty based on the length of the last agent response."""
    def __init__(self, weight: float = -0.01, max_length: int = 500, min_length: int = 10, penalty_type: str = "linear"):
        """
        Initializes the length penalty component.

        Args:
            weight (float): The weight for the penalty (typically negative).
            max_length (int): Responses longer than this will be penalized.
            min_length (int): Responses shorter than this can optionally be penalized.
            penalty_type (str): The type of penalty calculation ('linear', 'quadratic', 'log').
        """
        super().__init__(weight=weight)
        assert max_length > min_length, "max_length must be greater than min_length"
        self.max_length = max_length
        self.min_length = min_length
        self.penalty_type = penalty_type

    def _get_last_response_length(self, trajectory: List[Dict[str, Any]]) -> int:
        """Helper to find the length of the last 'gpt' response."""
        for msg in reversed(trajectory):
            if msg.get('from') == 'gpt':
                # Consider using token length for better consistency
                # For now, using character length
                return len(msg.get('value', ""))
        return 0

    def compute(self, trajectory: List[Dict[str, Any]], **kwargs) -> float:
        """
        Calculates the penalty based on the length of the last agent response.

        Args:
            trajectory: The rollout trajectory.
            **kwargs: Not used by this component.

        Returns:
            A non-positive value representing the penalty (0 if within bounds).
        """
        length = self._get_last_response_length(trajectory)
        penalty = 0.0

        if length > self.max_length:
            diff = length - self.max_length
            if self.penalty_type == "linear":
                penalty = diff
            elif self.penalty_type == "quadratic":
                penalty = diff ** 2
            elif self.penalty_type == "log":
                 # Use log1p to handle diff=0 gracefully (log(1)=0)
                 penalty = np.log1p(diff)
            else:
                 raise ValueError(f"Unknown penalty_type: {self.penalty_type}")
        elif length < self.min_length:
             diff = self.min_length - length
             # Apply similar penalty logic for being too short (optional)
             if self.penalty_type == "linear":
                 penalty = diff # Example: Linear penalty for being too short
             # Add quadratic/log if needed for short responses

        # The penalty value itself is positive or zero, weight makes it negative
        return penalty


class FormatReward(RewardComponent):
    """Rewards responses that match specific regular expression patterns."""
    def __init__(self, weight: float = 0.5, required_patterns: List[str] = None):
        """
        Initializes the format reward component.

        Args:
            weight (float): The reward weight (typically positive).
            required_patterns (List[str]): A list of regex patterns. A reward is given
                                           if the last response matches *any* of these patterns.
        """
        super().__init__(weight=weight)
        # Compile regex patterns for efficiency
        self.required_patterns = [re.compile(p, re.DOTALL) for p in required_patterns] if required_patterns else []

    def _get_last_response(self, trajectory: List[Dict[str, Any]]) -> str:
        """Helper to find the text of the last 'gpt' response."""
        for msg in reversed(trajectory):
            if msg.get('from') == 'gpt':
                return msg.get('value', "")
        return ""

    def compute(self, trajectory: List[Dict[str, Any]], **kwargs) -> float:
        """
        Checks if the last agent response matches any of the required patterns.

        Args:
            trajectory: The rollout trajectory.
            **kwargs: Not used by this component.

        Returns:
            1.0 if a match is found, 0.0 otherwise (before applying weight).
        """
        if not self.required_patterns:
            return 0.0

        last_response = self._get_last_response(trajectory)

        # Check if any pattern matches
        for pattern in self.required_patterns:
            if pattern.search(last_response):
                # Reward is 1.0 if format matches, weight scales it
                return 1.0

        # No pattern matched
        return 0.0

# --- Reward Composer ---
class RewardComposer:
    """Combines multiple reward components to compute a total reward."""
    def __init__(self, components: List[RewardComponent]):
        """
        Initializes the composer with a list of reward components.

        Args:
            components (List[RewardComponent]): The reward components to combine.
        """
        self.components = components

    def compute_total_reward(self, trajectory: List[Dict[str, Any]], **kwargs) -> Tuple[float, Dict[str, float]]:
        """
        Computes the total weighted reward by summing the outputs of all components.

        Args:
            trajectory (List[Dict[str, Any]]): The rollout trajectory.
            **kwargs: Additional context passed to each component's compute method.

        Returns:
            Tuple[float, Dict[str, float]]: A tuple containing:
                - The total weighted reward.
                - A dictionary breaking down the reward contributed by each component (by name).
        """
        total_reward = 0.0
        reward_breakdown = {}
        for component in self.components:
            try:
                # Call the component's __call__ method to get the weighted reward
                component_reward = component(trajectory, **kwargs)
                total_reward += component_reward
                # Store the individual weighted reward for analysis
                reward_breakdown[component.name] = component_reward
            except Exception as e:
                # Log error but continue, potentially assigning 0 reward for the failed component
                print(f"Error computing reward for component {component.name}: {e}")
                reward_breakdown[component.name] = 0.0 # Or handle as needed

        return total_reward, reward_breakdown 