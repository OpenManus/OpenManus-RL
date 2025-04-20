import re
from typing import Dict, Any, List, Optional

def _normalize_text(text: str) -> str:
    """Lowercase and remove punctuation and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Keep spaces and alphanumeric, remove others
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split()) # Remove extra whitespace
    return text


def compute_score(
    env_name: str,
    **kwargs
    ) -> float:
    """
    Computes a score for an AgentGym environment based on rollout results passed via kwargs.

    It expects the full interaction trajectory and reward model info.

    Args:
        env_name: The name of the AgentGym environment.
        **kwargs: Must contain 'trajectory' (List[Dict]) and 'reward_model_info' (Dict).

    Returns:
        The calculated score as a float (typically 0.0 or 1.0).
    """
    trajectory = kwargs.get('trajectory')
    reward_model_info = kwargs.get('reward_model_info')
    env_name_lower = env_name.lower()
    score = 0.0

    if not trajectory or not reward_model_info:
        print(f"Warning: 'trajectory' or 'reward_model_info' missing in kwargs for env '{env_name}'. Cannot compute score.")
        return 0.0
        
    style = reward_model_info.get("style")

    try:
        # --- WebShop Specific Logic --- 
        if env_name_lower in ["webshop", "webarena", "maze", "wordle", "alfworld", "sciworld", "babyai", "textcraft", "weather", "movie", "academia", "todo", "sheet", "sqlgym"]:
            print(f"Warning: Trajectory-based scoring logic not yet implemented for env '{env_name}'. Returning 0.")
            # Implement specific scoring functions for these envs based on their trajectory structure and success criteria
            score = 0.0 
        
        else:
            print(f"Warning: Unknown AgentGym environment '{env_name}' for reward scoring. Returning 0.")

    except Exception as e:
        print(f"Error computing AgentGym score from trajectory for env='{env_name}', style='{style}': {e}")
        # Optionally log traceback: import traceback; print(traceback.format_exc())
        score = 0.0 # Return 0 on error

    return score 