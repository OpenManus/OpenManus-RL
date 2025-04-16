import re
from typing import Dict, Any, List, Optional

def _normalize_text(text: str) -> str:
    """Lowercase and remove punctuation and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^ws]', '', text) # Remove punctuation
    text = ' '.join(text.split()) # Remove extra whitespace
    return text

def _compute_webshop_ground_truth_score(
    prediction_text: str, 
    ground_truth_str: Optional[str],
    **kwargs # Consume other potential args
    ) -> float:
    """
    Computes a score for WebShop based on ground truth attributes.

    This is an *approximation* based on text matching in the final prediction.
    It checks if the predicted final action context mentions required attributes.
    A more accurate measure would involve checking the actual final environment state,
    which is not available here.

    Args:
        prediction_text: The final text output from the agent's trajectory.
        ground_truth_str: A comma-separated string of required attributes from input data.

    Returns:
        1.0 if the prediction seems to mention the required attributes before buying, 0.0 otherwise.
    """
    if not prediction_text or not ground_truth_str:
        return 0.0

    # Simple check: Did the agent decide to buy? Look for a common final action pattern.
    # This pattern might need adjustment based on actual agent outputs.
    buy_action_patterns = [r'click[buy now]', r'action:s*buy now'] 
    found_buy = any(re.search(pattern, prediction_text.lower()) for pattern in buy_action_patterns)

    if not found_buy:
         # Agent didn't attempt to buy, score 0 based on this simple check.
         # A more complex check could analyze the final response even without buying.
        return 0.0
        
    # Extract required attributes
    required_attributes = set(attr.strip() for attr in ground_truth_str.split(',') if attr.strip())
    if not required_attributes:
        return 0.0 # No ground truth provided

    # Check if the required attributes are mentioned *somewhere* in the final prediction text.
    # This is a weak proxy for checking if the *correct* item was selected.
    normalized_prediction = _normalize_text(prediction_text)
    
    missing_attributes = 0
    for attr in required_attributes:
        if _normalize_text(attr) not in normalized_prediction:
            missing_attributes += 1
            # print(f"Debug: Missing attribute '{_normalize_text(attr)}'") # for debugging

    # Score based on how many attributes were mentioned (simple version: all or nothing)
    if missing_attributes == 0:
        return 1.0
    else:
        # Could also return a partial score: 1.0 - (missing_attributes / len(required_attributes))
        return 0.0


# --- Main Dispatch Function ---

def compute_agentgym_score(
    prediction: str, 
    reward_model_info: Dict[str, Any], 
    env_name: str, 
    # Add meta_info as an expected argument to potentially get env_score
    meta_info: Optional[Dict[str, Any]] = None, 
    **kwargs # Allow for future flexibility 
    ) -> float:
    """
    Computes a score for an AgentGym environment.
    
    Priority 1: Return the final environment score if available in meta_info.
    Priority 2: Fallback to comparing prediction text with ground_truth if available.

    Args:
        prediction: The final textual output from the agent (used for fallback).
        reward_model_info: The dictionary from the input data's 'reward_model' field (used for fallback).
        env_name: The name of the AgentGym environment.
        meta_info: The meta_info dictionary from the DataProto, potentially containing 'env_scores'.
        **kwargs: Additional arguments.

    Returns:
        A numerical score.
    """
    # --- Priority 1: Get score directly from environment interaction --- 
    if meta_info and 'env_scores' in meta_info:
        # Assuming env_scores is a tensor or list corresponding to the batch
        # This function likely processes one sample at a time, so we need the index.
        # The caller (PPO trainer) needs to handle batch iteration and pass the correct score.
        # For now, let's assume the caller passes the single correct score via kwargs or a direct arg.
        # We add 'env_score' to kwargs check as a simpler way for the caller.
        if 'env_score' in kwargs:
            try:
                env_score = float(kwargs['env_score'])
                # print(f"Debug: Using env_score from kwargs: {env_score}") # Debug
                return env_score
            except (ValueError, TypeError):
                 print(f"Warning: Could not convert provided env_score '{kwargs['env_score']}' to float.")
        else:
             # If the full env_scores list/tensor is in meta_info, we can't easily use it here
             # without knowing the index for the current sample.
             print(f"Debug: 'env_scores' found in meta_info, but individual 'env_score' not passed directly. Falling back.")

    # --- Priority 2: Fallback to text/ground_truth comparison --- 
    # print(f"Debug: env_score not found directly, falling back to text comparison for {env_name}.") # Debug
    style = reward_model_info.get("style")
    env_name_lower = env_name.lower()
    score = 0.0 # Default score for fallback

    KNOWN_AGENTGYM_ENVS = [
        "webshop", "webarena", "maze", "wordle", "alfworld", 
        "sciworld", "babyai", "textcraft", "weather", "movie", 
        "academia", "todo", "sheet", "sqlgym"
    ]

    try:
        if env_name_lower in KNOWN_AGENTGYM_ENVS:
            if style == "ground_truth":
                gt = reward_model_info.get("ground_truth")
                score = _compute_webshop_ground_truth_score(prediction, gt, **kwargs)
            elif style:
                 print(f"Warning (Fallback): Unsupported reward style '{style}' for env '{env_name}'. Returning 0.")
            else:
                 print(f"Warning (Fallback): Missing 'style' in reward_model_info for env '{env_name}'. Returning 0.")
        else:
            print(f"Warning (Fallback): Unknown AgentGym environment '{env_name}'. Returning 0.")

    except Exception as e:
        print(f"Error computing AgentGym fallback score for env='{env_name}', style='{style}': {e}")
        score = 0.0

    return score 