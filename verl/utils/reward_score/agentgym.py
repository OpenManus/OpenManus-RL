import re
from typing import Dict, Any, List, Optional
from collections import Counter # For potential use in advanced text matching

# Helper to normalize text, might be useful for content comparison
def _normalize_text(text: str) -> str:
    """Lowercase and remove punctuation and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Keep spaces and alphanumeric, remove others
    text = re.sub(r'[^a-z0-9\\s]', '', text) # Keep original regex
    text = ' '.join(text.split()) # Remove extra whitespace
    return text

# --- Component 1: Environment Reward Summation ---
def _compute_env_reward_sum(trajectory: List[Dict], reward_scale: float = 1.0, reward_clip: Optional[float] = None) -> float:
    """
    Calculates the sum of rewards directly obtained from the environment steps.
    These are typically stored in the 'reward' field of turns from the 'env' or associated with 'gpt' turns.
    """
    raw_env_rewards = []
    # Iterate through the trajectory to find rewards associated with agent actions or env feedback
    for i, turn in enumerate(trajectory):
        if turn.get('from') == 'gpt': # Agent's turn
            # Check if the reward for this action is stored in this turn
            if 'reward' in turn and isinstance(turn['reward'], (int, float)):
                raw_env_rewards.append(float(turn['reward']))
            # Or if it's in the subsequent 'env' turn's info (less common for this arg structure)
            # This part might be double-counting if 'reward' is already on 'gpt' turn based on env step output.
            # elif i + 1 < len(trajectory) and trajectory[i+1].get('from') == 'env' and \\
            #      'reward' in trajectory[i+1] and isinstance(trajectory[i+1]['reward'], (int, float)):
            #     raw_env_rewards.append(float(trajectory[i+1]['reward']))

    sum_env_reward = sum(raw_env_rewards)
    
    scaled_reward = sum_env_reward * reward_scale
    if reward_clip is not None:
        scaled_reward = max(-reward_clip, min(reward_clip, scaled_reward))
        
    return scaled_reward

# --- Component 2: Format Reward ---
def _compute_format_reward(
    full_agent_generation_text: str, 
    max_reward: float, 
    min_reward: float,
    check_all_tags: bool = True 
    ) -> float:
    """
    Checks if the agent's output adheres to the specified format.
    Format: <think> ...<memory>...</memory> ...<plan>...</plan>...<think> <act>...</act>
    """
    if not full_agent_generation_text:
        return min_reward

    text_to_check = re.sub(r'\\s+', ' ', full_agent_generation_text).strip()
    score = min_reward # Default to min_reward

    if check_all_tags:
        # Pattern for the full sequence: <think>...<memory>...</memory>...<plan>...</plan>...<think>...</think>...<act>...</act>
        # This regex is complex and greedy. It tries to find one instance of this structure.
        # It allows any characters (including newlines due to re.DOTALL) within the tags and between them.
        full_pattern = r"<think>.*?</think>.*?<memory>.*?</memory>.*?<plan>.*?</plan>.*?<think>.*?</think>.*?<act>.*?</act>"
        
        # Check for presence of individual tags for partial credit
        has_think = bool(re.search(r"<think>.*?</think>", text_to_check, re.DOTALL))
        has_memory = bool(re.search(r"<memory>.*?</memory>", text_to_check, re.DOTALL))
        has_plan = bool(re.search(r"<plan>.*?</plan>", text_to_check, re.DOTALL))
        has_act = bool(re.search(r"<act>.*?</act>", text_to_check, re.DOTALL))
        num_think_tags = len(re.findall(r"<think>.*?</think>", text_to_check, re.DOTALL))

        if re.search(full_pattern, text_to_check, re.DOTALL):
            score = max_reward
        elif num_think_tags >= 1 and has_memory and has_plan and has_act:
            # All key components present, but maybe not in the perfect full sequence or with extra stuff
            score = (max_reward + min_reward) / 1.5 # Generous partial credit
        elif has_think and has_act : # Minimal: at least one think and one act
            score = (max_reward + min_reward) / 2.0 
        # else score remains min_reward
            
    else: # Simpler check for just a final <think>...<act> sequence
        # Looks for a think block followed by an act block, possibly with whitespace.
        # This is usually for the last action segment.
        simple_pattern = r"<think>.*?</think>\s*<act>.*?</act>"
        if re.search(simple_pattern, text_to_check, re.DOTALL):
            score = max_reward
        # else score remains min_reward
            
    return score

# --- Component 3: Length Reward ---
def _compute_length_reward(
    text_content: str, 
    max_reward: float, 
    min_reward: float, 
    target_len_words: int,
    penalty_if_missing: bool = True,
    too_short_penalty_factor: float = 0.5, 
    too_long_penalty_factor: float = 0.5,
    tolerance_factor: float = 0.2 # e.g., +/- 20% of target_len_words
    ) -> float:
    """
    Rewards based on the length of the provided text content (in words).
    """
    if not text_content:
        return min_reward if penalty_if_missing else (min_reward + max_reward) / 2

    num_words = len(text_content.split())

    if num_words == 0 and penalty_if_missing:
        return min_reward
    
    if target_len_words <=0: # Avoid division by zero if target length is invalid
        return (min_reward + max_reward) / 2 

    lower_bound = target_len_words * (1 - tolerance_factor)
    upper_bound = target_len_words * (1 + tolerance_factor)

    if lower_bound <= num_words <= upper_bound:
        return max_reward
    elif num_words < lower_bound:
        shortage_ratio = num_words / lower_bound
        # Reward decreases from max_reward as it gets shorter
        # Example: if num_words is 0, score is min_reward. If num_words is just below lower_bound, score is slightly less than max_reward.
        # This formula gives a linear ramp from min_reward to a point just below max_reward.
        # (1 - too_short_penalty_factor) controls how quickly it drops.
        # A simpler approach: score = max_reward - ( (lower_bound - num_words) / lower_bound ) * (max_reward - min_reward) * too_short_penalty_factor
        # Let's use: reward based on proximity to target, scaled by penalty factor for being too short.
        # Max penalty (max_reward - min_reward) * too_short_penalty_factor
        # Actual penalty = Max_penalty * (1 - shortage_ratio)
        penalty = (max_reward - min_reward) * too_short_penalty_factor * (1.0 - shortage_ratio)
        return max(min_reward, max_reward - penalty)

    else: # num_words > upper_bound
        # Penalize for being too long, similar logic
        excess_ratio = (num_words - upper_bound) / upper_bound # How much percentage wise it's over
        penalty = (max_reward - min_reward) * too_long_penalty_factor * min(1.0, excess_ratio) # Cap penalty effect
        return max(min_reward, max_reward - penalty)


# --- Component 4: Ground Truth Trajectory Similarity ---
def _extract_actions_from_trajectory(trajectory: List[Dict]) -> List[str]:
    """Extracts content from <act>...</act> tags from 'gpt' turns."""
    actions = []
    act_pattern = r"<act>(.*?)</act>"
    for turn in trajectory:
        if turn.get('from') == 'gpt':
            value = turn.get('value', '')
            # Find all non-overlapping matches in the string
            matches = re.findall(act_pattern, value, re.DOTALL)
            actions.extend([match.strip() for match in matches])
    return actions

def _compute_gt_traj_similarity_reward(
    generated_actions: List[str], 
    ground_truth_actions: List[str], 
    max_reward: float, 
    min_reward: float
    ) -> float:
    """
    Compares a list of extracted agent actions with a list of ground truth actions.
    Uses a simple precision-like score based on sequential matching.
    """
    if not ground_truth_actions:
        # If no GT actions, it's hard to score. Neutral or max? Let's go neutral.
        return (max_reward + min_reward) / 2 

    if not generated_actions: # Agent took no valid actions
        return min_reward

    len_gt = len(ground_truth_actions)
    
    matches = 0
    gt_idx = 0
    # Try to match generated actions against GT actions in order
    for gen_act in generated_actions:
        if gt_idx < len_gt and _normalize_text(gen_act) == _normalize_text(ground_truth_actions[gt_idx]):
            matches += 1
            gt_idx += 1 # Move to next GT action only if current one matched
            
    # Similarity is the ratio of matched GT actions to total GT actions
    similarity = matches / len_gt if len_gt > 0 else 0.0 
    
    score = min_reward + (max_reward - min_reward) * similarity
    return score


def compute_score(
    env_name: str, 
    **kwargs
    ) -> float:
    """
    Computes a composite score for an AgentGym environment trajectory.

    Args:
        env_name: The name of the AgentGym environment.
        **kwargs: Expected to contain:
            - 'trajectory' (List[Dict]): The agent's interaction log.
                 Each dict: {'from': 'gpt'/'env', 'value': str, 'reward': float (from env step), ...}
            - 'reward_model_info' (Dict, optional): Contains parameters and ground truth. E.g.:
                - 'ground_truth_actions': List[str] (for GT trajectory comparison)
                - 'env_reward_weight', 'env_reward_scale', 'env_reward_clip'
                - 'format_reward_weight', 'format_max_r', 'format_min_r', 'format_check_all_tags'
                - 'length_reward_weight', 'length_max_r', 'length_min_r', 
                  'length_target_words', 'length_penalty_if_missing', 
                  'length_too_short_penalty_factor', 'length_too_long_penalty_factor', 'length_tolerance_factor'
                - 'gt_sim_reward_weight', 'gt_sim_max_r', 'gt_sim_min_r'
            - 'step' (int, optional): Current training step (for potential future scheduling).

    Returns:
        The calculated composite score as a float.
    """
    trajectory = kwargs.get('trajectory')
    reward_model_info = kwargs.get('reward_model_info') if kwargs.get('reward_model_info') is not None else {}
    current_step = kwargs.get('step', 0) 

    if not trajectory:
        print(f"Warning: 'trajectory' missing in kwargs for env '{env_name}'. Cannot compute score. Returning 0.0.")
        return 0.0
        
    # --- Define default weights and parameters ---
    env_reward_weight = float(reward_model_info.get('env_reward_weight', 0.25))
    env_reward_scale = float(reward_model_info.get('env_reward_scale', 1.0))
    # Clip summed env reward; if None, no clipping
    env_reward_clip_val = reward_model_info.get('env_reward_clip', 5.0) 
    env_reward_clip = float(env_reward_clip_val) if env_reward_clip_val is not None else None

    format_reward_weight = float(reward_model_info.get('format_reward_weight', 0.25))
    format_max_r = float(reward_model_info.get('format_max_r', 1.0))
    format_min_r = float(reward_model_info.get('format_min_r', -0.5)) # Allow penalty for bad format
    format_check_all_tags = bool(reward_model_info.get('format_check_all_tags', True))

    length_reward_weight = float(reward_model_info.get('length_reward_weight', 0.15))
    length_max_r = float(reward_model_info.get('length_max_r', 0.5)) # Max reward for good length might be less than 1
    length_min_r = float(reward_model_info.get('length_min_r', -0.25)) 
    length_target_words = int(reward_model_info.get('length_target_words', 50)) 
    length_penalty_if_missing = bool(reward_model_info.get('length_penalty_if_missing', True))
    length_too_short_penalty_factor = float(reward_model_info.get('length_too_short_penalty_factor', 0.5))
    length_too_long_penalty_factor = float(reward_model_info.get('length_too_long_penalty_factor', 0.5))
    length_tolerance_factor = float(reward_model_info.get('length_tolerance_factor', 0.3))


    gt_sim_reward_weight = float(reward_model_info.get('gt_sim_reward_weight', 0.35))
    gt_sim_max_r = float(reward_model_info.get('gt_sim_max_r', 1.0))
    gt_sim_min_r = float(reward_model_info.get('gt_sim_min_r', 0.0))
    ground_truth_actions = reward_model_info.get('ground_truth_actions', [])

    # --- Component 1: Environment Reward Summation ---
    env_reward_score_component = _compute_env_reward_sum(trajectory, env_reward_scale, env_reward_clip)
    
    # --- Consolidate Agent Text for Format/Length ---
    agent_generations_text = ""
    if isinstance(trajectory, list): 
        agent_generations_text = "\\n".join([turn['value'] for turn in trajectory if turn.get('from') == 'gpt' and isinstance(turn.get('value'), str)])
    else:
        print(f"Warning: Unexpected trajectory format: {type(trajectory)}. Format/length/GT rewards might be inaccurate.")

    # --- Component 2: Format Reward ---
    format_score_component = _compute_format_reward(
        agent_generations_text, format_max_r, format_min_r, format_check_all_tags
    )

    # --- Component 3: Length Reward (e.g., for combined <think> content) ---
    all_think_content = ""
    if agent_generations_text:
        think_pattern = r"<think>(.*?)</think>"
        for match in re.finditer(think_pattern, agent_generations_text, re.DOTALL):
            all_think_content += match.group(1).strip() + " "
        all_think_content = all_think_content.strip()
    
    length_score_component = _compute_length_reward(
        all_think_content, length_max_r, length_min_r, length_target_words,
        length_penalty_if_missing, length_too_short_penalty_factor, length_too_long_penalty_factor,
        length_tolerance_factor
    )

    # --- Component 4: Ground Truth Trajectory Similarity ---
    generated_actions = []
    if isinstance(trajectory, list): 
        generated_actions = _extract_actions_from_trajectory(trajectory) 
    
    gt_sim_score_component = _compute_gt_traj_similarity_reward(
        generated_actions, ground_truth_actions, gt_sim_max_r, gt_sim_min_r
    )
    
    # --- Total Score ---
    total_score = (
        env_reward_weight * env_reward_score_component +
        format_reward_weight * format_score_component +
        length_reward_weight * length_score_component +
        gt_sim_reward_weight * gt_sim_score_component
    )
    
    # Overall clipping/scaling if desired, e.g., to a standard range like [-1, 1] or [0, 1]
    # For example, if weights sum to 1, this might not be strictly needed unless components can be large.
    # total_score = max(-1.0, min(1.0, total_score)) # Example clip

    # For debugging, print individual scores:
    # print(f"[compute_score] Env '{env_name}': \\
    #       EnvR_raw={env_reward_score_component:.2f} (w={env_reward_weight:.2f}), \\
    #       FmtR_raw={format_score_component:.2f} (w={format_reward_weight:.2f}), \\
    #       LenR_raw={length_score_component:.2f} (w={length_reward_weight:.2f}), \\
    #       GtSimR_raw={gt_sim_score_component:.2f} (w={gt_sim_reward_weight:.2f}) --- TOTAL_raw={total_score:.2f}")

    return total_score 