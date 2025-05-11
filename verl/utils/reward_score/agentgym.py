import re
import json  # Added for parsing ground truth conversations
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

# New sequential ground truth trajectory similarity function
def _compute_gt_traj_similarity_reward_sequential(
    generated_actions: List[str], 
    ground_truth_actions: List[str], 
    max_reward: float, 
    min_reward: float
) -> float:
    """Compares generated actions to ground truth actions sequentially.

    If a mismatch occurs at step i, score is i / len(ground_truth_actions).
    Full match up to len(ground_truth_actions) gets full score for this component.
    """
    if not ground_truth_actions: # No ground truth to compare against
        return (max_reward + min_reward) / 2.0 # Neutral score

    if not generated_actions: # Agent produced no actions
        return min_reward

    len_gt = len(ground_truth_actions)
    len_gen = len(generated_actions)
    
    matched_until = 0
    for i in range(min(len_gen, len_gt)):
        # Normalize text for comparison to handle minor differences if needed
        # For now, let's assume direct comparison is intended or normalization is simple
        # if _normalize_text(generated_actions[i]) == _normalize_text(ground_truth_actions[i]):
        # Using direct comparison based on previous _normalize_text usage in the file
        if _normalize_text(generated_actions[i]) == _normalize_text(ground_truth_actions[i]):
            matched_until += 1
        else:
            break # Mismatch found, stop comparing

    raw_score_ratio = matched_until / len_gt
    
    # Scale the raw_score_ratio to the [min_reward, max_reward] range
    final_score = min_reward + (max_reward - min_reward) * raw_score_ratio
    return final_score


# ------------------------------
# Revised Scoring Function
# ------------------------------
def compute_score(solution_str: str, ground_truth: str, step: int = 0, sum_env_rewards: float = 0.0, **kwargs) -> float:
    """Compute a heuristic reward score for AgentGym style trajectories.

    The scoring criteria is kept simple to work with the data available at
    training-time (``solution_str`` and the serialized ``ground_truth``
    conversations):

    1. *Format reward* – Does the agent output contain the required XML-like
       tags (<think>, <memory>, <plan>, <act>) in the correct order?
    2. *Length reward* – Is the content inside the last <think> tag close to
       a target length (number of words)?  This prevents overly short answers.
    3. *Ground-truth action similarity* – How similar are the <act> actions
       generated by the agent to the ground-truth actions contained in the
       reference trajectory (decoded from ``ground_truth``).
    4. *Summed Environment Rewards* - The sum of rewards obtained directly from environment steps during rollout.

    Only these three components are used because we do **not** have access to
    real-time environment rewards here (the PPO loop has already finished).  The
    weights for these four components are set to be equal (0.25 each).

    Returns
    -------
    float
        The weighted sum of the three sub-scores.
    """

    # ------------------------------
    # Component-1:  Format Reward
    # ------------------------------
    FORMAT_MAX_R, FORMAT_MIN_R = 1.0, -0.5 # Max score 1, min score -0.5 for bad format
    format_score_component = _compute_format_reward(
        solution_str,
        FORMAT_MAX_R,
        FORMAT_MIN_R,
        check_all_tags=True,
    )

    # ------------------------------
    # Component-2:  Length Reward  (words inside the **last** <think> tag)
    # ------------------------------
    LENGTH_TARGET_WORDS = 50  # Empirical target length
    LENGTH_MAX_R, LENGTH_MIN_R = 0.5, -0.25 # Max score 0.5, min score -0.25

    # Extract the content of the **last** <think>...</think> pair – this is
    # where the agent is supposed to do chain-of-thought reasoning.
    think_matches = list(re.finditer(r"<think>(.*?)</think>", solution_str, re.DOTALL))
    last_think_content = think_matches[-1].group(1).strip() if think_matches else ""

    length_score_component = _compute_length_reward(
        last_think_content,
        LENGTH_MAX_R,
        LENGTH_MIN_R,
        LENGTH_TARGET_WORDS,
        penalty_if_missing=True,
        too_short_penalty_factor=0.5,
        too_long_penalty_factor=0.5,
        tolerance_factor=0.3,
    )

    # ------------------------------
    # Component-3:  Ground-truth trajectory similarity (action level)
    # ------------------------------
    GT_SIM_MAX_R, GT_SIM_MIN_R = 1.0, 0.0 # Max score 1 (full match), min score 0 (no match at start)

    # 1) Extract generated actions from *solution_str*
    generated_actions = re.findall(r"<act>(.*?)</act>", solution_str, re.DOTALL)

    # 2) Parse *ground_truth* JSON → trajectory dict list, then extract actions
    ground_truth_actions: List[str] = []
    try:
        trajectory_ref = json.loads(ground_truth)
        if isinstance(trajectory_ref, list):
            ground_truth_actions = _extract_actions_from_trajectory(trajectory_ref)
    except Exception as e:
        # Fall back silently; similarity reward will be minimal.
        print(f"[agentgym.compute_score] Warning: failed to parse ground_truth JSON – {e}")

    gt_sim_score_component = _compute_gt_traj_similarity_reward_sequential(
        generated_actions,
        ground_truth_actions,
        GT_SIM_MAX_R,
        GT_SIM_MIN_R,
    )

    # ------------------------------
    # Component-4: Summed Environment Rewards
    # ------------------------------
    # These are rewards directly from env.step() during the rollout phase.
    # We can apply a simple scaling and clipping if needed.
    ENV_SUM_REWARD_SCALE = kwargs.get('env_sum_reward_scale', 1.0) # Allow override via kwargs
    ENV_SUM_REWARD_CLIP = kwargs.get('env_sum_reward_clip', None)  # Allow override via kwargs

    env_sum_score_component = sum_env_rewards * ENV_SUM_REWARD_SCALE
    if ENV_SUM_REWARD_CLIP is not None:
        env_sum_score_component = max(-ENV_SUM_REWARD_CLIP, min(ENV_SUM_REWARD_CLIP, env_sum_score_component))

    # ------------------------------
    # Combine sub-scores (weights sum to 1)
    # ------------------------------
    FORMAT_W, LENGTH_W, GT_SIM_W, ENV_SUM_W = 0.25, 0.25, 0.25, 0.25

    total_score = (
        FORMAT_W * format_score_component
        + LENGTH_W * length_score_component
        + GT_SIM_W * gt_sim_score_component
        + ENV_SUM_W * env_sum_score_component
    )

    return float(total_score) 