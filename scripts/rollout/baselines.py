"""
Baselines for rollout strategies: Best-of-N, Tree-of-Thought DFS (ToT-DFS), and DFSDT.

These strategies plug into the existing environment managers via reset() and step(),
and reuse UnifiedAgent for proposal and value scoring prompts. We keep prompts simple
and environment-agnostic, while leveraging available action lists when present in the
observation text (AlfWorld/WebShop) or infos.
"""

from __future__ import annotations

import json
import re
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import os


# Lightweight utils to extract available actions from the observation text
def extract_actions_from_obs(obs_text: str, env_type: str) -> List[str]:
    """Extract available/admissible actions if the environment embeds them in text.

    - AlfWorld/WebShop templates include a bracketed list. We parse strings within quotes.
    - If nothing is found, return empty list to signal fallback to model proposals.
    """
    try:
        # Find bracketed block following key phrase; be permissive across envs/templates
        if env_type in ("alfworld", "webshop"):
            # Look for content within square brackets, then quoted entries inside
            # Choose the last bracketed list to avoid history blocks
            bracket_blocks = re.findall(r"\[(.*?)\]", obs_text, flags=re.DOTALL)
            if bracket_blocks:
                block = bracket_blocks[-1]
                # Extract entries within single quotes '...'
                actions = re.findall(r"'([^']+)'", block)
                # For WebShop template we sometimes have commas at line ends; quotes parsing handles it
                # Filter trivial entries
                actions = [a.strip() for a in actions if a.strip()]
                return actions
        return []
    except Exception:
        return []


@dataclass
class SearchParams:
    beam_size: int = 3                  # Number of candidates to expand per state
    value_threshold: float = 0.15       # Prune if top value below this
    max_nodes: int = 100                # Safety budget for expansions
    max_depth: Optional[int] = None     # Defaults to episode max steps if None
    diversity_back_steps: int = 2       # DFSDT: steps to backtrack on failure
    diversity_back_steps_alt: int = 3   # DFSDT: escalated backtrack if still failing
    propose_k: int = 4                  # Number of proposals when env doesn't supply actions


class ValueScorer:
    """Wraps an agent to score candidate actions with a single LLM call.

    The scorer returns scores in [0, 1]. We use a simple format to be robust.
    """

    def __init__(self, agent, env_type: str):
        self.agent = agent
        self.env_type = env_type

    def score(self, obs: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        # Build a compact JSON request to encourage a JSON map of action->score
        cand_json = json.dumps(candidates, ensure_ascii=False)
        prompt = (
            f"You are evaluating candidate next actions for an agent in {self.env_type}.\n"
            f"Observation:\n{obs}\n\n"
            f"Candidate actions (JSON list): {cand_json}\n\n"
            f"For each action, assign a usefulness score between 0.0 and 1.0 inclusive,\n"
            f"where higher is more promising. Return a JSON array of scores aligned with the input order."
        )
        try:
            txt = self.agent.get_action_from_llm(prompt)
            # Try to parse a JSON array; fallback to extract floats
            scores: List[float] = []
            try:
                maybe = json.loads(txt)
                if isinstance(maybe, list):
                    scores = [float(x) for x in maybe][: len(candidates)]
            except Exception:
                pass
            if not scores:
                floats = re.findall(r"\d+\.\d+|\d+", txt)
                scores = [min(max(float(x), 0.0), 1.0) for x in floats][: len(candidates)]
            # Pad/trim to length
            if len(scores) < len(candidates):
                scores += [0.0] * (len(candidates) - len(scores))
            return scores[: len(candidates)]
        except Exception as e:
            logging.warning(f"Value scoring failed, defaulting zeros: {e}")
            return [0.0] * len(candidates)


def propose_candidates(agent, obs: str, env_type: str, k: int, avoid: Optional[List[str]] = None) -> List[str]:
    """Ask the model to propose up to k actionable next steps. Avoid duplicates in avoid list."""
    avoid = avoid or []
    avoid_desc = ("\nPreviously tried at this state: " + json.dumps(avoid, ensure_ascii=False)) if avoid else ""
    prompt = (
        f"You are choosing the agent's next action in {env_type}.\n"
        f"Observation:\n{obs}\n\n"
        f"Propose up to {k} different, concrete next actions the agent could take next.\n"
        f"Each proposal must be a single executable action string in the exact format expected by the environment.\n"
        f"Return only a JSON list of strings, no extra text.{avoid_desc}"
    )
    try:
        txt = agent.get_action_from_llm(prompt)
        arr = json.loads(txt)
        if isinstance(arr, list):
            # Deduplicate while preserving order
            out, seen = [], set()
            for a in arr:
                if not isinstance(a, str):
                    continue
                a = a.strip()
                if a and a not in seen and a not in avoid:
                    out.append(a)
                    seen.add(a)
                if len(out) >= k:
                    break
            return out
    except Exception:
        pass
    # Fallback: return empty to signal caller to use default policy
    return []


def replay_until(env_manager, actions: List[str]) -> Tuple[Dict, Dict, int]:
    """Reset env and replay a sequence of actions. Return (obs, info, depth_reached)."""
    obs_dict, info_dict = env_manager.reset_single(0)
    obs = obs_dict["text"][0]
    info = info_dict[0] if isinstance(info_dict, list) else info_dict
    if not actions:
        return obs_dict, info_dict, 0
    for i, a in enumerate(actions):
        obs_dict, reward_dict, done_dict, info_dict = env_manager.step_single(0, a)
        obs = obs_dict["text"][0]
        done = bool(done_dict[0])
        if done:
            return obs_dict, info_dict, i + 1
    return obs_dict, info_dict, len(actions)


def run_best_of_n(
    N: int,
    env_manager,
    agent,
    max_steps: int,
    env_type: str,
    single_attempt_fn: Callable[[], Dict],
) -> Dict:
    """Run N independent attempts and return the first successful result if any.

    single_attempt_fn must encapsulate a single rollout attempt from scratch and return
    the same structure as run_environment_with_retry does. We stop on first success.
    """
    best = None
    for i in range(max(1, int(N))):
        logging.info(f"[Best-of-N] Attempt {i + 1}/{N}")
        res = single_attempt_fn()
        if best is None:
            best = res
        # Prefer successful
        if res.get("won", False):
            res["strategy"] = "best_of_n"
            res["bon_attempt"] = i + 1
            return res
    # None succeeded; return the last/best available
    if best is None:
        best = {"won": False, "reward": 0.0, "steps": 0}
    best["strategy"] = "best_of_n"
    best["bon_attempt"] = N
    return best


def run_tree_search(
    env_manager,
    agent,
    max_steps: int,
    env_type: str,
    params: SearchParams,
    mode: str = "tot",  # "tot" or "dfsdt"
    dump_cb: Optional[Callable[[Dict], None]] = None,
    task_dir: Optional[str] = None,
    attempt_id: Optional[int] = None,
) -> Dict:
    """Generic DFS-style search with value scoring. Implements ToT-DFS and DFSDT.

    We maintain a current path of actions. To backtrack, we reset and replay actions
    up to the desired depth. For ToT: expand top-k children (from available list or
    model proposals), score and sort, DFS into the best first, and prune by value_threshold.
    For DFSDT: greedily choose the single best proposal; on failure at leaf, backtrack by
    diversity_back_steps (or diversity_back_steps_alt on next failure) and enforce diversity.
    """
    assert mode in ("tot", "dfsdt")
    scorer = ValueScorer(agent, env_type)

    # Trajectory tracking across the final chosen path or best-effort path
    trajectory_steps: List[Dict[str, Any]] = []
    all_attempts: List[Dict[str, Any]] = []

    # State for DFSDT diversity: map state_key -> tried candidates
    tried_at_state: Dict[str, List[str]] = {}

    # Logging structure to persist search process for analysis
    search_log: Dict[str, Any] = {
        "mode": mode,
        "beam_size": params.beam_size,
        "value_threshold": params.value_threshold,
        "max_nodes": params.max_nodes,
        "max_depth": params.max_depth or max_steps,
        "diversity_back_steps": params.diversity_back_steps,
        "diversity_back_steps_alt": params.diversity_back_steps_alt,
        "propose_k": params.propose_k,
        "expansions": [],  # list of dict per expansion
    }

    def state_key_from_obs(obs_text: str) -> str:
        # Use a compact key; observations can be long, so truncate hash-like
        return str(abs(hash(obs_text)) % (10 ** 12))

    # Helper to record a step
    def record_step(step_idx: int, obs: str, action: str, reward: float, done: bool, info: Dict):
        trajectory_steps.append({
            "step": step_idx,
            "observation": obs,
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "won": bool(info.get("won", False)),
        })

    # Depth-first search using replay on backtrack
    best_found = {"won": False, "reward": 0.0, "steps": 0, "trajectory": []}
    expansions = 0

    # Initial reset
    obs_dict, info_dict = env_manager.reset_single(0)
    obs = obs_dict["text"][0]
    info = info_dict[0] if isinstance(info_dict, list) else info_dict

    current_path: List[str] = []  # actions
    step_idx = 0
    won = False
    final_info = {}

    def next_candidates_for_state(obs_text: str, info: Dict, avoid: Optional[List[str]] = None) -> List[str]:
        # Prefer explicit list when available; otherwise ask model to propose
        avail = extract_actions_from_obs(obs_text, env_type)
        if not avail:
            avail = propose_candidates(agent, obs_text, env_type, params.propose_k, avoid)
        # Enforce diversity if requested
        if avoid:
            avail = [a for a in avail if a not in avoid]
        # Cap to beam_size for ToT; DFSDT uses greedy selection anyway
        return avail[: params.beam_size] if mode == "tot" else avail

    # DFS stack: each frame = (depth, obs_text, info, candidate_list, next_index)
    stack: List[Tuple[int, str, Dict, List[str], int]] = []
    # Initialize first node
    first_candidates = next_candidates_for_state(obs, info)
    stack.append((0, obs, info, first_candidates, 0))

    while stack and expansions < params.max_nodes and step_idx < (params.max_depth or max_steps):
        depth, node_obs, node_info, cand_list, next_i = stack.pop()
        # Refresh state by replaying to 'depth'
        if depth != len(current_path):
            obs_dict, info_dict, reached = replay_until(env_manager, current_path[:depth])
            obs = obs_dict["text"][0]
            info = info_dict[0] if isinstance(info_dict, list) else info_dict
            step_idx = reached
            # Trim trajectory_steps if we backtracked
            trajectory_steps[:] = trajectory_steps[:step_idx]
            # Reset diversity record downstream if needed (keep global record for DFSDT)
        else:
            obs = node_obs
            info = node_info

        # If no candidates available, try to propose
        tried_here = tried_at_state.get(state_key_from_obs(obs), []) if mode == "dfsdt" else []
        if not cand_list:
            cand_list = next_candidates_for_state(obs, info, tried_here)

        # For ToT: score and sort; for DFSDT: pick a single best
        if mode == "tot":
            # Score and prune
            scores = scorer.score(obs, cand_list)
            # Pair and sort high->low
            ranked = sorted(zip(cand_list, scores), key=lambda x: x[1], reverse=True)
            cand_list = [a for a, s in ranked if s is not None]
            if ranked and ranked[0][1] < params.value_threshold:
                # Prune this subtree
                search_log["expansions"].append({
                    "depth": depth,
                    "obs_key": state_key_from_obs(obs),
                    "candidates": cand_list,
                    "scores": scores,
                    "action": None,
                    "event": "prune_by_value",
                })
                continue
            # Expand best-first: push siblings (after the first) then go into first
            if len(cand_list) > 1:
                # Push back the remaining candidates for later expansion
                stack.append((depth, obs, info, cand_list[1:], 0))
            # Choose the first candidate to execute now
            chosen = cand_list[0] if cand_list else None
        else:
            # DFSDT: greedy choose one; if none, propose from model
            if not cand_list:
                cand_list = propose_candidates(agent, obs, env_type, max(1, params.propose_k), tried_here)
            if not cand_list:
                # Give up at this state
                continue
            # Score to pick best one; do not prune by threshold at this step
            scores = scorer.score(obs, cand_list)
            best_idx = max(range(len(cand_list)), key=lambda i: scores[i] if i < len(scores) else 0.0)
            chosen = cand_list[best_idx]

        if not chosen:
            continue

        # Apply chosen action
        expansions += 1
        search_log["expansions"].append({
            "depth": depth,
            "obs_key": state_key_from_obs(obs),
            "candidates": cand_list,
            "scores": scores if 'scores' in locals() else [],
            "action": chosen,
            "event": "choose_and_step",
        })
        obs_dict, reward_dict, done_dict, info_dict = env_manager.step_single(0, chosen)
        new_obs = obs_dict["text"][0]
        reward = float(reward_dict[0])
        done = bool(done_dict[0])
        new_info = info_dict[0] if isinstance(info_dict, list) else info_dict
        record_step(step_idx, new_obs, chosen, reward, done, new_info)
        current_path.append(chosen)
        step_idx += 1

        # DFSDT diversity bookkeeping
        if mode == "dfsdt":
            key = state_key_from_obs(obs)
            tried_at_state.setdefault(key, []).append(chosen)

        if done:
            won = bool(new_info.get("won", False))
            final_info = new_info
            if won:
                break
            # DFSDT backtracking on failure: back off N steps and try diversity
            if mode == "dfsdt":
                # First backtrack N steps; if already tried, escalate
                back = params.diversity_back_steps
                if step_idx >= 2 and not won:
                    back = params.diversity_back_steps_alt if len(current_path) >= (params.diversity_back_steps + 1) else params.diversity_back_steps
                new_depth = max(0, len(current_path) - back)
                # Replay to new_depth
                obs_dict, info_dict, reached = replay_until(env_manager, current_path[:new_depth])
                obs_back = obs_dict["text"][0]
                info_back = info_dict[0] if isinstance(info_dict, list) else info_dict
                # Prepare diverse candidates by avoiding previously tried at this state
                tried_here = tried_at_state.get(state_key_from_obs(obs_back), [])
                diverse = next_candidates_for_state(obs_back, info_back, tried_here)
                # Push back the backtracked node with new candidate list
                search_log["expansions"].append({
                    "depth": new_depth,
                    "obs_key": state_key_from_obs(obs_back),
                    "candidates": diverse,
                    "scores": [],
                    "action": None,
                    "event": "backtrack_and_diversify",
                })
                stack.append((new_depth, obs_back, info_back, diverse, 0))
                # Trim path and trajectory
                current_path[:] = current_path[:new_depth]
                trajectory_steps[:] = trajectory_steps[:new_depth]
                step_idx = new_depth
                continue
            # ToT: if terminal and not won, backtrack by one to try siblings if any
            continue

        # Not done; continue DFS from the new state
        next_cands = next_candidates_for_state(new_obs, new_info)
        stack.append((len(current_path), new_obs, new_info, next_cands, 0))

    # Build result
    res = {
        "env_id": 0,
        "won": bool(won),
        "first_attempt_success": bool(won),  # single attempt notion here
        "reward": 1.0 if won else 0.0,
        "retries": 1,
        "steps": len(trajectory_steps),
        "env_type": env_type,
        "trajectory": trajectory_steps,
        "all_attempts": None,
        "strategy": mode,
    }
    # Persist logs if requested
    if task_dir and attempt_id is not None:
        try:
            os.makedirs(task_dir, exist_ok=True)
            with open(os.path.join(task_dir, f"attempt_{attempt_id}_trajectory.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "retry_idx": attempt_id - 1,
                    "trajectory": trajectory_steps,
                    "won": res["won"],
                    "reward": res["reward"],
                    "steps": res["steps"],
                    "mode": mode,
                }, f, ensure_ascii=False, indent=2)
            with open(os.path.join(task_dir, f"search_attempt_{attempt_id}_log.json"), "w", encoding="utf-8") as f:
                json.dump(search_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save search logs: {e}")
    return res
