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
import copy
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import os


DFSDT_DIVERSITY_PROMPT = (
    "This state has been explored multiple times. Previous attempts and their outcomes were:\n"
    "{previous_attempts}\n"
    "Generate new action ideas that differ from every listed attempt while remaining executable."
)


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
                if not actions:
                    # Some templates use double quotes instead
                    actions = re.findall(r'"([^"]+)"', block)
                if not actions:
                    # Fall back to numbered enumerations such as 1. take apple
                    actions = re.findall(r"\d+\.?\s*([^\n\r]+)", block)
                # Filter trivial entries
                actions = [a.strip() for a in actions if a and a.strip()]
                return actions
        return []
    except Exception:
        return []


@dataclass
class SearchParams:
    beam_size: int = 3                  # Number of candidates to expand per state
    value_threshold: float = 0.15       # Prune if top value below this
    max_try: int = 10                   # Maximum complete trajectory attempts
    max_depth: Optional[int] = None     # Defaults to episode max steps if None
    diversity_back_steps: int = 2       # DFSDT: steps to backtrack on failure
    diversity_back_steps_alt: int = 3   # DFSDT: escalated backtrack if still failing
    propose_k: int = 4                  # Number of proposals when env doesn't supply actions
    history_window: int = 4             # Number of recent steps to expose in prompts
    history_observation_trim: int = 400 # Trim each observation snippet to this length


class ValueScorer:
    """Wraps an agent to score candidate actions with a single LLM call.

    The scorer returns scores in [0, 1]. We use a simple format to be robust.
    """

    def __init__(self, agent, env_type: str):
        self.agent = agent
        self.env_type = env_type

    def score(
        self,
        obs: str,
        candidates: List[str],
        history: Optional[str] = None,
    ) -> List[float]:
        """Score all candidates in a single call with a strict rubric and tie‑breakers.

        Behavior:
        - Requests a JSON array with one float per candidate in the same order.
        - Encourages using the full 0.0–1.0 range and penalizing redundant/invalid actions.
        - If the first pass yields low variance or malformed output, performs a strict re‑scoring pass.
        - As a last resort, applies a deterministic tie‑breaker jitter to avoid flat rankings.
        """
        if not candidates:
            return []

        def _parse_scores(text: str, n: int) -> List[float]:
            try:
                arr = json.loads(text)
                if isinstance(arr, list):
                    vals = [float(x) for x in arr[:n]]
                    return vals
            except Exception:
                pass
            floats = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            try:
                vals = [float(x) for x in floats[:n]]
                return vals
            except Exception:
                return []

        def _normalize(scores: List[float], n: int) -> List[float]:
            if not scores:
                return [0.0] * n
            scores = [max(0.0, min(1.0, float(s))) for s in scores]
            if len(scores) < n:
                scores += [0.0] * (n - len(scores))
            return scores[:n]

        cand_json = json.dumps(candidates, ensure_ascii=False)
        history_section = ""
        if history:
            history_section = (
                "Recent trajectory (oldest to newest):\n"
                f"{history.strip()}\n\n"
            )

        base_prompt = (
            f"You are evaluating candidate NEXT actions for an agent in {self.env_type}.\n"
            f"Rate how promising each action is for achieving the goal from the CURRENT state.\n"
            f"{history_section}Current observation:\n{obs}\n\n"
            f"Candidates (JSON list): {cand_json}\n\n"
            "Scoring rubric (strict):\n"
            "- Use the FULL 0.0–1.0 range. Do NOT give all 1.0 or all equal scores.\n"
            "- 0.9–1.0: Directly and obviously advances the goal with minimal risk.\n"
            "- 0.7–0.9: Strongly promising next step.\n"
            "- 0.4–0.7: Plausible but uncertain; depends on missing preconditions.\n"
            "- 0.2–0.4: Weak progress or likely redundant.\n"
            "- 0.0–0.2: Invalid, circular, or contradicts recent history/admissible options.\n"
            "- Penalize repeating the same action that just failed, no‑ops, or irrelevant moves.\n"
            "Return ONLY a JSON array of floats, aligned to input order, length equals number of candidates."
        )

        strict_prompt = (
            base_prompt + "\n\nImportant: Provide a spread of scores across candidates; avoid ties."
        )

        try:
            txt = self.agent.get_action_from_llm(strict_prompt)
            scores = _normalize(_parse_scores(txt, len(candidates)), len(candidates))

            # If variance is too low or all equal, attempt one strict re‑scoring pass
            if len(scores) == len(candidates):
                if max(scores) - min(scores) < 0.05 or all(abs(s - scores[0]) < 1e-6 for s in scores):
                    reinforce_prompt = (
                        base_prompt
                        + "\n\nYour previous scores lacked differentiation. Now RESCORE STRICTLY:"
                        + " At least one candidate must be <= 0.25 and at least one must be >= 0.75,"
                        + " unless all are truly equivalent (rare). Return ONLY the JSON array."
                    )
                    txt2 = self.agent.get_action_from_llm(reinforce_prompt)
                    scores2 = _normalize(_parse_scores(txt2, len(candidates)), len(candidates))
                    if max(scores2) - min(scores2) >= 0.05:
                        scores = scores2

            # Final guard: deterministic tie‑breaker jitter to avoid flat rankings
            if max(scores) - min(scores) < 1e-3:
                def _jitter(s: str) -> float:
                    h = hashlib.md5(s.encode("utf-8")).hexdigest()
                    # Map hash to small epsilon in [0, 0.02)
                    return (int(h[:6], 16) % 2000) / 100000.0
                scores = [max(0.0, min(1.0, s + _jitter(candidates[i]))) for i, s in enumerate(scores)]

            return scores
        except Exception as e:
            logging.warning(f"Value scoring failed, defaulting zeros: {e}")
            return [0.0] * len(candidates)


def propose_candidates(
    agent,
    obs: str,
    env_type: str,
    k: int,
    avoid: Optional[List[str]] = None,
    diversity_context: Optional[str] = None,
    history: Optional[str] = None,
) -> List[str]:
    """Ask the model to propose up to k actionable next steps. Avoid duplicates in avoid list."""
    avoid = avoid or []
    diversity_desc = ""
    if diversity_context:
        diversity_desc = "\n\n" + diversity_context.strip()
    elif avoid:
        diversity_desc = "\nPreviously tried at this state: " + json.dumps(avoid, ensure_ascii=False)
    history_desc = ""
    if history:
        history_desc = "\nRecent trajectory (oldest to newest):\n" + history.strip() + "\n"
    prompt = (
        f"You are choosing the agent's next action in {env_type}.\n"
        f"{history_desc}Current observation:\n{obs}\n\n"
        f"Propose up to {k} different, concrete next actions the agent could take next.\n"
        f"Each proposal must be a single executable action string in the exact format expected by the environment.\n"
        f"Avoid repeating ineffective or redundant actions from the history above.\n"
        f"Return only a JSON list of strings, no extra text.{diversity_desc}"
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
    single_attempt_fn: Callable[[int], Dict],  # Now takes attempt_idx parameter
    task_dir: Optional[str] = None,
) -> Dict:
    """Run N independent attempts and return the first successful result if any.

    single_attempt_fn must encapsulate a single rollout attempt from scratch and return
    the same structure as run_environment_with_retry does. We stop on first success.
    """
    best = None
    all_attempts = []
    
    for i in range(max(1, int(N))):
        logging.info(f"[Best-of-N] Attempt {i + 1}/{N}")
        res = single_attempt_fn(i + 1)  # Pass attempt number
        all_attempts.append({
            "attempt": i + 1,
            "won": res.get("won", False),
            "reward": res.get("reward", 0.0),
            "steps": res.get("steps", 0)
        })
        
        if best is None:
            best = res
        # Prefer successful
        if res.get("won", False):
            res["strategy"] = "best_of_n"
            res["bon_attempt"] = i + 1
            res["all_bon_attempts"] = all_attempts
            return res
    
    # None succeeded; return the last/best available
    if best is None:
        best = {"won": False, "reward": 0.0, "steps": 0}
    best["strategy"] = "best_of_n"
    best["bon_attempt"] = N
    best["all_bon_attempts"] = all_attempts
    
    # Save Best-of-N summary to task directory
    if task_dir:
        try:
            import os
            import json
            summary_file = os.path.join(task_dir, "best_of_n_summary.json")
            with open(summary_file, "w") as f:
                json.dump({
                    "strategy": "best_of_n",
                    "total_attempts": N,
                    "won": best.get("won", False),
                    "best_attempt": best.get("bon_attempt", N),
                    "all_attempts": all_attempts
                }, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save Best-of-N summary: {e}")
    
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
    value_agent=None,
) -> Dict:
    """Generic DFS-style search with value scoring. Implements ToT-DFS and DFSDT.

    The search replays actions from the root on every expansion so we can backtrack
    without modifying the underlying environment state in-place. For ToT we expand
    up to ``beam_size`` children ordered by model value scores and rely on DFS with
    backtracking to explore alternative reasoning paths. DFSDT keeps a single best
    child per depth, but performs diversity-driven backtracking when trajectories
    fail to terminate successfully.
    """

    assert mode in ("tot", "dfsdt")
    scorer_agent = value_agent or agent
    scorer = ValueScorer(scorer_agent, env_type)

    # Trajectory tracking across the final chosen path or best-effort path
    trajectory_steps: List[Dict[str, Any]] = []
    all_attempts: List[Dict[str, Any]] = []

    # State for DFSDT diversity: map state_key -> tried candidates
    tried_at_state: Dict[str, List[str]] = {}
    state_action_history: Dict[str, List[Dict[str, Any]]] = {}

    # Logging structure to persist search process for analysis
    search_log: Dict[str, Any] = {
        "mode": mode,
        "beam_size": params.beam_size,
        "value_threshold": params.value_threshold,
        "max_try": params.max_try,
        "max_depth": params.max_depth or max_steps,
        "diversity_back_steps": params.diversity_back_steps,
        "diversity_back_steps_alt": params.diversity_back_steps_alt,
        "propose_k": params.propose_k,
        "expansions": [],
        "trajectories": [],
        "termination_reason": None,
    }

    def _extract_state_signature(obs_text: str) -> str:
        try:
            match = re.search(
                r"current observation is:\s*(.*?)(?:\n\s*Your admissible actions|$)",
                obs_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                return match.group(1).strip()
        except Exception:
            pass
        return obs_text.strip()[:1024]

    def state_key_from_obs(obs_text: str) -> str:
        signature = _extract_state_signature(obs_text)
        return hashlib.md5(signature.encode("utf-8", "ignore")).hexdigest()

    def record_step(
        step_idx: int,
        observation_before: str,
        observation_after: str,
        action: str,
        reward: float,
        done: bool,
        info: Dict,
        history_context: Optional[str],
        candidate_pool: List[str],
        ranked_candidates: List[str],
        candidate_scores: List[float],
        chosen_rank_index: Optional[int],
        chosen_original_index: Optional[int],
        state_key: str,
    ) -> None:
        candidate_pool = list(candidate_pool)
        ranked_candidates = list(ranked_candidates)
        candidate_scores = [float(s) for s in candidate_scores]

        entry: Dict[str, Any] = {
            "step": int(step_idx),
            "observation": observation_after,
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "won": bool(info.get("won", False)),
            "candidate_pool": candidate_pool,
            "ranked_candidates": ranked_candidates,
            "candidate_scores": candidate_scores,
        }
        if chosen_rank_index is not None:
            entry["chosen_rank_index"] = int(chosen_rank_index)
        if chosen_original_index is not None:
            entry["chosen_original_index"] = int(chosen_original_index)

        trajectory_steps.append(entry)

    # Initial reset
    obs_dict, info_dict = env_manager.reset_single(0)
    obs = obs_dict["text"][0]
    info = info_dict[0] if isinstance(info_dict, list) else info_dict
    initial_obs_text = obs

    def _trim_history_observation(text: str) -> str:
        if not text:
            return ""
        limit = max(32, int(params.history_observation_trim)) if params.history_observation_trim else 0
        snippet = text.strip()
        if not limit or len(snippet) <= limit:
            return snippet
        return snippet[: max(0, limit - 3)] + "..."

    def build_history_context(depth: int) -> Optional[str]:
        if not trajectory_steps or depth <= 0:
            return None
        if params.history_window is not None and params.history_window <= 0:
            return None

        entries = []
        max_depth = min(depth, len(trajectory_steps))
        for idx in range(max_depth):
            step = trajectory_steps[idx]
            entries.append({
                "idx": step.get("step", idx),
                "action": step.get("action", ""),
                "reward": float(step.get("reward", 0.0)),
                "done": bool(step.get("done", False)),
                "won": bool(step.get("won", False)),
                "obs_after": step.get("observation", ""),
            })

        if not entries:
            return None

        window = params.history_window or len(entries)
        selected = entries[-window:]
        omitted = len(entries) - len(selected)

        lines: List[str] = []
        if omitted > 0:
            lines.append(f"... {omitted} earlier step(s) omitted ...")

        if selected:
            first_idx = selected[0]["idx"]
            before_obs = initial_obs_text if first_idx == 0 else None
            if before_obs is None and 0 < first_idx <= len(trajectory_steps):
                prev_entry = trajectory_steps[first_idx - 1]
                before_obs = prev_entry.get("observation")
            before_obs_trimmed = _trim_history_observation(before_obs or "")
            if before_obs_trimmed:
                lines.append(f"Observation before step {first_idx}:\n{before_obs_trimmed}")

        for entry in selected:
            status_bits = []
            if entry["done"]:
                status_bits.append("done")
            if entry["won"]:
                status_bits.append("won")
            status_str = f"; {' '.join(status_bits)}" if status_bits else ""
            reward_str = f"{entry['reward']:.2f}".rstrip("0").rstrip(".")
            lines.append(
                f"Step {entry['idx']} action: {entry['action']} (reward {reward_str}{status_str})"
            )
            obs_after = _trim_history_observation(entry.get("obs_after", ""))
            if obs_after:
                lines.append(f"Observation after:\n{obs_after}")

        return "\n".join(lines)

    def snapshot_trajectory(length: int) -> List[Dict[str, Any]]:
        return copy.deepcopy(trajectory_steps[:length])

    first_attempt_success = False
    dfsdt_failure_streak = 0
    current_path: List[str] = []
    step_idx = 0
    won = False
    final_info = {}

    def finalize_attempt(
        termination: str,
        won_flag: bool,
        reward_value: float,
        path_actions: List[str],
        info_snapshot: Optional[Dict[str, Any]],
        trajectory_snapshot: List[Dict[str, Any]],
    ) -> None:
        nonlocal first_attempt_success
        attempt_idx = len(all_attempts) + 1
        safe_info: Dict[str, Any] = {}
        if info_snapshot is not None:
            try:
                safe_info = json.loads(json.dumps(info_snapshot, default=str))
            except Exception:
                safe_info = {}
        attempt_record = {
            "attempt": attempt_idx,
            "won": bool(won_flag),
            "reward": float(reward_value),
            "steps": len(path_actions),
            "actions": path_actions,
            "trajectory": trajectory_snapshot,
            "termination": termination,
            "info": safe_info,
        }
        all_attempts.append(attempt_record)
        search_log["trajectories"].append({
            "attempt": attempt_idx,
            "won": bool(won_flag),
            "reward": float(reward_value),
            "steps": len(path_actions),
            "termination": termination,
        })
        if won_flag and attempt_idx == 1:
            first_attempt_success = True

    def next_candidates_for_state(obs_text: str, info: Dict, avoid: Optional[List[str]] = None) -> List[str]:
        state_key = state_key_from_obs(obs_text)
        history = state_action_history.get(state_key, [])
        diversity_context = None
        avoid_set: List[str] = []
        if avoid:
            avoid_set.extend(avoid)
        if history:
            avoid_set.extend([h.get("action", "") for h in history])
            avoid_set = [a for a in avoid_set if a]
            if mode == "dfsdt" and history:
                attempts_json = json.dumps(history, ensure_ascii=False, indent=2)
                diversity_context = DFSDT_DIVERSITY_PROMPT.format(previous_attempts=attempts_json)
        history_text = build_history_context(len(current_path))
        avail: List[str] = []
        if isinstance(info, dict):
            for key in (
                "admissible_commands",
                "admissible_actions",
                "available_commands",
                "available_actions",
                "commands",
            ):
                maybe = info.get(key)
                if not maybe:
                    continue
                if isinstance(maybe, dict):
                    maybe = list(maybe.values())
                if isinstance(maybe, (list, tuple)):
                    avail = [str(a).strip() for a in maybe if str(a).strip()]
                    if avail:
                        break
        if not avail:
            avail = extract_actions_from_obs(obs_text, env_type)
        if avail:
            seen = set()
            avail = [a for a in avail if not (a in seen or seen.add(a))]
        if not avail:
            avail = propose_candidates(
                agent,
                obs_text,
                env_type,
                params.propose_k,
                avoid=avoid_set,
                diversity_context=diversity_context,
                history=history_text,
            )
        if avoid_set:
            seen_avoid = set(avoid_set)
            avail = [a for a in avail if a not in seen_avoid]
        return avail[: params.beam_size] if mode == "tot" else avail

    stack: List[Tuple[int, str, Dict, List[str], int]] = []
    first_candidates = next_candidates_for_state(obs, info)
    stack.append((0, obs, info, first_candidates, 0))

    while stack and len(all_attempts) < params.max_try:
        depth, node_obs, node_info, cand_list, next_i = stack.pop()

        if depth != len(current_path):
            obs_dict, info_dict, reached = replay_until(env_manager, current_path[:depth])
            obs = obs_dict["text"][0]
            info = info_dict[0] if isinstance(info_dict, list) else info_dict
            step_idx = reached
            current_path = current_path[:depth]
            trajectory_steps[:] = trajectory_steps[:step_idx]
        else:
            obs = node_obs
            info = node_info
            step_idx = depth
            current_path = current_path[:depth]

        state_key = state_key_from_obs(obs)
        # Avoid repeating the same action at the same state across the entire search,
        # for both ToT and DFSDT. This ensures when we backtrack to a state, we pick
        # a previously untried candidate.
        tried_here = tried_at_state.get(state_key, [])

        if not cand_list:
            cand_list = next_candidates_for_state(obs, info, tried_here)

        if not cand_list:
            # Dead end at this state: backtrack via stack without counting an attempt.
            search_log["expansions"].append({
                "depth": depth,
                "obs_key": state_key_from_obs(obs),
                "candidates": [],
                "scores": [],
                "action": None,
                "event": "dead_end_no_candidates",
            })
            continue

        history_for_node = build_history_context(step_idx)
        raw_scores = scorer.score(obs, cand_list, history=history_for_node) if cand_list else []
        if raw_scores and len(raw_scores) < len(cand_list):
            raw_scores = raw_scores + [0.0] * (len(cand_list) - len(raw_scores))

        if raw_scores:
            ranked = sorted(
                zip(cand_list, raw_scores),
                key=lambda x: x[1] if x[1] is not None else 0.0,
                reverse=True,
            )
        else:
            ranked = [(a, None) for a in cand_list]

        ordered_candidates = [a for a, _ in ranked]
        ordered_scores = [s if s is not None else 0.0 for _, s in ranked]

        chosen = None
        if mode == "tot":
            # Only prune by value if we actually obtained numeric scores
            if raw_scores and ordered_scores and ordered_scores[0] < params.value_threshold:
                search_log["expansions"].append({
                    "depth": depth,
                    "obs_key": state_key,
                    "candidates": ordered_candidates,
                    "scores": ordered_scores,
                    "action": None,
                    "event": "prune_by_value",
                })
                # Pruned by heuristic value; do not count as a full attempt.
                continue
            if ordered_candidates:
                chosen = ordered_candidates[0]
                if len(ordered_candidates) > 1:
                    stack.append((depth, obs, info, ordered_candidates[1:], 0))
        else:
            if next_i >= len(ordered_candidates):
                continue
            chosen = ordered_candidates[next_i]
            if next_i + 1 < len(ordered_candidates):
                stack.append((depth, obs, info, ordered_candidates, next_i + 1))

        if not chosen:
            continue

        if step_idx >= (params.max_depth or max_steps):
            logging.debug(
                f"[{mode.upper()}] Reached max depth {params.max_depth or max_steps} at depth {depth}, pruning branch"
            )
            finalize_attempt(
                termination="max_depth",
                won_flag=False,
                reward_value=0.0,
                path_actions=current_path.copy(),
                info_snapshot=info,
                trajectory_snapshot=snapshot_trajectory(step_idx),
            )
            if mode == "dfsdt":
                dfsdt_failure_streak += 1
                back_steps = params.diversity_back_steps_alt if dfsdt_failure_streak > 1 else params.diversity_back_steps
                trim_depth = max(0, depth - max(0, back_steps))
                current_path = current_path[:trim_depth]
                trajectory_steps[:] = trajectory_steps[:trim_depth]
                step_idx = trim_depth
                stack = [frame for frame in stack if frame[0] <= trim_depth]
            else:
                stack = [frame for frame in stack if frame[0] <= depth]
            if len(all_attempts) >= params.max_try:
                break
            continue

        search_log["expansions"].append({
            "depth": depth,
            "obs_key": state_key,
            "candidates": ordered_candidates,
            "scores": ordered_scores,
            "action": chosen,
            "event": "choose_and_step",
            "history": history_for_node,
        })

        original_candidates = list(cand_list)
        try:
            chosen_rank_index = ordered_candidates.index(chosen)
        except ValueError:
            chosen_rank_index = None
        try:
            chosen_original_index = original_candidates.index(chosen)
        except ValueError:
            chosen_original_index = None

        # Environment adapters expect markup like <plan>...</plan><action>...</action>
        # for WebShop/AlfWorld. Wrap plain candidates accordingly.
        env_action = chosen
        if env_type in ("alfworld", "webshop"):
            env_action = f"<plan>follow admissible action</plan><action>{chosen}</action>"

        obs_dict, reward_dict, done_dict, info_dict = env_manager.step_single(0, env_action)
        new_obs = obs_dict["text"][0]
        reward = float(reward_dict[0])
        done = bool(done_dict[0])
        new_info = info_dict[0] if isinstance(info_dict, list) else info_dict

        record_step(
            step_idx,
            obs,
            new_obs,
            chosen,
            reward,
            done,
            new_info,
            history_for_node,
            original_candidates,
            ordered_candidates,
            ordered_scores,
            chosen_rank_index,
            chosen_original_index,
            state_key,
        )

        current_path.append(chosen)
        step_idx += 1

        tried_at_state.setdefault(state_key, []).append(chosen)
        state_action_history.setdefault(state_key, []).append({
            "action": chosen,
            "reward": reward,
            "done": done,
            "won": bool(new_info.get("won", False)),
            "observation": new_obs[:120],
        })

        if done:
            won = bool(new_info.get("won", False))
            if won:
                final_info = new_info
            attempt_path = current_path.copy()
            attempt_traj = snapshot_trajectory(step_idx)
            finalize_attempt(
                termination="success" if won else "terminal_failure",
                won_flag=won,
                reward_value=reward,
                path_actions=attempt_path,
                info_snapshot=new_info,
                trajectory_snapshot=attempt_traj,
            )

            if won:
                logging.info(
                    f"[{mode.upper()}] SUCCESS! Won on trajectory {len(all_attempts)}"
                )
                break

            if len(all_attempts) >= params.max_try:
                logging.info(f"[{mode.upper()}] Reached max_try limit, stopping search")
                break

            if mode == "dfsdt":
                dfsdt_failure_streak += 1
                back_steps = params.diversity_back_steps_alt if dfsdt_failure_streak > 1 else params.diversity_back_steps
                trim_depth = max(0, depth - max(0, back_steps))
                current_path = current_path[:trim_depth]
                trajectory_steps[:] = trajectory_steps[:trim_depth]
                stack = [frame for frame in stack if frame[0] <= trim_depth]
                step_idx = trim_depth
            else:
                current_path = current_path[:depth]
                trajectory_steps[:] = trajectory_steps[:depth]
                stack = [frame for frame in stack if frame[0] <= depth]
                step_idx = depth
            continue

        next_cands = next_candidates_for_state(new_obs, new_info)
        stack.append((len(current_path), new_obs, new_info, next_cands, 0))

    if won:
        search_log["termination_reason"] = "success"
    elif len(all_attempts) >= params.max_try:
        search_log["termination_reason"] = "max_try_reached"
    elif not stack:
        search_log["termination_reason"] = "search_space_exhausted"
    else:
        search_log["termination_reason"] = "unknown"

    search_log["complete_trajectories"] = len(all_attempts)

    if dump_cb:
        try:
            dump_cb(search_log)
        except Exception as exc:  # pragma: no cover - external callback
            logging.debug(f"Failed to execute dump callback: {exc}")

    res = {
        "env_id": 0,
        "won": bool(won),
        "first_attempt_success": bool(first_attempt_success),
        "reward": 1.0 if won else 0.0,
        "retries": max(1, len(all_attempts)) if all_attempts else 1,
        "steps": len(trajectory_steps),
        "env_type": env_type,
        "trajectory": trajectory_steps,
        "all_attempts": all_attempts,
        "strategy": mode,
        "termination_reason": search_log.get("termination_reason", "unknown"),
        "search_attempts": all_attempts,
        "search_log": search_log,
        "final_info": final_info,
    }

    if task_dir:
        try:
            os.makedirs(task_dir, exist_ok=True)
            summary_path = os.path.join(task_dir, "task_search_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "strategy": mode,
                            "env_type": env_type,
                            "beam_size": params.beam_size,
                            "value_threshold": params.value_threshold,
                            "max_try": params.max_try,
                            "won": bool(won),
                        },
                        "search_log": search_log,
                        "final_result": {
                            "won": bool(won),
                            "steps": len(trajectory_steps),
                            "reward": res["reward"],
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as exc:  # pragma: no cover - best effort
            logging.warning(f"Failed to persist search summary: {exc}")

    return res
