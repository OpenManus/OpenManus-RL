import requests
import logging
from typing import List, Tuple, Optional
from .memory import SimpleMemory

logger = logging.getLogger(__name__)


def simple_summarize(
    history_steps: List[str],
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    env_type: Optional[str] = None,
    model: str = "gpt-4o",
    timeout_s: int = 15,
) -> str:
    """
    Summarize history steps using an LLM API with environment-specific prompts.
    
    Args:
        history_steps: List of formatted history strings
        api_key: OpenAI/Azure API key
        endpoint: API endpoint URL (Azure OpenAI compatible)
        env_type: Environment type hint (e.g., "webshop", "alfworld").
        
    Returns:
        Summarized history string
    """
    if not api_key:
        # Fallback: return truncated recent history
        return "\n".join(history_steps[-3:])  # Last 3 steps
    
    # Join all history into one text
    full_history = "\n".join(history_steps)
    
    # Join all history into one text for the LLM to compress.
    # History lines look like: [Observation t: '...', Action t: '...']

    if (env_type or "").lower().startswith("webshop") or (env_type == "webshop"):
        # WebShop-focused summarization prompt with bounded length and item count.
        prompt = f"""
You are an information extraction assistant.
Given a multi-step WebShop interaction history (search, pagination, product clicks, option selections, detail views), produce a compact, factual snapshot for decision-making.

Output EXACTLY these labeled lines (ASCII only, keep total length <= 700 chars):
- SearchQuery: <exact query or 'unknown'>
- PagesVisited: <Page 1, Page 2, ... or 'unknown'>
- RelevantProducts (max 5):
  [ProductID] — [Product Name] — [Price or Range] — [Attrs: color=..., size=..., material=...]
- Selections: <selected color/size/other or 'none'>
- IrrelevantSummary: <one line about clearly off-target results or 'none'>

Rules:
- Facts only from history; no recommendations, prioritization, or planning.
- Do not speculate; if missing, write 'unknown' or 'none'.
- Prefer products matching the goal (category/color/size/price). If too many, pick up to 5 most on-target.
- Preserve the initial search query exactly as used.

History to summarize:
{full_history}
"""
    else:
        # Default (AlfWorld-style) summarization prompt.
        prompt = f"""Compress this ALFRED history into a current state snapshot.

Output EXACTLY these labeled lines (one line each, ASCII only):
Task:
Location: <last known location or 'unknown'>
Inventory: <items held or 'none'>
Discovered: <key objects/containers with states; aggregate sets; limit to top 5>
KeyEvents: <1-2 important actions and outcomes>

Rules:
- Facts only; no suggestions or analysis.
- Do not copy long quotes; use key nouns.
- If unknown, write 'unknown'.
- Total length <= 600 characters.

History to summarize:
{full_history}
"""

    try:
        # Use OpenAI public API by default; if a custom endpoint is provided, use it as base_url.
        base_url = (endpoint.rstrip("/") if endpoint else "https://api.openai.com/v1")
        url = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You summarize task progress concisely with factual, structured outputs."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            logger.debug(f"Summary generated: {len(content)} chars")
            return content.strip()
        else:
            logger.warning(f"OpenAI API error {response.status_code}, using fallback")
            return "\n".join(history_steps[-3:])
            
    except Exception as e:
        logger.warning(f"Summarization failed: {e}, using fallback")
        return "\n".join(history_steps[-3:])


class SummarizedMemory(SimpleMemory):
    """
    Memory manager with summarization capability.
    Inherits from SimpleMemory and adds optional history summarization.
    """
    
    def __init__(self):
        super().__init__()
        self.summaries = []  # Cache summaries for each environment
        self.last_summary_step = []  # Track when each env was last summarized
        
    def reset(self, batch_size: int):
        """Reset memory and summary caches."""
        super().reset(batch_size)
        self.summaries = [None] * batch_size
        self.last_summary_step = [0] * batch_size
        
    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
        use_summary: bool = False,
        summary_api_key: str = None,
        summary_endpoint: str = None,
        summary_model: Optional[str] = None,
        env_type: Optional[str] = None,
        summary_threshold: Optional[int] = None,  # kept for backward compatibility, ignored
        summary_concurrency: int = 8,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch history with optional summarization.
        
        Design:
        - use_summary == False: Return raw history window controlled by `history_length`.
        - use_summary == True: Always return a summary of ALL recorded steps so far,
          ignoring `history_length`.
        
        Args:
            history_length: Max steps for regular mode (ignored in summary mode)
            obs_key: Key for observations
            action_key: Key for actions  
            use_summary: Whether to use summarization
            summary_api_key: API key for LLM
            summary_endpoint: API endpoint for LLM
            
        Returns:
            Tuple of (memory_contexts, valid_lengths)
        """
        if not use_summary:
            # Use original SimpleMemory behavior
            return super().fetch(history_length, obs_key, action_key)
            
        return self._fetch_with_summary(
            obs_key=obs_key,
            action_key=action_key,
            api_key=summary_api_key,
            endpoint=summary_endpoint,
            env_type=env_type,
            model=summary_model or "gpt-4o",
            concurrency=max(1, int(summary_concurrency) if summary_concurrency else 1),
        )
    
    def _fetch_with_summary(
        self,
        obs_key: str,
        action_key: str,
        api_key: Optional[str],
        endpoint: Optional[str],
        env_type: Optional[str] = None,
        model: str = "gpt-4o",
        concurrency: int = 8,
    ) -> Tuple[List[str], List[int]]:
        """Fetch history using summarization strategy with optional parallel updates.

        The function updates summaries after every new step
        and performs updates in parallel with a bounded thread pool.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        memory_contexts: List[str] = [""] * self.batch_size
        valid_lengths: List[int] = [0] * self.batch_size

        # Determine which envs need summary update this call.
        to_update = []  # list of (env_idx, all_history)
        for env_idx in range(self.batch_size):
            total_steps = len(self._data[env_idx])
            valid_lengths[env_idx] = total_steps
            if total_steps <= 0:
                memory_contexts[env_idx] = ""
                continue

            # Always refresh summary after each new step
            last_step = self.last_summary_step[env_idx] if env_idx < len(self.last_summary_step) else 0
            need_update = (
                self.summaries[env_idx] is None
                or (total_steps - last_step) >= 1
            )

            if need_update:
                # Build full history for summarization.
                all_history = []
                for j, rec in enumerate(self._data[env_idx]):
                    step_num = j + 1
                    act = rec[action_key]
                    obs = rec[obs_key]
                    all_history.append(
                        f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                    )
                to_update.append((env_idx, all_history))
            else:
                # Use cached
                memory_contexts[env_idx] = self.summaries[env_idx] or ""

        # Run summarization for the update list in parallel.
        if to_update:
            max_workers = max(1, concurrency)

            def _summ_one(item):
                idx, hist = item
                try:
                    text = simple_summarize(
                        hist,
                        api_key=api_key,
                        endpoint=endpoint,
                        env_type=env_type,
                        model=model,
                        timeout_s=30,
                    )
                except Exception:
                    text = "\n".join(hist[-3:])
                return idx, text

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_summ_one, item) for item in to_update]
                for fut in as_completed(futs):
                    idx, text = fut.result()
                    # Update caches
                    self.summaries[idx] = text
                    self.last_summary_step[idx] = len(self._data[idx])
                    memory_contexts[idx] = text

        return memory_contexts, valid_lengths
    
    def _get_or_create_summary(
        self, 
        env_idx: int, 
        obs_key: str, 
        action_key: str,
        api_key: str,
        endpoint: Optional[str],
        env_type: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> str:
        """Get existing summary or create a new one."""
        total_steps = len(self._data[env_idx])
        
        # Update summary whenever step count has advanced (or first time)
        if self.summaries[env_idx] is None or total_steps != self.last_summary_step[env_idx]:
            
            # Create formatted history for all steps
            all_history = []
            for j, rec in enumerate(self._data[env_idx]):
                step_num = j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                all_history.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )
            
            # Generate summary with environment-specific template
            self.summaries[env_idx] = simple_summarize(
                all_history,
                api_key=api_key,
                endpoint=endpoint,
                env_type=env_type,
                model=model,
            )
            self.last_summary_step[env_idx] = total_steps
            
            logger.debug(f"Updated summary for env {env_idx}, covering {total_steps} steps")
            
        return self.summaries[env_idx]
