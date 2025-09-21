import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import requests

from .memory import SimpleMemory

logger = logging.getLogger(__name__)


def simple_summarize(
    history_steps: List[str],
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    env_type: Optional[str] = None,
    model: str = "gpt-4o",
    timeout_s: int = 30,
) -> str:
    """Summarize history steps using the OpenAI Chat Completions API.

    Args:
        history_steps: List of formatted history strings.
        api_key: OpenAI API key.
        endpoint: Optional OpenAI base URL (defaults to public endpoint).
        env_type: Environment type hint for prompt selection.
        model: Target OpenAI model name.
        timeout_s: HTTP timeout in seconds.

    Returns:
        Summarized history string.
    """
    if not api_key:
        # Fallback: return truncated recent history
        return "\n".join(history_steps[-3:])  # Last 3 steps
    
    # Join all history into one text
    full_history = "\n".join(history_steps)
    
    env_type_norm = (env_type or "").strip().lower()
    if env_type_norm.startswith("webshop") or env_type_norm == "webshop":
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
        base_url = endpoint.rstrip("/") if endpoint else "https://api.openai.com/v1"
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You summarize task progress concisely with factual, structured outputs.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            logger.debug("Summary generated: %d chars", len(content))
            return content.strip()
        logger.warning("API error %s, using fallback", response.status_code)
        return "\n".join(history_steps[-3:])

    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Summarization failed: %s, using fallback", exc)
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
        summary_concurrency: Optional[int] = None,
        summary_threshold: Optional[int] = None,  # kept for backward compatibility, ignored
        **kwargs,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch history with optional summarization.
        
        Strategy:
        - 1 step: return original history (no summarization needed)  
        - >1 steps: return summarized history (information compression)
        
        Args:
            history_length: Max steps for regular mode (ignored in summary mode)
            obs_key: Key for observations
            action_key: Key for actions  
            use_summary: Whether to use summarization
            summary_api_key: API key for LLM
            summary_endpoint: API endpoint for LLM
            summary_model: Optional model/deployment identifier for the LLM
            env_type: Optional environment identifier (affects prompt template)
            summary_concurrency: Optional concurrency hint for summary generation
            
        Returns:
            Tuple of (memory_contexts, valid_lengths)
        """
        if kwargs:
            logger.debug("Ignoring extra summary kwargs: %s", list(kwargs.keys()))
        if not use_summary:
            # Use original SimpleMemory behavior
            return super().fetch(history_length, obs_key, action_key)
            
        return self._fetch_with_summary(
            obs_key,
            action_key,
            summary_api_key,
            summary_endpoint,
            summary_model,
            env_type,
            summary_concurrency,
        )
    
    def _fetch_with_summary(
        self, 
        obs_key: str, 
        action_key: str,
        api_key: str,
        endpoint: str,
        summary_model: Optional[str],
        env_type: Optional[str],
        summary_concurrency: Optional[int],
    ) -> Tuple[List[str], List[int]]:
        """Fetch history using summarization strategy."""
        raw_contexts, raw_lengths = super().fetch(1, obs_key=obs_key, action_key=action_key)
        memory_contexts = list(raw_contexts)
        valid_lengths = list(raw_lengths)

        to_update = []
        for env_idx in range(self.batch_size):
            total_steps = len(self._data[env_idx])
            valid_lengths[env_idx] = total_steps
            if total_steps <= 1:
                continue

            if self.summaries[env_idx] is None or total_steps != self.last_summary_step[env_idx]:
                history_steps = self._build_history(env_idx, obs_key, action_key)
                to_update.append((env_idx, history_steps))
            else:
                memory_contexts[env_idx] = self.summaries[env_idx]

        if to_update:
            max_workers = max(1, summary_concurrency or 1)

            def _summ_one(item):
                idx, history_steps = item
                try:
                    summary_text = simple_summarize(
                        history_steps,
                        api_key=api_key,
                        endpoint=endpoint,
                        env_type=env_type,
                        model=summary_model or "gpt-4o",
                        timeout_s=30,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Summary generation failed for env %s: %s", idx, exc)
                    summary_text = "\n".join(history_steps[-3:])
                return idx, summary_text

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_summ_one, item) for item in to_update]
                for future in as_completed(futures):
                    idx, summary_text = future.result()
                    self.summaries[idx] = summary_text
                    self.last_summary_step[idx] = len(self._data[idx])
                    memory_contexts[idx] = summary_text

        return memory_contexts, valid_lengths
    
    def _get_or_create_summary(
        self, 
        env_idx: int, 
        obs_key: str, 
        action_key: str,
        api_key: str,
        endpoint: str,
    ) -> str:
        """Get existing summary or create a new one."""
        # This method is retained for backward compatibility but now delegates to
        # _fetch_with_summary, which handles caching and batching.
        history_steps = self._build_history(env_idx, obs_key, action_key)
        summary_text = simple_summarize(
            history_steps,
            api_key=api_key,
            endpoint=endpoint,
            env_type=None,
            model="gpt-4o",
            timeout_s=30,
        )
        self.summaries[env_idx] = summary_text
        self.last_summary_step[env_idx] = len(self._data[env_idx])
        logger.debug("Updated summary for env %s, covering %s steps", env_idx, len(history_steps))
        return summary_text

    def _build_history(self, env_idx: int, obs_key: str, action_key: str) -> List[str]:
        """Build formatted history for the given environment index."""
        history_lines: List[str] = []
        for step_idx, record in enumerate(self._data[env_idx]):
            step_num = step_idx + 1
            action = record[action_key]
            obs = record[obs_key]
            history_lines.append(
                f"[Observation {step_num}: '{obs}', Action {step_num}: '{action}']"
            )
        return history_lines
