import requests
import logging
from typing import List, Tuple, Optional
from .memory import SimpleMemory

logger = logging.getLogger(__name__)


def simple_summarize(history_steps: List[str], api_key: str = None, endpoint: str = None) -> str:
    """
    Simple function to summarize history steps using LLM API.
    
    Args:
        history_steps: List of formatted history strings
        api_key: OpenAI API key
        endpoint: API endpoint URL
        
    Returns:
        Summarized history string
    """
    if not api_key or not endpoint:
        # Fallback: return truncated recent history
        return "\n".join(history_steps[-3:])  # Last 3 steps
    
    # Join all history into one text
    full_history = "\n".join(history_steps)
    
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
{full_history}"""

    try:
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        
        # Azure OpenAI format
        url = f"{endpoint}/openai/deployments/gpt-4o/chat/completions?api-version=2024-05-13"
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes task progress concisely."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            logger.debug(f"Summary generated: {len(content)} chars")
            return content.strip()
        else:
            logger.warning(f"API error {response.status_code}, using fallback")
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
        summary_threshold: Optional[int] = None,  # kept for backward compatibility, ignored
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
            
        Returns:
            Tuple of (memory_contexts, valid_lengths)
        """
        if not use_summary:
            # Use original SimpleMemory behavior
            return super().fetch(history_length, obs_key, action_key)
            
        return self._fetch_with_summary(
            obs_key, action_key, summary_api_key, summary_endpoint
        )
    
    def _fetch_with_summary(
        self, 
        obs_key: str, 
        action_key: str,
        api_key: str,
        endpoint: str
    ) -> Tuple[List[str], List[int]]:
        """Fetch history using summarization strategy."""
        memory_contexts, valid_lengths = [], []
        
        for env_idx in range(self.batch_size):
            total_steps = len(self._data[env_idx])
            
            if total_steps <= 1:
                # Only 1 step, use regular history (no need to summarize)
                ctx, vlen = super().fetch(1, obs_key=obs_key, action_key=action_key)
                memory_contexts.append(ctx[0])
                valid_lengths.append(vlen[0])
            else:
                # More than 1 step, use summarization
                summary_context = self._get_or_create_summary(
                    env_idx, obs_key, action_key, api_key, endpoint
                )
                memory_contexts.append(summary_context)
                valid_lengths.append(total_steps)  # Return total steps covered
                
        return memory_contexts, valid_lengths
    
    def _get_or_create_summary(
        self, 
        env_idx: int, 
        obs_key: str, 
        action_key: str,
        api_key: str,
        endpoint: str
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
            
            # Generate summary
            self.summaries[env_idx] = simple_summarize(all_history, api_key, endpoint)
            self.last_summary_step[env_idx] = total_steps
            
            logger.debug(f"Updated summary for env {env_idx}, covering {total_steps} steps")
            
        return self.summaries[env_idx]
