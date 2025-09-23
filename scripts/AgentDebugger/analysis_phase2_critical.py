#!/usr/bin/env python3
"""
Phase 2: Critical Failure Point Identification (Version 2)
Identifies the earliest critical error that led to task failure
No scoring, no agent feedback - just critical error identification
"""

import json
import os
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from error_definitions_loader import ErrorDefinitionsLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CriticalError:
    """Critical error identification result"""
    critical_step: int
    critical_module: str
    error_type: str
    root_cause: str
    evidence: str
    correction_guidance: str
    cascading_effects: List[Dict[str, Any]]
    follow_up_instruction: Optional[str] = None


FOLLOW_UP_PROMPT_TEMPLATE = (
    "Current result (critical step {critical_step}, {critical_module}::{error_type}):\n"
    "{trajectory_summary}\n\n"
    "Task: {task_description}\n"
    "Root cause: {root_cause}\n\n"
    "Why is this trajectory not finished the task?\n\n"
    "Feedback:"
)


class CriticalErrorAnalyzer:
    """Identifies critical failure points in trajectories"""
    
    def __init__(self, api_config: Dict[str, Any], capture_debug_data: bool = False):
        self.config = api_config
        # Build headers; include Authorization only if api_key present
        self.headers = {
            "Content-Type": "application/json",
        }
        api_key = (api_config.get('api_key') or '').strip()
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Load error definitions
        self.error_loader = ErrorDefinitionsLoader()

        # Module-specific error types (from loader)
        self.module_error_types = {
            'memory': [e for e in self.error_loader.get_valid_error_types('memory') if e != 'no_error'],
            'reflection': [e for e in self.error_loader.get_valid_error_types('reflection') if e != 'no_error'],
            'planning': [e for e in self.error_loader.get_valid_error_types('planning') if e != 'no_error'],
            'action': [e for e in self.error_loader.get_valid_error_types('action') if e != 'no_error'],
            'system': [e for e in self.error_loader.get_valid_error_types('system') if e != 'no_error'],
            'others': [e for e in self.error_loader.get_valid_error_types('others') if e != 'no_error']
        }

        # Optional debugging toggle so callers can capture raw prompts/responses when parsing fails
        self.capture_debug_data = capture_debug_data
        self._last_debug_payload: Optional[Dict[str, Any]] = None
    
    def load_phase1_results(self, file_path: str) -> Dict[str, Any]:
        """Load Phase 1 error detection results"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_original_trajectory(self, trajectory_path: str) -> Dict[str, Any]:
        """Load original trajectory file"""
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support both old format (chat_history) and new format (messages)
        chat_history = data.get('messages', data.get('chat_history', []))
        metadata = data.get('metadata', {})
        
        return {
            'chat_history': chat_history,
            'metadata': metadata,
            'task_success': metadata.get('success', metadata.get('won', False))
        }
    
    def _build_critical_error_prompt(
        self,
        step_analyses: List[Dict],
        task_description: str,
        chat_history: List[Dict],
        attempt_index: int = 1
    ) -> Tuple[str, str]:
        """Build prompt for critical error identification and return trajectory summary."""

        # Format step analyses with errors
        step_summaries = []
        for analysis in step_analyses:
            step_num = analysis['step']
            errors = analysis.get('errors', {}) or {}

            # Find agent content for this step
            agent_content = ""
            env_response = ""
            assistant_count = 0
            for i, msg in enumerate(chat_history):
                if msg.get('role') == 'assistant':
                    assistant_count += 1
                    if assistant_count == step_num:
                        agent_content = msg.get('content', '')
                        if i + 1 < len(chat_history) and chat_history[i + 1].get('role') == 'user':
                            env_response = chat_history[i + 1].get('content', '')[:600]
                        break

            step_summary = f"""
Step {step_num}:
Agent Output: {agent_content}
Environment Response: {env_response}

Errors Detected:"""

            for module, error_info in errors.items():
                if error_info and error_info.get('error_detected'):
                    step_summary += f"""
  - {module}: {error_info.get('error_type')}
    Evidence: {error_info.get('evidence', '')}
    Reasoning: {error_info.get('reasoning', '')}"""

            if not any(e.get('error_detected') for e in errors.values() if e):
                step_summary += "\n  No errors detected in this step"

            step_summaries.append(step_summary)

        all_steps = "\n".join(step_summaries)

        # Get complete error definitions
        error_reference = self.error_loader.format_for_phase2_prompt()

        prompt = f"""
You are an expert at identifying critical failure points in agent trajectories and providing high-priority, iterative follow-up instructions that MUST be followed across all subsequent steps.

TASK: {task_description}
TASK RESULT: FAILED

DEBUG ITERATION CONTEXT:
- Current debug attempt index: {attempt_index}
- Previously issued follow-up instructions:

STEP-BY-STEP ERROR ANALYSIS:
{all_steps}


ERROR DEFINITIONS:
{error_reference}

Your job is to identify the CRITICAL ERROR - the earliest and most important error that led to task failure, and produce an iterative follow-up instruction that will help avoid similar mistakes in future attempts.

CRITICAL ERROR IDENTIFICATION APPROACH:
You must take a HOLISTIC, GLOBAL perspective to identify the true root cause of failure. Do NOT rely on any predetermined severity weights or rankings.

ANALYSIS GUIDELINES:
1. Consider the ENTIRE trajectory from a global perspective - understand the task goal and how the agent's path diverged from success
2. Find the EARLIEST point where the agent made a decision or error that set it on an irreversible path to failure
3. Early exploration steps (steps 1-3) are often normal and should NOT be marked as critical unless there's a clear, fundamental error
4. An error is critical if:
   - It represents the ROOT CAUSE that made task success impossible
   - It caused a cascade of subsequent errors
   - The trajectory could have succeeded if THIS specific error had not occurred
   - **IMPORTANT: Correcting this specific error would fundamentally change the trajectory toward success**
5. Focus on causal chains - trace backwards from the failure to find the origin point
6. **IMPORTANT: Step 1 only has planning and action modules** - no memory or reflection is possible at step 1 since there's no history yet
   - Do NOT mark step 1 memory/reflection as critical errors
   - Early steps without memory/reflection modules are expected
7. Consider System and Others categories as potential critical errors:
   - System errors (step_limit, tool_execution_error, llm_limit, environment_error) may also be the true cause of failure
   - For example, if the agent was performing correctly but hit step_limit, that IS the critical error
   - Others category captures unusual failures not covered by standard error types


Identify the TRUE ROOT CAUSE that made the task unrecoverable.

REQUIRED OUTPUT FORMAT (JSON):
{{
    "critical_step": <step_number>,
    "critical_module": "<module_name: memory|reflection|planning|action|system|others>",
    "error_type": "<specific_error_type_from_definitions>",
    "root_cause": "Concise description of the fundamental problem",
    "evidence": "Specific quote or observation from trajectory supporting this identification",
    "correction_guidance": "Actionable advice for the agent to avoid the same mistake in that step",
    "cascading_effects": [{{ "step": <step_number>, "impact": "description" }}]
}}
"""
        return prompt, all_steps

    def _parse_critical_error(self, response: str) -> CriticalError:
        """Parse LLM response for critical error."""

        try:
            error_data = json.loads(response)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode critical error response: %s", exc)
            logger.error("Raw response (first 500 chars): %s", response[:500])
            raise ValueError(f"Invalid JSON format: {exc}") from exc

        if not isinstance(error_data, dict):
            logger.error("Critical error response is not a JSON object: %s", type(error_data).__name__)
            raise ValueError("Critical error response must be a JSON object")
        
        # Validate error type matches module
        module = error_data.get('critical_module', 'unknown')
        error_type = error_data.get('error_type', 'unknown')
        
        # Auto-correct if error type doesn't match module
        if module in self.module_error_types:
            if error_type not in self.module_error_types[module]:
                # Try to find the correct module for this error type
                for mod, types in self.module_error_types.items():
                    if error_type in types:
                        logger.warning(f"Correcting module from {module} to {mod} for error type {error_type}")
                        module = mod
                        break
        
        return CriticalError(
            critical_step=error_data.get('critical_step', 1),
            critical_module=module,
            error_type=error_type,
            root_cause=error_data.get('root_cause', 'No root cause identified'),
            evidence=error_data.get('evidence', 'No evidence provided'),
            correction_guidance=error_data.get('correction_guidance', 'No guidance provided'),
            cascading_effects=error_data.get('cascading_effects', []),
            follow_up_instruction=None,
        )

    async def _generate_follow_up_instruction(
        self,
        trajectory_summary: str,
        critical_error: CriticalError,
        task_description: str,
    ) -> str:
        snippet = trajectory_summary.strip()
        max_len = self.config.get('follow_up_summary_max_chars', 4000)
        if len(snippet) > max_len:
            snippet = snippet[: max_len - 3].rstrip() + "..."

        prompt = FOLLOW_UP_PROMPT_TEMPLATE.format(
            trajectory_summary=snippet,
            critical_step=critical_error.critical_step,
            critical_module=critical_error.critical_module,
            error_type=critical_error.error_type,
            root_cause=critical_error.root_cause,
            task_description=task_description,
        )

        try:
            response = await self.call_llm(prompt, response_format="text")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to generate follow-up instruction: {exc}")
            return ""

        return response.strip()

            
    
    async def call_llm(self, prompt: str, response_format: str = "json") -> str:
        """Call LLM API"""
        payload = {
            "model": self.config['model'],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at identifying critical failure points and drafting high-priority, iterative follow-up instructions to guide all subsequent steps."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.get('temperature', 0.0),
        }

        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        elif response_format and response_format != "text":
            payload["response_format"] = {"type": response_format}

        proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')

        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config.get('max_retries', 3)):
                try:
                    async with session.post(
                        self.config['base_url'],
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.get('timeout', 60)),
                        proxy=proxy if proxy else None
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                except Exception as e:
                    if attempt == self.config.get('max_retries', 3) - 1:
                        logger.error(f"API call failed: {e}")
                        raise
                    await asyncio.sleep(2 ** attempt)
        
        return ""
    
    async def identify_critical_error(self, phase1_results: Dict, trajectory_data: Dict, attempt_index: int = 1) -> Dict[str, Any]:
        """Identify critical error from phase1 results and trajectory (for API use)"""
        task_description = phase1_results.get('task_description', 'Unknown task')
        step_analyses = phase1_results.get('step_analyses', [])

        # Build analysis prompt
        chat_history = trajectory_data.get('messages') or trajectory_data.get('chat_history') or []
        prompt, trajectory_summary = self._build_critical_error_prompt(
            step_analyses,
            task_description,
            chat_history,
            attempt_index=attempt_index
        )

        # Call LLM
        response = await self.call_llm(prompt)

        # Parse response
        critical_error = self._parse_critical_error(response)

        if critical_error is not None:
            follow_up_instruction = await self._generate_follow_up_instruction(
                trajectory_summary,
                critical_error,
                task_description,
            )
            if follow_up_instruction:
                critical_error.follow_up_instruction = follow_up_instruction

        result = {
            'critical_error': asdict(critical_error),
            'task_success': False,
            'environment': phase1_results.get('environment', 'unknown')
        }

        if self.capture_debug_data:
            debug_payload = {
                'prompt': prompt,
                'raw_response': response,
                'parsed_critical_error': result['critical_error'],
            }
            self._last_debug_payload = debug_payload
            result['debug_payload'] = debug_payload
        else:
            self._last_debug_payload = None

        return result

    async def process_trajectory(
        self,
        phase1_file: str,
        original_trajectory_file: str,
        output_dir: str,
        *,
        attempt_index: int = 1,
    ) -> Dict[str, Any]:
        """Process a trajectory to identify the critical error for the current attempt."""
        
        try:
            # Load data
            phase1_results = self.load_phase1_results(phase1_file)
            original_trajectory = self.load_original_trajectory(original_trajectory_file)
            
            # Identify critical error
            critical_error = await self.identify_critical_error(
                phase1_results,
                original_trajectory,
                attempt_index=attempt_index,
            )
            
            # Prepare result
            if critical_error:
                result = {
                    'task_id': phase1_results['task_id'],
                    'task_description': phase1_results['task_description'],
                    'task_success': phase1_results['task_success'],
                    'environment': phase1_results['environment'],
                    'critical_error': asdict(critical_error),
                    'error_summary': {
                        'total_steps': phase1_results['total_steps'],
                        'critical_at': f"Step {critical_error.critical_step} - {critical_error.critical_module}:{critical_error.error_type}",
                    }
                }
            else:
                # Task succeeded
                result = {
                    'task_id': phase1_results['task_id'],
                    'task_description': phase1_results['task_description'],
                    'task_success': phase1_results['task_success'],
                    'environment': phase1_results['environment'],
                    'critical_error': None,
                    'error_summary': {
                        'total_steps': phase1_results['total_steps'],
                        'message': 'Task succeeded - no critical error'
                    }
                }
            
            # Save result
            output_file = Path(output_dir) / f"{Path(phase1_file).stem}_critical_error.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            if critical_error:
                logger.info(f"Critical error identified: Step {critical_error.critical_step} - {critical_error.error_type}")
            else:
                logger.info("No critical error (task succeeded)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing trajectory: {e}")
            import traceback
            traceback.print_exc()
            return None
