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
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from error_definitions_loader_v3 import ErrorDefinitionsLoader

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
    confidence: float


class CriticalErrorAnalyzer:
    """Identifies critical failure points in trajectories"""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.config = api_config
        self.headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
        
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
    
    async def identify_critical_error(
        self,
        phase1_results: Dict[str, Any],
        original_trajectory: Dict[str, Any],
        retry_count: int = 0
    ) -> Optional[CriticalError]:
        """Identify the critical error that led to failure"""

        # Skip successful tasks
        if phase1_results['task_success']:
            logger.info("Task succeeded - no critical error to identify")
            return None

        # Prepare analysis data
        step_analyses = phase1_results['step_analyses']
        task_description = phase1_results['task_description']
        # Support both 'messages' and 'chat_history' field names
        chat_history = original_trajectory.get('messages', original_trajectory.get('chat_history', []))

        # Build critical error identification prompt
        prompt = self._build_critical_error_prompt(
            step_analyses,
            task_description,
            chat_history,
            retry_count > 0  # Add flag to indicate this is a retry
        )

        # Call LLM for critical error identification
        response = await self.call_llm(prompt)

        # Parse the result
        critical_error = self._parse_critical_error(response)

        # Validate the result
        if critical_error and critical_error.critical_step == 1 and critical_error.critical_module in ['memory', 'reflection']:
            logger.warning(f"Invalid analysis detected: Step 1 with {critical_error.critical_module} error")

            # Retry up to 2 times
            if retry_count < 2:
                logger.info(f"Retrying analysis (attempt {retry_count + 2}/3)...")
                return await self.identify_critical_error(
                    phase1_results,
                    original_trajectory,
                    retry_count + 1
                )
            else:
                logger.error("Failed after 3 attempts. Returning invalid result.")
                # Return a special error indicating analysis failure
                return CriticalError(
                    critical_step=1,
                    critical_module='unknown',
                    error_type='analysis_failure',
                    root_cause="LLM repeatedly identified invalid step 1 memory/reflection error after 3 attempts",
                    evidence=f"Failed analysis: {critical_error.critical_module}/{critical_error.error_type}",
                    correction_guidance="Manual review required",
                    cascading_effects=[],
                    confidence=0.0
                )

        return critical_error
    
    def _build_critical_error_prompt(
        self,
        step_analyses: List[Dict],
        task_description: str,
        chat_history: List[Dict],
        is_retry: bool = False
    ) -> str:
        """Build prompt for critical error identification"""
        
        # Format step analyses with errors
        step_summaries = []
        for analysis in step_analyses:
            step_num = analysis['step']
            errors = analysis.get('errors', {})
            
            # Find agent content for this step
            agent_content = ""
            env_response = ""
            assistant_count = 0
            for i, msg in enumerate(chat_history):
                if msg['role'] == 'assistant':
                    assistant_count += 1
                    if assistant_count == step_num:
                        agent_content = msg['content']  # Full content, no truncation
                        if i + 1 < len(chat_history) and chat_history[i + 1]['role'] == 'user':
                            env_response = chat_history[i + 1]['content'][:500]  # Increase env response limit too
                        break
            
            step_summary = f"""
Step {step_num}:
Agent Output: {agent_content}
Environment Response: {env_response}

Errors Detected:"""
            
            for module, error_info in errors.items():
                # Skip memory/reflection errors for step 1 (no history to remember or reflect on)
                if step_num == 1 and module in ['memory', 'reflection']:
                    continue

                if error_info and error_info.get('error_detected'):
                    step_summary += f"""
  - {module}: {error_info['error_type']}
    Evidence: {error_info['evidence']}
    Reasoning: {error_info['reasoning']}"""
            
            if not any(e.get('error_detected') for e in errors.values() if e):
                step_summary += "\n  No errors detected in this step"
            
            step_summaries.append(step_summary)
        
        all_steps = "\n".join(step_summaries)
        
        # Get complete error definitions
        error_reference = self.error_loader.format_for_phase2_prompt()

        # Add retry warning if this is a retry
        retry_warning = ""
        if is_retry:
            retry_warning = """
⚠️ IMPORTANT WARNING: Your previous analysis was INVALID!
You incorrectly identified a memory or reflection error at Step 1.
Step 1 CANNOT have memory or reflection errors because:
- Memory requires previous steps to remember (Step 1 has no history)
- Reflection requires previous actions to reflect on (Step 1 has no prior actions)

ONLY planning and action modules are possible at Step 1.
Please re-analyze carefully!
"""

        prompt = f"""
You are an expert at identifying critical failure points in agent trajectories.
{retry_warning}

TASK: {task_description}
TASK RESULT: FAILED

STEP-BY-STEP ERROR ANALYSIS:
{all_steps}

{error_reference}

Your job is to identify the CRITICAL ERROR - the earliest and most important error that led to task failure.

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
   - Do NOT ignore these categories
   
KEY DECISION PRINCIPLE:
Think globally: "What was the FIRST decision or error that doomed this trajectory to failure?"
NOT: "Which error type seems most severe based on a predefined scale?"

The critical error is the one where, if we could go back in time and fix ONLY that error, the entire trajectory would likely succeed.

REQUIRED OUTPUT FORMAT (JSON):
{{
    "critical_step": <step_number>,
    "critical_module": "<module_name>",
    "error_type": "<specific_error_type_from_definitions_above>",
    "root_cause": "Detailed explanation of why this specific error at this step caused the task to fail",
    "evidence": "Specific quote or observation from the trajectory supporting this identification",
    "correction_guidance": "Specific guidance on what the agent should have done differently to avoid this error and succeed",
    "cascading_effects": [
        {{
            "step": <later_step>,
            "effect": "How the critical error affected this later step"
        }}
    ],
    "confidence": 0.0-1.0
}}

IMPORTANT: 
- Error types MUST be selected from the definitions provided above
- The error_type must match one of the defined types for that module
- Valid modules include: memory, reflection, planning, action, system, others
- System errors (step_limit, tool_execution_error, llm_limit, environment_error) are VALID critical errors
- Others category is for unusual failures not covered by standard types
- Focus on the error that, if corrected, would have the highest impact on task success

Identify the TRUE ROOT CAUSE that made the task unrecoverable.
"""
        return prompt
    
    def _parse_critical_error(self, response: str) -> CriticalError:
        """Parse LLM response for critical error"""

        try:
            # Try to extract JSON from response - handle nested braces
            # First try to find a complete JSON object
            json_start = response.find('{')
            if json_start == -1:
                raise ValueError("No JSON found in response")

            # Count braces to find the complete JSON object
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if brace_count != 0:
                # Fallback to regex if brace counting fails
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("Malformed JSON in response")
            else:
                json_str = response[json_start:json_end]

            # Clean up common JSON issues
            # Remove any trailing commas before closing braces/brackets
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            # Try multiple approaches to fix JSON
            error_data = None

            # Approach 1: Try to parse as-is
            try:
                error_data = json.loads(json_str)
            except json.JSONDecodeError as e1:
                # Approach 2: Fix common quote issues
                try:
                    # Fix patterns like "agent"s" -> "agent's"
                    fixed_str = json_str
                    # Find and fix unescaped quotes in the middle of strings
                    # Look for patterns like: "...text"s text..." and fix to "...text's text..."
                    fixed_str = re.sub(r'([a-zA-Z])"s\s', r"\1's ", fixed_str)
                    fixed_str = re.sub(r'([a-zA-Z])"t\s', r"\1't ", fixed_str)
                    fixed_str = re.sub(r'([a-zA-Z])"ll\s', r"\1'll ", fixed_str)
                    fixed_str = re.sub(r'([a-zA-Z])"ve\s', r"\1've ", fixed_str)
                    fixed_str = re.sub(r'([a-zA-Z])"re\s', r"\1're ", fixed_str)
                    fixed_str = re.sub(r'([a-zA-Z])"d\s', r"\1'd ", fixed_str)

                    error_data = json.loads(fixed_str)
                except json.JSONDecodeError as e2:
                    # Approach 3: Use ast.literal_eval as fallback
                    try:
                        import ast
                        # Convert to Python dict format
                        py_str = json_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                        error_data = ast.literal_eval(py_str)
                    except (ValueError, SyntaxError) as e3:
                        logger.error(f"All JSON parse attempts failed")
                        logger.error(f"Error 1 (direct): {e1}")
                        logger.error(f"Error 2 (fixed quotes): {e2}")
                        logger.error(f"Error 3 (ast.literal_eval): {e3}")
                        logger.error(f"Attempted to parse: {json_str[:500]}...")
                        raise ValueError(f"Invalid JSON format: {e1}")

            if not error_data:
                raise ValueError("Failed to parse JSON after all attempts")
            
            # Validate error type matches module
            module = error_data.get('critical_module', 'unknown')
            error_type = error_data.get('error_type', 'unknown')
            critical_step = error_data.get('critical_step', 1)

            # Note: Step 1 validation is now handled in identify_critical_error with retry mechanism
            # We don't modify the results here, just log if there's an issue
            if critical_step == 1 and module in ['memory', 'reflection']:
                logger.warning(f"Parsed result has invalid Step 1 {module} error - will be handled by retry mechanism")

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
                critical_step=critical_step,
                critical_module=module,
                error_type=error_type,
                root_cause=error_data.get('root_cause', 'No root cause identified'),
                evidence=error_data.get('evidence', 'No evidence provided'),
                correction_guidance=error_data.get('correction_guidance', 'No guidance provided'),
                cascading_effects=error_data.get('cascading_effects', []),
                confidence=float(error_data.get('confidence', 0.5))
            )
            
        except Exception as e:
            logger.error(f"Failed to parse critical error: {e}")

            # Try to extract key information from the raw response using regex
            step_match = re.search(r'(?:critical_step|step)["\s:]+(\d+)', response, re.IGNORECASE)
            module_match = re.search(r'(?:critical_module|module)["\s:]+["\']*([a-z_]+)', response, re.IGNORECASE)
            error_match = re.search(r'(?:error_type|type)["\s:]+["\']*([a-z_]+)', response, re.IGNORECASE)

            if step_match and module_match and error_match:
                logger.info("Extracted partial information from malformed response")
                critical_step = int(step_match.group(1))
                module = module_match.group(1)
                error_type = error_match.group(1)

                # Note: For regex extraction, we can't retry, so we mark as invalid
                if critical_step == 1 and module in ['memory', 'reflection']:
                    logger.error(f"INVALID ANALYSIS from regex extraction: Step 1 cannot have {module} errors!")
                    # For regex extraction, we have limited info, so mark as unknown
                    module = 'unknown'
                    error_type = 'analysis_error'

                return CriticalError(
                    critical_step=critical_step,
                    critical_module=module,
                    error_type=error_type,
                    root_cause="Extracted from malformed JSON response",
                    evidence="Parse error - partial extraction",
                    correction_guidance="Review original trajectory for details",
                    cascading_effects=[],
                    confidence=0.3
                )
            else:
                return CriticalError(
                    critical_step=1,
                    critical_module='unknown',
                    error_type='parse_error',
                    root_cause=f"Parse error: {str(e)}",
                    evidence="Failed to parse analysis",
                    correction_guidance="Unable to provide guidance due to parse error",
                    cascading_effects=[],
                    confidence=0.0
                )
    
    async def call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        payload = {
            "model": self.config['model'],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at identifying critical failure points in agent trajectories."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.get('temperature', 0.0)
        }
        
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
    
    async def process_trajectory(
        self,
        phase1_file: str,
        original_trajectory_file: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Process a trajectory to identify critical error"""
        
        try:
            # Load data
            phase1_results = self.load_phase1_results(phase1_file)
            original_trajectory = self.load_original_trajectory(original_trajectory_file)
            
            # Identify critical error
            critical_error = await self.identify_critical_error(
                phase1_results,
                original_trajectory
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
                        'confidence': critical_error.confidence
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