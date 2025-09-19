#!/usr/bin/env python3
"""
Unified rollout script for AlfWorld, GAIA, and WebShop environments.
Provides a single interface for running rollouts across all three environments.
"""

import os
import time
import json
import logging
import argparse
import shutil
from types import SimpleNamespace
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import queue
import numpy as np
import random
import hashlib
import sys
from openmanus_rl.environments.env_manager import *
from scripts.rollout.baselines import run_best_of_n, run_tree_search, SearchParams
from openai import OpenAI


class TeeOutput:
    """Class to duplicate stdout/stderr to both terminal and file."""
    def __init__(self, file_path, stream):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.stream = stream
        self.original = getattr(sys, stream)
        # Copy attributes from original stream
        self.encoding = getattr(self.original, 'encoding', 'utf-8')

    def write(self, data):
        self.original.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.original.flush()
        self.file.flush()

    def fileno(self):
        """Return file descriptor of the original stream."""
        return self.original.fileno()

    def isatty(self):
        """Return whether the original stream is a tty."""
        return self.original.isatty()

    def __getattr__(self, name):
        """Delegate any other attributes to the original stream."""
        return getattr(self.original, name)

    def __enter__(self):
        setattr(sys, self.stream, self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(sys, self.stream, self.original)
        self.file.close()
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor as AsyncThreadPoolExecutor
from dataclasses import asdict, is_dataclass

try:
    from scripts.AgentDebugger.api_interface import AgentErrorDetectorAPI
    ADVANCED_DEBUGGER_AVAILABLE = True
except ImportError:
    AgentErrorDetectorAPI = None  # type: ignore[assignment]
    ADVANCED_DEBUGGER_AVAILABLE = False


def _json_safe_copy(data: Any) -> Any:
    """Return a JSON-serializable copy of the provided data."""
    try:
        return json.loads(json.dumps(data, default=str))
    except Exception:
        return str(data)

try:
    import dotenv
    dotenv.load_dotenv()
except Exception:
    pass


class UnifiedAgent:
    """Unified agent that can work with all environments"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.4, 
                 base_url: str | None = None, env_type: str = "alfworld",
                 use_together: bool = False):
        self.model_name = model_name
        self.temperature = temperature
        self.env_type = env_type
        
        # Select client based on explicit Together flag
        self.is_together = bool(use_together)
        if self.is_together:
            # Use OpenAI-compatible client pointed at Together endpoint
            self.client = OpenAI(
                api_key=os.environ.get("TOGETHER_API_KEY", ""),
                base_url="https://api.together.xyz/v1",
            )
        elif base_url:
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
                base_url=base_url,
            )
        else:
            self.client = OpenAI(
                api_key=os.environ.get('OPENAI_API_KEY'),
            )
        
        # Set environment-specific system prompts
        self.system_prompts = {
            "webshop": (
                "You are an expert web shopping agent. Respond strictly as "
                "<think>...</think><action>...</action>. The <action> must be a single "
                "admissible action exactly from the provided list, or a search[query]."
            ),
            "gaia": None,  # GAIA uses prompt templates in the environment
            "alfworld": None,  # AlfWorld uses prompt templates in the environment
        }
        
    def get_action_from_llm(self, obs: str, log_timing: bool = True) -> str:
        """Get action from LLM for a single observation"""
        if log_timing:
            llm_start = time.time()
            thread_id = threading.get_ident()
        
        messages = []
        
        # Add system prompt if available for this environment
        system_prompt = self.system_prompts.get(self.env_type)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": obs})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            n=1,
        )
        
        if log_timing:
            llm_time = time.time() - llm_start
            logging.debug(f"[LLM] Thread {thread_id} LLM call took {llm_time:.3f}s")
        
        return response.choices[0].message.content.strip()
    
    def get_action_from_llm_with_shared_pool(self, obs: str, shared_executor, log_timing: bool = True):
        """Get action from LLM using a shared thread pool executor for better global concurrency"""
        def _call_llm():
            return self.get_action_from_llm(obs, log_timing)
        
        # Submit to shared executor and return future
        return shared_executor.submit(_call_llm)
    
    def get_actions_batch(self, prompts: List[str], concurrency: int = 4, 
                         retries: int = 3, backoff: float = 0.5) -> List[str]:
        """Get actions for multiple observations in parallel"""
        actions = [None] * len(prompts)
        
        def _one(idx_prompt):
            idx, prompt = idx_prompt
            delay = backoff
            for attempt in range(retries):
                try:
                    act = self.get_action_from_llm(prompt)
                    return idx, act
                except Exception as e:
                    if attempt == retries - 1:
                        # Return a default action based on environment
                        default_actions = {
                            "webshop": "<think>error</think><action>search[product]</action>",
                            "gaia": "None",
                            "alfworld": "None"
                        }
                        return idx, default_actions.get(self.env_type, "None")
                    time.sleep(delay)
                    delay *= 2
        
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures = [ex.submit(_one, (i, p)) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                i, act = fut.result()
                actions[i] = act
        
        return actions


class LLMDebugger:
    """Enhanced LLM-based debugger with comprehensive error type awareness"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3,
                 base_url: str | None = None, api_key: str | None = None,
                 use_together: bool = False):
        """Initialize the LLM-based debugger client.

        Args:
            model_name: Debugger model name.
            temperature: Sampling temperature for the debugger model.
            base_url: Optional OpenAI-compatible base URL (e.g., vLLM endpoint).
            api_key: Optional API key for the debugger client. If None, falls back to
                OPENAI_API_KEY environment variable. For local vLLM without auth, this may be empty.
        """
        self.model_name = model_name
        self.temperature = temperature

        # Initialize OpenAI-compatible client (OpenAI/vLLM/Together)
        if use_together:
            self.client = OpenAI(
                api_key=os.environ.get("TOGETHER_API_KEY", ""),
                base_url="https://api.together.xyz/v1",
            )
        else:
            key = api_key if api_key is not None else os.environ.get('OPENAI_API_KEY', '')
            if base_url:
                self.client = OpenAI(
                    api_key=key,
                    base_url=base_url,
                )
            else:
                self.client = OpenAI(
                    api_key=key,
                )
        
        # Enhanced error type definitions aligned with AgentDebugger
        self.error_definitions = self._load_error_definitions()
        self.module_error_types = {
            'memory': ['over_simplification', 'memory_retrieval_failure', 'hallucination'],
            'reflection': ['progress_misjudge', 'outcome_misinterpretation', 'causal_misattribution', 'hallucination'],
            'planning': ['constraint_ignorance', 'impossible_action', 'inefficient_plan'],
            'action': ['misalignment', 'invalid_action', 'format_error', 'parameter_error'],
            'system': ['step_limit', 'tool_execution_error', 'llm_limit', 'environment_error'],
            'others': ['others']
        }

    def _log_llm_call(self, log_path: Optional[str], prompt: str, response_text: str) -> None:
        """Append a single LLM interaction record to a JSONL file."""
        if not log_path:
            return

        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            record = {"input": prompt, "output": response_text}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logging.debug(f"Failed to log LLM call to {log_path}: {exc}")
    
    def _load_error_definitions(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Load comprehensive error type definitions from AgentDebugger"""
        return {
            'memory': {
                'over_simplification': {
                    'definition': 'Agent oversimplifies complex information from previous N steps, ignoring details and key factors, leading to decisions based on partial or oversimplified summaries',
                    'example': 'Agent simplifies multiple product selection criteria to just "item found", ignoring price, features, and inventory factors'
                },
                'memory_retrieval_failure': {
                    'definition': 'Relevant information exists in agent memory but fails to be retrieved when needed',
                    'example': 'Agent explored kitchen and observed "knife on countertop" but later fails to recall knife location when needed'
                },
                'hallucination': {
                    'definition': 'Agent "recalls" events that never happened, object states never observed, or actions never executed',
                    'example': 'Agent claims "I remember seeing a knife in the first drawer" when "open drawer 1" was never successfully executed'
                }
            },
            'reflection': {
                'progress_misjudge': {
                    'definition': 'Agent incorrectly evaluates progress toward completing the overall task goal',
                    'example': 'Agent enters kitchen without finding cup yet but reflects "progress is smooth, task nearly complete"'
                },
                'outcome_misinterpretation': {
                    'definition': 'Agent correctly executes an action but incorrectly interprets the direct result or environment feedback',
                    'example': 'After Put(Apple, Microwave) fails with "Nothing happens", agent reflects "Successfully placed apple in microwave"'
                },
                'causal_misattribution': {
                    'definition': 'Agent correctly identifies a failure phenomenon but attributes it to the wrong cause',
                    'example': 'Take(Key) fails because key is in locked safe, but agent attributes to "mechanical arm malfunction"'
                },
                'hallucination': {
                    'definition': 'Agent believes it performed actions that never actually occurred',
                    'example': 'Agent interprets plan generated in step 1 as operations already completed'
                }
            },
            'planning': {
                'constraint_ignorance': {
                    'definition': 'Planning ignores task constraints like resource limits (time, budget, space)',
                    'example': 'Budget is $40 but selects $55 product, not considering time/interaction limits'
                },
                'impossible_action': {
                    'definition': 'Agent plans to execute an action that is fundamentally impossible under current conditions',
                    'example': 'Plans Slice(Desk) with knife, or Put(Mug, Sink) when inventory is empty'
                },
                'inefficient_plan': {
                    'definition': 'Agent creates plan that can theoretically complete task but is extremely inefficient',
                    'example': 'Takes circuitous route through multiple rooms instead of direct path to destination'
                }
            },
            'action': {
                'misalignment': {
                    'definition': 'Generated specific action completely contradicts the intention stated in current plan',
                    'example': 'Plans to "slice the apple" but executes GoTo(Bedroom 1) instead'
                },
                'invalid_action': {
                    'definition': 'Uses action that does not exist in the available action list',
                    'example': 'Attempts to use undefined action not in environment action space'
                },
                'format_error': {
                    'definition': 'Generated action has invalid format causing parse failure',
                    'example': 'click"product" instead of correct format click["product"]'
                },
                'parameter_error': {
                    'definition': 'Action parameters are unreasonable or incorrectly chosen',
                    'example': 'search[query repeated 100 times] or using invalid object names'
                }
            },
            'system': {
                'step_limit': {
                    'definition': 'Agent executes reasonably but fails due to reaching system maximum step limit',
                    'example': 'First item found and placed, searching for second item when 30-step limit reached'
                },
                'tool_execution_error': {
                    'definition': 'External tool or API called by agent returns error or exhibits unpredictable behavior',
                    'example': 'Object recognition tool misidentifies apple as tomato, causing subsequent failures'
                },
                'llm_limit': {
                    'definition': 'Agent response limitations cause failure',
                    'example': 'API call timeout, max token exceeded, rate limiting issues'
                },
                'environment_error': {
                    'definition': 'Simulation environment itself has bugs or unexpected behavior',
                    'example': 'Agent executes valid Open(Drawer) but environment crashes or object disappears'
                }
            },
            'others': {
                'others': {
                    'definition': 'All remaining problems not previously defined or discussed',
                    'example': 'Issues not covered by any of the above error categories'
                }
            }
        }
    
    def _build_error_reference(self) -> str:
        """Build comprehensive error reference for prompt"""
        reference = "COMPLETE ERROR TYPE REFERENCE WITH DEFINITIONS:\n\n"
        
        module_order = ['memory', 'reflection', 'planning', 'action', 'system', 'others']
        
        for module in module_order:
            reference += f"━━━ {module.upper()} MODULE ERRORS ━━━\n"
            module_defs = self.error_definitions.get(module, {})
            
            for error_type, details in module_defs.items():
                reference += f"• {error_type}: {details['definition']}\n"
                if details.get('example'):
                    reference += f"  Example: {details['example']}\n"
            
            reference += "\n"
        
        return reference
    
    def _validate_and_enhance_analysis(self, analysis: Dict, trajectory: List[Dict]) -> Dict:
        """Validate and enhance analysis results"""
        # Ensure required fields exist
        analysis.setdefault("failure_step", len(trajectory) - 1)
        analysis.setdefault("critical_module", "others")
        analysis.setdefault("failure_type", "others")
        analysis.setdefault("reason", "Unknown error")
        analysis.setdefault("suggestion", "Try a different approach")
        analysis.setdefault("critical_step", 0)
        analysis.setdefault("evidence", "No evidence provided")
        analysis.setdefault("confidence", 0.5)
        analysis.setdefault("root_cause", analysis.get("reason", "Unknown error"))
        
        # Validate module and error type consistency
        critical_module = analysis.get("critical_module", "others")
        failure_type = analysis.get("failure_type", "others")
        
        if critical_module in self.module_error_types:
            if failure_type not in self.module_error_types[critical_module]:
                # Try to find correct module for this error type
                for module, types in self.module_error_types.items():
                    if failure_type in types:
                        logging.warning(f"Correcting module from {critical_module} to {module} for error type {failure_type}")
                        analysis["critical_module"] = module
                        break
                else:
                    # If error type not found anywhere, default to 'others'
                    logging.warning(f"Unknown error type {failure_type}, defaulting to 'others'")
                    analysis["critical_module"] = "others"
                    analysis["failure_type"] = "others"
        
        # Ensure failure_step is within bounds
        max_step = len(trajectory) - 1
        if analysis["failure_step"] > max_step:
            analysis["failure_step"] = max_step
        if analysis["failure_step"] < 0:
            analysis["failure_step"] = 0
            
        # Ensure critical_step is before failure_step
        if analysis["critical_step"] >= analysis["failure_step"]:
            analysis["critical_step"] = max(0, analysis["failure_step"] - 1)
        
        # Add backwards compatibility for old format
        formatted_failure_type = f"{analysis['critical_module']}::{analysis['failure_type']}"
        analysis["raw_critical_error"] = {
            "critical_step": analysis["failure_step"] + 1,  # Convert to 1-based for consistency with advanced debugger
            "critical_module": analysis["critical_module"],
            "error_type": analysis["failure_type"],
            "root_cause": analysis["root_cause"],
            "evidence": analysis["evidence"],
            "correction_guidance": analysis["suggestion"],
            "confidence": analysis["confidence"],
            "cascading_effects": []
        }
        
        return analysis
    
    def analyze_trajectory(
        self,
        trajectory: List[Dict],
        env_type: str,
        chat_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        previous_instructions: Optional[List[str]] = None,
        generate_follow_up: bool = False,
        log_path: Optional[str] = None,
    ) -> Dict:
        """
        Enhanced trajectory analysis with comprehensive error type awareness
        
        Args:
            trajectory: List of trajectory steps
            env_type: Type of environment (alfworld, gaia, webshop)
            chat_history: Optional chat history for better context
            metadata: Optional metadata about the task
            previous_instructions: List of previous follow-up instructions from earlier failures
            generate_follow_up: Whether to generate follow-up instructions for future steps
        
        Returns:
            Analysis dict with detailed error classification and guidance
        """
        
        # Format trajectory for LLM analysis
        trajectory_text = self._format_trajectory(trajectory)
        
        # Get task description from metadata if available
        task_description = "Unknown task"
        if metadata:
            task_description = metadata.get("task", metadata.get("initial_observation", "Unknown task"))
        
        # Create environment-specific context
        env_contexts = {
            "alfworld": """The agent is completing household tasks in a text-based simulation. Common tasks include:
- pick_and_place: Pick up an object and place it somewhere
- pick_two_obj_and_place: Pick up two objects and place them  
- look_at_obj_in_light: Examine an object under a light source
- pick_heat_then_place_in_recep: Heat an object and place it in a receptacle
- pick_cool_then_place_in_recep: Cool an object and place it in a receptacle
- pick_clean_then_place_in_recep: Clean an object and place it in a receptacle

Key mechanics: Agent navigates rooms, interacts with objects, and uses appliances to complete multi-step tasks.""",
            
            "gaia": """The agent is solving complex reasoning tasks using various tools including:
- google_search: Search for information online
- wikipedia_knowledge_searcher: Search Wikipedia for knowledge  
- python_code_generator: Generate and execute Python code
- Other specialized tools for information gathering and processing

Tasks require multi-step reasoning, information synthesis, and tool orchestration.""",
            
            "webshop": """The agent is shopping for products on a web interface. The agent must:
- Search for products matching specific requirements (price, features, ratings)
- Navigate through product listings and categories
- Select products that meet all given criteria
- Complete the purchase process within budget constraints

Success requires understanding product specifications and constraint satisfaction."""
        }
        
        env_context = env_contexts.get(env_type, "The agent is solving a task in an interactive environment.")
        
        # Build comprehensive error reference
        error_reference = self._build_error_reference()
        
        # Include previous instructions context if available
        previous_instructions_context = ""
        if previous_instructions:
            previous_instructions_context = f"""
PREVIOUS DEBUGGER INSTRUCTIONS FROM EARLIER FAILURES:
{chr(10).join(f"- {instruction}" for instruction in previous_instructions)}

These instructions were generated from analyzing previous failures. Consider how the current failure might relate to or differ from these earlier patterns.
"""
        
        # Configure output format based on whether follow-up instruction is needed
        if generate_follow_up:
            follow_up_instruction = """
FOLLOW-UP INSTRUCTION GENERATION:
In addition to the regular analysis, generate a "follow_up_instruction" - a concise, actionable guidance that should be applied to ALL FUTURE STEPS to prevent similar errors. This instruction should:
1. Be general enough to apply across multiple steps
2. Address the root cause category (not just the specific instance)
3. Build upon previous instructions to form cumulative guidance
4. Be clear and actionable for the agent to follow
5. Focus on preventing common error patterns that could occur in future attempts

The follow_up_instruction should provide proactive guidance to avoid similar failure modes. Please be succint and concise.
"""
            # JSON format with follow_up_instruction field
            json_format = """{
    "failure_step": <step_number>,
    "critical_module": "<module_name: memory|reflection|planning|action|system|others>", 
    "failure_type": "<specific_error_type_from_definitions>",
    "reason": "Detailed explanation of why this specific error at this step caused task failure",
    "suggestion": "Specific guidance on what the agent should have done differently",
    "critical_step": <last_step_that_was_definitely_correct>,
    "evidence": "Specific quote or observation from trajectory supporting this identification",
    "confidence": <0.0-1.0>,
    "root_cause": "Concise description of the fundamental problem",
    "follow_up_instruction": "General guidance for future steps to prevent similar errors"
}"""
            additional_requirement = "- Generate a follow_up_instruction that provides proactive guidance for preventing similar errors in future steps"
        else:
            follow_up_instruction = ""
            # JSON format without follow_up_instruction field
            json_format = """{
    "failure_step": <step_number>,
    "critical_module": "<module_name: memory|reflection|planning|action|system|others>", 
    "failure_type": "<specific_error_type_from_definitions>",
    "reason": "Detailed explanation of why this specific error at this step caused task failure",
    "suggestion": "Specific guidance on what the agent should have done differently",
    "critical_step": <last_step_that_was_definitely_correct>,
    "evidence": "Specific quote or observation from trajectory supporting this identification",
    "confidence": <0.0-1.0>,
    "root_cause": "Concise description of the fundamental problem"
}"""
            additional_requirement = ""
        
        prompt = f"""You are an expert at identifying critical failure points in agent trajectories. Analyze this failed trajectory using comprehensive error type classification.

TASK: {task_description}
ENVIRONMENT: {env_type}
TASK RESULT: FAILED

ENVIRONMENT CONTEXT:
{env_context}
{previous_instructions_context}
TRAJECTORY ANALYSIS:
{trajectory_text}

{error_reference}

CRITICAL ERROR IDENTIFICATION:

Your task is to identify the CRITICAL ERROR - the earliest and most important error that led to task failure.

ANALYSIS GUIDELINES:
1. Take a HOLISTIC, GLOBAL perspective - understand the task goal and how the agent's path diverged from success
2. Find the EARLIEST point where the agent made a decision/error that set it on an irreversible path to failure  
3. Early exploration steps (1-3) are often normal - don't mark as critical unless there's a clear fundamental error
4. An error is critical if:
   - It represents the ROOT CAUSE that made task success impossible
   - It caused a cascade of subsequent errors
   - The trajectory could have succeeded if THIS specific error had not occurred
   - Correcting this specific error would fundamentally change the trajectory toward success

5. Consider all error modules:
   - **Memory**: Issues with information storage, retrieval, or recall
   - **Reflection**: Problems evaluating progress, outcomes, or causal relationships  
   - **Planning**: Flawed task decomposition, constraint violations, impossible actions
   - **Action**: Misaligned actions, format errors, invalid parameters
   - **System**: Step limits, tool errors, LLM failures, environment issues
   - **Others**: Unusual failures not covered by standard categories

KEY PRINCIPLE: "What was the FIRST decision or error that doomed this trajectory to failure?"
{follow_up_instruction}
REQUIRED OUTPUT FORMAT (JSON):
{json_format}

IMPORTANT: 
- Error types MUST match the definitions provided above
- Focus on the error that, if corrected, would have the highest impact on task success
- The critical_module and failure_type must be consistent with the error taxonomy
{additional_requirement}

Identify the TRUE ROOT CAUSE that made the task unrecoverable."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying critical failure points in agent trajectories using comprehensive error type classification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            self._log_llm_call(log_path, prompt, response_text)
            analysis = json.loads(response_text)
            
            # Validate and enhance the analysis
            analysis = self._validate_and_enhance_analysis(analysis, trajectory)
            
            return analysis
            
        except Exception as e:
            logging.error(f"Failed to analyze trajectory: {e}")
            # Return enhanced default analysis
            return {
                "failure_step": len(trajectory) - 1,
                "critical_module": "others",
                "failure_type": "others", 
                "reason": "Failed to analyze trajectory due to technical error",
                "suggestion": "Try a different approach or review the task requirements",
                "critical_step": 0,
                "evidence": "Analysis failed due to technical error",
                "confidence": 0.1,
                "root_cause": "Technical analysis failure"
            }
    
    def generate_feedback(
        self,
        observation: str,
        analysis: Dict,
        previous_action: str,
        env_type: str,
        log_path: Optional[str] = None,
    ) -> str:
        """
        Generate enhanced feedback based on critical error analysis
        
        Args:
            observation: Current observation from the environment
            analysis: Enhanced analysis dict from analyze_trajectory
            previous_action: The action that led to failure
            env_type: Type of environment
        
        Returns:
            Detailed feedback string to inject into observation
        """
        
        # Extract comprehensive error information
        critical = analysis.get('raw_critical_error', {})
        critical_module = critical.get('critical_module', analysis.get('critical_module', 'unknown'))
        error_type = critical.get('error_type', analysis.get('failure_type', 'unknown'))
        root_cause = critical.get('root_cause', analysis.get('root_cause', analysis.get('reason', 'Unknown error')))
        correction_guidance = critical.get('correction_guidance', analysis.get('suggestion', 'Try a different approach'))
        evidence = critical.get('evidence', analysis.get('evidence', ''))
        confidence = critical.get('confidence', analysis.get('confidence', 0.5))
        
        # Build context-aware feedback prompt
        feedback_prompt = f"""You are an expert agent coach. Based on a detailed failure analysis, generate specific, actionable feedback for an agent that made a critical error.

ENVIRONMENT: {env_type}
CURRENT OBSERVATION: {observation}
PREVIOUS FAILED ACTION: {previous_action}

DETAILED FAILURE ANALYSIS:
- Critical Module: {critical_module}
- Error Type: {error_type}  
- Root Cause: {root_cause}
- Evidence: {evidence}
- Correction Guidance: {correction_guidance}
- Confidence: {confidence:.2f}

FEEDBACK REQUIREMENTS:
Generate a clear, actionable feedback message that:
1. SPECIFICALLY addresses the {critical_module} module error of type '{error_type}'
2. Explains WHY the previous approach failed (based on root cause)
3. Provides CONCRETE guidance on what to do differently
4. References specific evidence from the trajectory when relevant
5. Is concise but comprehensive (3-4 sentences max)

Focus on helping the agent understand both the specific mistake and the corrective action needed.

Output only the feedback message, no additional formatting."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert agent coach providing specific, actionable feedback based on detailed error analysis."},
                    {"role": "user", "content": feedback_prompt}
                ],
                temperature=self.temperature * 0.8  # Slightly lower temperature for more focused feedback
            )
            
            response_text = response.choices[0].message.content
            self._log_llm_call(log_path, feedback_prompt, response_text)
            feedback = response_text.strip()
            
            # Format comprehensive feedback for injection
            formatted_feedback = f"""
[DEBUGGER FEEDBACK - Critical Error Detected]
Error Type: {critical_module}::{error_type} (Confidence: {confidence:.2f})
Root Cause: {root_cause}

{feedback}

Specific Guidance: {correction_guidance}
"""
            
            return formatted_feedback
            
        except Exception as e:
            logging.error(f"Failed to generate enhanced feedback: {e}")
            # Fallback to basic feedback with available information
            fallback_feedback = f"""
[DEBUGGER FEEDBACK - Previous Attempt Failed]
Error: {critical_module}::{error_type}
Issue: {root_cause}
Recommendation: {correction_guidance}
"""
            return fallback_feedback
    
    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """Format trajectory for LLM analysis"""
        lines = []
        for step in trajectory:
            lines.append(f"Step {step['step']}:")
            lines.append(f"  Observation: {step.get('observation', 'N/A')}")
            lines.append(f"  Action: {step.get('action', 'N/A')}")
            if step.get('reward') is not None:
                lines.append(f"  Reward: {step['reward']}")
            if step.get('done'):
                lines.append(f"  Done: {step['done']}, Won: {step.get('won', False)}")
            lines.append("")
        return "\n".join(lines)


class AdvancedDebugger(LLMDebugger):
    """Adapter that connects the rollout debugger to the advanced analysis API."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        base_url: str | None = None,
        api_key: Optional[str] = None,
        analysis_model: Optional[str] = None,
        capture_debug_data: bool = False,
    ) -> None:
        super().__init__(model_name=model_name, temperature=temperature, base_url=base_url, api_key=api_key)

        if not ADVANCED_DEBUGGER_AVAILABLE:
            raise ImportError("Advanced debugger API is not available in the current environment")

        # Prefer provided API key, fall back to environment.
        # Allow empty key when using local OpenAI-compatible endpoints (e.g., vLLM) via base_url.
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key and base_url is None:
            raise ValueError("OPENAI_API_KEY must be set for AdvancedDebugger when no --debugger_base_url is provided")

        self.analysis_model = analysis_model or model_name
        self.capture_debug_data = capture_debug_data
        self.detector = AgentErrorDetectorAPI(
            self.api_key,
            model=self.analysis_model,
            capture_debug_data=capture_debug_data,
            base_url=base_url,
        )

    def analyze_trajectory(
        self,
        trajectory: List[Dict],
        env_type: str,
        chat_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        previous_instructions: Optional[List[str]] = None,
        generate_follow_up: bool = False,
        log_path: Optional[str] = None,
    ) -> Dict:
        if not chat_history:
            msg = "Advanced debugger requires chat history; none provided for analysis"
            logging.error(msg)
            raise RuntimeError(msg)

        # Build the trajectory_json payload that the API expects
        
        ### Fix: trajectory_json is not a valid JSON object for the API
        trajectory_json = self._build_trajectory_json(trajectory, env_type, chat_history, metadata)
        debug_input_payload = _json_safe_copy(trajectory_json) if self.capture_debug_data else None
        
        # Log the structure of trajectory_json for debugging
        logging.info(
            "Advanced debugger starting analysis: steps=%s chat_messages=%s env=%s",
            len(trajectory),
            len(chat_history),
            env_type,
        )
        
        # Log detailed structure for debugging
        if trajectory:
            logging.debug("First trajectory step: %s", trajectory[0])
        if chat_history:
            logging.debug("First chat message: %s", chat_history[0])

        try:
            # Call the API with the properly formatted trajectory_json
            result = self._run_async(self.detector.analyze_trajectory(trajectory_json))
            
            logging.info(
                "Advanced debugger API response received: type=%s keys=%s",
                type(result).__name__,
                list(result.keys()) if isinstance(result, dict) else None,
            )
            
            # Log the actual critical error if found
            if isinstance(result, dict) and 'critical_error' in result:
                critical = result['critical_error']
                if critical:
                    logging.info(
                        "Critical error identified: step=%s module=%s type=%s",
                        critical.get('critical_step'),
                        critical.get('critical_module'),
                        critical.get('error_type')
                    )
                else:
                    logging.warning("No critical error found by advanced debugger")
        except Exception as exc:
            logging.error(f"Advanced debugger API call failed: {exc}")
            logging.error(f"Exception type: {type(exc).__name__}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Advanced debugger API call failed: {exc}") from exc

        # Convert the API result to the expected format
        converted = self._convert_api_result(result, trajectory, env_type)

        if self.capture_debug_data:
            if isinstance(result, dict):
                debug_payload = result.get('debug_payload')
                if debug_payload is not None:
                    converted['debug_payload'] = _json_safe_copy(debug_payload)
            if debug_input_payload is not None:
                converted['debug_input'] = debug_input_payload

        # Add metadata
        safe_metadata = _json_safe_copy(metadata or {})
        converted["metadata"] = safe_metadata

        return converted

    def _run_async(self, coroutine):
        """Run an async coroutine in a temporary event loop."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coroutine)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def _validate_trajectory_json(self, data: Dict) -> None:
        """Validate trajectory_json conforms to API expected format.

        Raises:
            ValueError: If the structure doesn't meet API requirements
        """
        if not isinstance(data.get('metadata'), dict):
            raise ValueError("trajectory_json missing valid 'metadata' dict")

        metadata = data['metadata']
        if not metadata.get('environment'):
            raise ValueError("metadata missing 'environment' field")

        if not ('success' in metadata or 'won' in metadata):
            raise ValueError("metadata missing 'success' or 'won' field")

        messages = data.get('messages', [])
        if not messages:
            raise ValueError("trajectory_json missing or empty 'messages'")

        for i, msg in enumerate(messages):
            if not isinstance(msg.get('role'), str):
                raise ValueError(f"Message {i} missing valid 'role' field")
            if not isinstance(msg.get('content'), str):
                raise ValueError(f"Message {i} missing valid 'content' field")

    def _build_trajectory_json(
        self,
        trajectory: List[Dict],
        env_type: str,
        chat_history: List[Dict],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the trajectory_json payload that the API expects."""

        # Create clean metadata
        safe_metadata = _json_safe_copy(metadata or {})
        if not isinstance(safe_metadata, dict):
            safe_metadata = {"metadata": safe_metadata}

        safe_metadata.setdefault("environment", env_type)
        # Note: task_success is kept for internal use, but API reads 'success' or 'won'
        safe_metadata.setdefault("task_success", False)
        safe_metadata.setdefault("won", False)
        # API expects 'success' field for task completion status
        safe_metadata.setdefault("success", safe_metadata.get("won", False))

        # Preserve task-related fields if they exist
        if metadata:
            if "task" in metadata:
                safe_metadata.setdefault("task", metadata["task"])
            if "task_id" in metadata:
                safe_metadata.setdefault("task_id", metadata["task_id"])

        # Build the trajectory_json structure expected by the API
        # The API's parse_trajectory_from_dict extracts steps from messages
        trajectory_json = {
            "metadata": safe_metadata,
            "messages": chat_history,  # API extracts trajectory from chat messages
            "chat_history": chat_history,  # Backward compatibility
        }

        # Validate the structure before returning
        try:
            self._validate_trajectory_json(trajectory_json)
        except ValueError as e:
            logging.error(f"Invalid trajectory_json structure: {e}")
            raise

        logging.debug(
            "Built trajectory_json with %d messages, metadata keys: %s",
            len(chat_history),
            list(safe_metadata.keys())
        )

        return trajectory_json

    def _convert_api_result(
        self,
        result: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        env_type: str,
    ) -> Dict[str, Any]:
        """Convert the API result to the expected format, no fallbacks."""
        
        if not isinstance(result, dict):
            raise TypeError(f"API result must be a dict, got {type(result).__name__}")
        
        # Extract phase1 errors and critical error from API response
        phase1_errors = result.get('phase1_errors')
        critical_error = result.get('critical_error')
        
        if not critical_error:
            raise ValueError("Advanced debugger API did not return a critical error")
        
        if not isinstance(critical_error, dict):
            raise TypeError(f"Critical error must be a dict, got {type(critical_error).__name__}")
        
        # Extract required fields from critical error
        critical_step = critical_error.get('critical_step')
        if critical_step is None:
            raise ValueError("Critical error missing 'critical_step' field")
        
        critical_module = critical_error.get('critical_module')
        if not critical_module:
            raise ValueError("Critical error missing 'critical_module' field")
        
        error_type = critical_error.get('error_type')
        if not error_type:
            raise ValueError("Critical error missing 'error_type' field")
        
        # Get other fields with required values (no defaults)
        root_cause = critical_error.get('root_cause')
        if not root_cause:
            raise ValueError("Critical error missing 'root_cause' field")
        
        correction_guidance = critical_error.get('correction_guidance')
        if not correction_guidance:
            raise ValueError("Critical error missing 'correction_guidance' field")
        
        # Convert step number (API uses 1-based, we need 0-based index)
        try:
            critical_step_int = int(critical_step)
            failure_step = max(critical_step_int - 1, 0)  # Convert to 0-based
            
            # Ensure within bounds
            if trajectory:
                max_index = len(trajectory) - 1
                failure_step = min(failure_step, max_index)
            
            # Calculate the step before failure
            critical_step_index = failure_step - 1
            if critical_step_index < -1:
                critical_step_index = -1
                
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid critical_step value: {critical_step}") from e
        
        # Build failure type string
        failure_type = f"{critical_module}::{error_type}"
        
        # Get evidence and other details
        evidence = critical_error.get('evidence', '')
        confidence = critical_error.get('confidence', 0.0)
        cascading_effects = critical_error.get('cascading_effects', [])
        
        # Build the result
        converted_result = {
            "failure_step": failure_step,
            "failure_type": failure_type,
            "reason": root_cause,
            "suggestion": correction_guidance,
            "critical_step": critical_step_index,
            "raw_critical_error": _json_safe_copy(critical_error),
            "phase1_errors": _json_safe_copy(phase1_errors) if phase1_errors else None,
            "evidence": evidence,
            "confidence": confidence,
            "cascading_effects": cascading_effects,
        }
        
        logging.info(
            "Converted API result: failure_step=%d, failure_type=%s, confidence=%.2f",
            failure_step,
            failure_type,
            confidence
        )
        
        return converted_result


class ContinuousInstructionManager:
    """Manages cumulative follow-up instructions for continuous debugging"""
    
    def __init__(self):
        self.instructions = {}  # env_id -> list of follow-up instructions
        self.instruction_history = {}  # env_id -> list of (attempt_idx, instruction) tuples
    
    def reset(self, env_id: int):
        """Reset instructions for an environment"""
        self.instructions[env_id] = []
        self.instruction_history[env_id] = []
    
    def add_instruction(self, env_id: int, instruction: str, attempt_idx: int):
        """Add a new follow-up instruction"""
        if env_id not in self.instructions:
            self.instructions[env_id] = []
        if env_id not in self.instruction_history:
            self.instruction_history[env_id] = []
            
        # Avoid duplicating similar instructions
        if instruction and instruction not in self.instructions[env_id]:
            self.instructions[env_id].append(instruction)
            self.instruction_history[env_id].append((attempt_idx, instruction))
    
    def get_instructions(self, env_id: int) -> List[str]:
        """Get all accumulated instructions for an environment"""
        return self.instructions.get(env_id, [])
    
    def get_instruction_history(self, env_id: int) -> List[Tuple[int, str]]:
        """Get instruction history with attempt indices"""
        return self.instruction_history.get(env_id, [])
    
    def format_instructions_for_observation(self, env_id: int) -> str:
        """Format instructions for injection into observation"""
        instructions = self.get_instructions(env_id)
        if not instructions:
            return ""
        
        formatted = "[CONTINUOUS DEBUGGER GUIDANCE]\n"
        formatted += "Based on previous failures, remember to:\n"
        for i, instruction in enumerate(instructions, 1):
            formatted += f"{i}. {instruction}\n"
        formatted += "Apply these guidelines throughout your problem-solving process.\n"
        return formatted


class TrajectoryManager:
    """Manages trajectory storage and replay for multiple environments"""
    
    def __init__(self):
        self.trajectories = {}  # env_id -> list of trajectory steps
        self.attempts = {}  # env_id -> list of all attempt trajectories
    
    def reset(self, env_id: int):
        """Reset trajectory for an environment"""
        self.trajectories[env_id] = []
        if env_id not in self.attempts:
            self.attempts[env_id] = []
    
    def add_step(self, env_id: int, step_data: Dict):
        """Add a step to the current trajectory"""
        if env_id not in self.trajectories:
            self.trajectories[env_id] = []
        self.trajectories[env_id].append(step_data)
    
    def save_attempt(self, env_id: int):
        """Save current trajectory as an attempt"""
        if env_id in self.trajectories:
            self.attempts[env_id].append(self.trajectories[env_id].copy())
    
    def get_trajectory(self, env_id: int) -> List[Dict]:
        """Get the current trajectory for an environment"""
        return self.trajectories.get(env_id, [])
    
    def get_all_attempts(self, env_id: int) -> List[List[Dict]]:
        """Get all attempt trajectories for an environment"""
        return self.attempts.get(env_id, [])
    
    def get_replay_point(self, env_id: int, target_step: int) -> Tuple[List[Dict], int]:
        """
        Get trajectory up to a certain step for replay
        
        Returns:
            Tuple of (trajectory_up_to_step, actual_step_reached)
        """
        trajectory = self.trajectories.get(env_id, [])
        if target_step >= len(trajectory):
            return trajectory, len(trajectory) - 1
        return trajectory[:target_step + 1], target_step


class ExtendedEnvironmentManager:
    """Single‑env helpers on top of an existing manager.

    Important:
    - These helpers are only correct when the underlying manager controls
      exactly one environment instance (env_num == 1).
    - This script uses one‑env managers for debugger rollouts so we can
      freely reset during a rollout without affecting others.
    """

    def __init__(self, base_manager):
        # Keep a handle to the original manager; delegate attribute access.
        self.base_manager = base_manager
        self.__dict__.update(base_manager.__dict__)

    def reset_single(self, env_id: int):
        """Reset the underlying single environment.

        Note: env_id is ignored by design since this wrapper is only used
        when env_num == 1. We keep the signature for compatibility.
        """
        obs, infos = self.reset()
        return obs, infos

    def step_single(self, env_id: int, action: str):
        """Step the underlying single environment with the given action."""
        # For a single environment we just forward a singleton action list.
        obs, rewards, dones, infos = self.step([action])
        return obs, rewards, dones, infos

    def __getattr__(self, name):
        # Delegate unknown attributes to the base manager instance.
        return getattr(self.base_manager, name)


class EnvironmentFactory:
    """Factory for creating different environment types"""
    
    @staticmethod
    def build_env(env_type: str, with_debugger: bool = False, **kwargs) -> Any:
        """Build environment based on type"""
        
        if env_type == "alfworld":
            env = EnvironmentFactory._build_alfworld(**kwargs)
        elif env_type == "gaia":
            env = EnvironmentFactory._build_gaia(**kwargs)
        elif env_type == "webshop":
            env = EnvironmentFactory._build_webshop(**kwargs)
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
        
        # Wrap with extended manager if debugger is enabled
        if with_debugger:
            return ExtendedEnvironmentManager(env)
        return env
    
    @staticmethod
    def _build_alfworld(env_num: int = 1, seed: int = 1, history_length: int = 2,
                       alf_env_type: str = "alfworld/AlfredTWEnv", 
                       game_files: Optional[List[str]] = None, **kwargs):
        """Build AlfWorld environment"""
        from openmanus_rl.environments.env_package.alfworld import alfworld_projection
        from openmanus_rl.environments.env_package.alfworld import build_alfworld_envs
        
        alf_config_path = os.path.join(
            os.path.dirname(__file__), 
            '../../openmanus_rl/environments/env_package/alfworld/configs/config_tw.yaml'
        )
        
        envs = build_alfworld_envs(
            alf_config_path, 
            seed=seed, 
            env_num=env_num, 
            group_n=1, 
            is_train=True, 
            env_kwargs={}, 
            game_files=game_files
        )
        
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name=alf_env_type, 
                history_length=history_length
            )
        )
        
        return AlfWorldEnvironmentManager(envs, alfworld_projection, cfg)
    
    @staticmethod
    def _build_gaia(tasks_data: List[Dict], available_tools: List[str], 
                   env_num: int = 1, seed: int = 1, history_length: int = 2,
                   max_steps: int = 30, **kwargs):
        """Build GAIA/Tool Use environment"""
        from openmanus_rl.environments.env_package.tool_use import tool_use_projection
        from openmanus_rl.environments.env_package.tool_use import build_tool_use_envs
        from openmanus_rl.environments.env_package.tool_use.manager import ToolUseEnvironmentManager
        
        envs = build_tool_use_envs(
            tasks_data=tasks_data,
            available_tools=available_tools,
            seed=seed,
            env_num=env_num,
            group_n=1,
            is_train=True
        )
        
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name="tool_use",
                history_length=history_length,
                max_steps=max_steps
            )
        )
        
        return ToolUseEnvironmentManager(envs, tool_use_projection, cfg)
    
    @staticmethod
    def _build_webshop(env_num: int = 1, seed: int = 1, history_length: int = 2,
                       use_train_set: bool = False, **kwargs):
        """Build WebShop environment"""
        from openmanus_rl.environments.env_package.webshop import build_webshop_envs, webshop_projection
        
        env_kwargs = {"observation_mode": "text"}
        
        envs = build_webshop_envs(
            seed=seed,
            env_num=env_num,
            group_n=1,
            is_train=use_train_set,
            env_kwargs=env_kwargs,
        )
        
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name="webshop/WebAgentTextEnv",
                history_length=history_length
            )
        )
        
        return WebshopEnvironmentManager(envs, webshop_projection, cfg)


def load_gaia_tasks(data_path: str, max_tasks: Optional[int] = None) -> List[Dict]:
    """Load GAIA tasks from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    return tasks


def get_task_id(env_type: str, env_id: int, info: Dict, batch_idx: int = 0) -> str:
    """
    Get a unique task identifier for organizing outputs
    
    Args:
        env_type: Type of environment
        env_id: Environment ID within batch
        info: Info dictionary from environment
        batch_idx: Batch index
        
    Returns:
        Unique task identifier string
    """
    if env_type == "alfworld":
        # Prefer the trial directory name (unique) rather than the flat filename 'game.tw-pddl'.
        gamefile = info.get("extra.gamefile", "")
        if gamefile:
            trial_dir = os.path.basename(os.path.dirname(gamefile)) or "unknown"
            return f"alfworld_b{batch_idx:03d}_e{env_id:03d}_{trial_dir}"
        else:
            return f"alfworld_b{batch_idx:03d}_e{env_id:03d}_unknown"
    elif env_type == "gaia":
        pid = info.get("pid", f"unknown_{env_id}")
        return f"gaia_b{batch_idx:03d}_e{env_id:03d}_{pid}"
    elif env_type == "webshop":
        # Try to extract task ID from info
        task_id = info.get("task_id", f"task_{env_id}")
        return f"webshop_b{batch_idx:03d}_e{env_id:03d}_{task_id}"
    else:
        return f"{env_type}_b{batch_idx:03d}_e{env_id:03d}"


def prepare_alfworld_game_files(env_type: str, total_envs: int, seed: int) -> Optional[List[str]]:
    """Prepare unique game files for AlfWorld if requested"""
    if env_type != "alfworld":
        return None
        
    from openmanus_rl.environments.env_package.alfworld.envs import load_config_file
    from openmanus_rl.environments.env_package.alfworld.alfworld.agents.environment import get_environment
    
    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        '../../openmanus_rl/environments/env_package/alfworld/configs/config_tw.yaml'
    )
    
    try:
        cfg = load_config_file(alf_config_path)
        env_type = cfg['env']['type']
        BaseEnvCls = get_environment(env_type)
        tmp_env = BaseEnvCls(cfg, train_eval='train')
        tmp_env.collect_game_files()
        all_game_files = list(getattr(tmp_env, 'game_files', []))

        # Deduplicate while preserving order—some AlfWorld configs can yield
        # repeated entries when multiple workers share the same task list.
        seen = set()
        unique_game_files = []
        for path in all_game_files:
            if path not in seen:
                seen.add(path)
                unique_game_files.append(path)
        all_game_files = unique_game_files

        if len(all_game_files) < total_envs:
            logging.error(f"Not enough game files: need {total_envs}, have {len(all_game_files)}")
            return None
            
        # Preserve the natural order reported by the environment so downstream scheduling
        # can rely on deterministic task ordering.
        return all_game_files[:total_envs]
        
    except Exception as e:
        logging.error(f"Failed to collect game files: {e}")
        return None


def generate_debugger_feedback_text(analysis: Dict[str, Any]) -> str:
    """
    Generate enhanced debugger feedback text based on comprehensive analysis results.
    
    Args:
        analysis: Enhanced analysis dict with detailed error classification
        
    Returns:
        Formatted feedback string to inject into observation  
    """
    # Extract comprehensive error information
    raw_critical = analysis.get('raw_critical_error', {})
    
    if raw_critical:
        critical_module = raw_critical.get('critical_module', 'unknown')
        error_type = raw_critical.get('error_type', 'unknown')
        root_cause = raw_critical.get('root_cause', 'An error occurred in your previous attempt.')
        correction_guidance = raw_critical.get('correction_guidance', 'Try a different approach.')
        evidence = raw_critical.get('evidence', '')
        confidence = raw_critical.get('confidence', 0.5)
        failure_type = f"{critical_module}::{error_type}"
    else:
        # Fallback to enhanced analysis format
        critical_module = analysis.get('critical_module', 'unknown')
        error_type = analysis.get('failure_type', 'unknown')
        root_cause = analysis.get('root_cause', analysis.get('reason', 'An error occurred in your previous attempt.'))
        correction_guidance = analysis.get('suggestion', 'Try a different approach.')
        evidence = analysis.get('evidence', '')
        confidence = analysis.get('confidence', 0.5)
        failure_type = f"{critical_module}::{error_type}"
    
    # Build comprehensive feedback with error type awareness
    feedback_parts = [
        f"[DEBUGGER FEEDBACK - Critical Error Analysis]",
        f"Previous Attempt Failed: {failure_type} (Confidence: {confidence:.2f})",
        f"Root Cause: {root_cause}",
        f"Corrective Action: {correction_guidance}"
    ]
    
    if evidence and evidence.strip():
        feedback_parts.append(f"Supporting Evidence: {evidence}")
    
    feedback_parts.append("This is a replay attempt - apply the corrective guidance to avoid the same mistake.")
    
    return "\n".join(feedback_parts)


def run_environment_with_retry(
    env_id: int,
    env_manager,
    agent: UnifiedAgent,
    max_steps: int,
    env_type: str,
    debugger: Optional[LLMDebugger] = None,
    trajectory_manager: Optional[TrajectoryManager] = None,
    max_retries: int = 5,
    dump_fp=None,
    dump_lock=None,
    chat_base_dir: str = None,
    batch_idx: int = 0,
    test_idx: int = 0,
    global_env_counter: int = 0,
    run_ts: str = "",
    debug_output_dir: str = None,
    save_all_attempts: bool = False,
    task_dir: str = None,
    shared_llm_executor=None,
    continuous_instruction_manager: Optional[ContinuousInstructionManager] = None,
    debugger_type: str = "naive"
) -> Dict:
    """
    Run a single environment with retry logic using debugger feedback
    
    Returns:
        Dict containing results and statistics for this environment
    """
    
    last_trajectory = None  # Track the last trajectory for debugging
    last_chat_history: Optional[List[Dict[str, Any]]] = None
    last_metadata: Optional[Dict[str, Any]] = None
    won = False
    final_info = {}
    all_attempt_trajectories = []
    final_reward = 0
    first_attempt_success = False  # Track if first attempt was successful

    analysis_log_path: Optional[str] = None
    feedback_log_path: Optional[str] = None
    if debugger and task_dir:
        analysis_log_path = os.path.join(task_dir, "debugger_analysis_calls.jsonl")
        feedback_log_path = os.path.join(task_dir, "debugger_feedback_calls.jsonl")
        for log_path in (analysis_log_path, feedback_log_path):
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                if not os.path.exists(log_path):
                    with open(log_path, "a", encoding="utf-8"):
                        pass
            except Exception as exc:
                logging.debug(f"Failed to initialize debugger log file {log_path}: {exc}")

    for retry_idx in range(max_retries):
        logging.info(f"  Env {env_id} - Attempt {retry_idx + 1}/{max_retries}")
        env_manager.clear_persistent_guidance(env_id)
        
        # Reset trajectory manager for this attempt
        if trajectory_manager:
            trajectory_manager.reset(env_id)
        
        # Initialize continuous instruction manager if needed
        if continuous_instruction_manager and retry_idx == 0:
            continuous_instruction_manager.reset(env_id)
        
        # Initialize tracking variables
        env_done = False
        chat_history = []
        trajectory_steps = []
        cumulative_reward = 0
        
        # Variables for replay
        replay_to_step = -1
        debugger_feedback = ""
        analysis = None
        follow_up_instruction = None
        instruction_overlay = ""
        guidance_list: List[str] = []
        guidance_history: List[Tuple[int, str]] = []
        
        # If this is a retry, analyze the failed trajectory
        if retry_idx > 0 and debugger and last_trajectory and not won:
            # Get previous instructions for continuous debugging
            previous_instructions = []
            generate_follow_up = False
            if continuous_instruction_manager and debugger_type == "continue":
                previous_instructions = continuous_instruction_manager.get_instructions(env_id)
                generate_follow_up = True
            
            analysis = debugger.analyze_trajectory(
                last_trajectory,
                env_type,
                chat_history=last_chat_history,
                metadata=last_metadata,
                previous_instructions=previous_instructions,
                generate_follow_up=generate_follow_up,
                log_path=analysis_log_path,
            )
            
            # Extract and store follow-up instruction for continuous debugging
            if continuous_instruction_manager and debugger_type == "continue":
                follow_up_instruction = analysis.get('follow_up_instruction')
                if follow_up_instruction:
                    continuous_instruction_manager.add_instruction(env_id, follow_up_instruction, retry_idx)
                    logging.info(f"    Added follow-up instruction: {follow_up_instruction}")

                guidance_list = continuous_instruction_manager.get_instructions(env_id)
                instruction_overlay = continuous_instruction_manager.format_instructions_for_observation(env_id)
                guidance_history = continuous_instruction_manager.get_instruction_history(env_id)

                if guidance_list:
                    logging.info(f"    Total accumulated instructions: {len(guidance_list)}")

                # Persist continuous guidance details inside the latest analysis for downstream consumers
                analysis['continuous_guidance'] = guidance_list
                analysis['continuous_guidance_history'] = [
                    {"attempt": attempt_idx, "instruction": guidance}
                    for attempt_idx, guidance in guidance_history
                ]
                if instruction_overlay:
                    analysis['continuous_guidance_overlay'] = instruction_overlay
            
            # Save debug analysis to task dir if specified
            if task_dir:
                debug_file = os.path.join(
                    task_dir,
                    f"debug_analysis_retry_{retry_idx}.json"
                )
                
                # Extract raw critical error information 
                raw_critical = analysis.get('raw_critical_error', {})
                debug_record = {
                    "retry": retry_idx,
                    "critical_step": raw_critical.get('critical_step', analysis.get('critical_step', -1)),
                    "critical_module": raw_critical.get('critical_module', 'unknown'),
                    "error_type": raw_critical.get('error_type', analysis.get('failure_type', 'unknown')),
                    "root_cause": raw_critical.get('root_cause', analysis.get('reason', 'Unknown error')),
                    "evidence": raw_critical.get('evidence', ''),
                    "correction_guidance": raw_critical.get('correction_guidance', analysis.get('suggestion', 'Try a different approach')),
                    "confidence": raw_critical.get('confidence', 0.0),
                    "cascading_effects": raw_critical.get('cascading_effects', []),
                    "trajectory": last_trajectory,
                    "env_type": env_type
                }

                if continuous_instruction_manager and debugger_type == "continue":
                    debug_record["follow_up_instruction"] = follow_up_instruction
                    debug_record["continuous_guidance"] = guidance_list
                    debug_record["continuous_guidance_history"] = [
                        {"attempt": attempt_idx, "instruction": guidance}
                        for attempt_idx, guidance in guidance_history
                    ]
                    if instruction_overlay:
                        debug_record["continuous_guidance_overlay"] = instruction_overlay

                if getattr(debugger, "capture_debug_data", False):
                    debug_record["chat_history"] = last_chat_history
                    debug_record["attempt_metadata"] = last_metadata
                    debug_record["full_analysis"] = analysis

                with open(debug_file, "w") as f:
                    json.dump(debug_record, f, indent=2)
            
            # Extract critical step from raw error or analysis
            raw_critical = analysis.get('raw_critical_error', {})
            critical_step_1based = raw_critical.get('critical_step') or analysis.get('critical_step', 1)
            
            logging.info(f"    Debugger analysis - Critical step: {critical_step_1based}, Error: {raw_critical.get('error_type', analysis.get('failure_type', 'unknown'))}")
            logging.info(f"    Root cause: {raw_critical.get('root_cause', analysis.get('reason', 'Unknown'))}")
            logging.info(f"    Correction guidance: {raw_critical.get('correction_guidance', analysis.get('suggestion', 'Try different approach'))}")
            

            critical_step_1based = int(critical_step_1based)
            # AlfWorld environment shows "step 1" in first observation, so offset is 1
            feedback_inject_step_0based = critical_step_1based  # Inject feedback at error step
            replay_to_step_0based = feedback_inject_step_0based -1  # Replay up to step before error

            
            # Handle bounds checking
            if feedback_inject_step_0based < 0:
                logging.info(f"    First step failure detected (critical_step={critical_step_1based}), will inject feedback at step 0")
                feedback_inject_step_0based = 0
                replay_to_step_0based = -1
            elif feedback_inject_step_0based >= len(last_trajectory):
                logging.warning(f"    Feedback inject step {feedback_inject_step_0based} exceeds trajectory length {len(last_trajectory)}, adjusting")
                feedback_inject_step_0based = len(last_trajectory) - 1
                replay_to_step_0based = feedback_inject_step_0based - 1
            
            logging.info(f"    Will inject feedback at trajectory step {feedback_inject_step_0based} (critical_step={critical_step_1based})")
            
            # Setup actions to replay (up to the step before the error)
            actions_to_replay = []
            if replay_to_step_0based >= 0:
                actions_to_replay = [step['action'] for step in last_trajectory[:replay_to_step_0based + 1]]
                logging.info(f"    Will replay {len(actions_to_replay)} actions before injecting feedback")
            
            # Setup replay mode in the environment manager
            feedback_text = ""
            if debugger:
                failure_step = None
                failure_step_idx = feedback_inject_step_0based
                if last_trajectory:
                    if 0 <= failure_step_idx < len(last_trajectory):
                        failure_step = last_trajectory[failure_step_idx]
                    else:
                        failure_step = last_trajectory[-1]
                        failure_step_idx = len(last_trajectory) - 1
                failure_observation = ""
                failure_action = ""
                if failure_step:
                    failure_observation = failure_step.get("observation", "") or ""
                    failure_action = failure_step.get("action", "") or ""
                try:
                    feedback_text = debugger.generate_feedback(
                        failure_observation,
                        analysis,
                        failure_action,
                        env_type,
                        log_path=feedback_log_path,
                    )
                except Exception as exc:
                    logging.warning(f"Failed to generate LLM feedback, falling back to template: {exc}")
                    feedback_text = generate_debugger_feedback_text(analysis)
                if not feedback_text:
                    feedback_text = generate_debugger_feedback_text(analysis)
            else:
                feedback_text = generate_debugger_feedback_text(analysis)
            logging.info(f"    Setting up replay: actions_to_replay={len(actions_to_replay)}, feedback_inject_step={feedback_inject_step_0based}")
            logging.info(f"    Feedback text: {feedback_text[:100]}...")
            logging.info(f"    Debug: env_manager type = {type(env_manager).__name__}")
            persistent_guidance_text = instruction_overlay if instruction_overlay else None
            if persistent_guidance_text:
                logging.info(
                    f"    Persistent guidance will be injected from step {feedback_inject_step_0based}: {persistent_guidance_text[:100]}..."
                )
            env_manager.setup_replay(
                env_id,
                actions_to_replay,
                feedback_inject_step_0based,
                feedback_text,
                persistent_guidance_text=persistent_guidance_text,
                persistent_guidance_start=feedback_inject_step_0based,
            )
            
            # Verify setup
            if hasattr(env_manager, 'debugger_feedback') and env_id in env_manager.debugger_feedback:
                logging.info(f"    Replay setup verified: feedback will be injected at step {env_manager.debugger_feedback[env_id]['step']}")

        if continuous_instruction_manager and debugger_type == "continue" and not instruction_overlay:
            instruction_overlay = continuous_instruction_manager.format_instructions_for_observation(env_id)

        # Get initial observation
        obs_dict, info_dict = env_manager.reset_single(env_id)
        obs = obs_dict["text"][env_id]
        info = info_dict[env_id] if isinstance(info_dict, dict) else info_dict

        if not isinstance(info, dict):
            info = {}

        initial_info = info.copy()
        initial_observation = obs

        for step_idx in range(max_steps):
            if env_done:
                break
                
            # Check if we should use a replay action first
            replay_action = env_manager.get_replay_action(env_id)
            if replay_action is not None:
                action = replay_action
                logging.debug(f"    Using replay action for step {step_idx}: {action}")
            else:
                # Get action from agent - replay mode is finished, get new action from LLM
                # The observation already includes debugger feedback if this is the critical step
                prompt = obs
                
                # For continuous debugging, inject accumulated instructions into every observation
                if continuous_instruction_manager and debugger_type == "continue" and instruction_overlay:
                    if instruction_overlay not in prompt:
                        prompt = instruction_overlay + "\n\n" + prompt
                
                # Log if we expect debugger feedback in this observation
                if debugger and analysis:
                    pending_feedback = getattr(env_manager, "debugger_feedback", {})
                    feedback_meta = pending_feedback.get(env_id)
                    if feedback_meta and feedback_meta.get('step') == step_idx:
                        logging.info(f"    Step {step_idx}: Debugger feedback should be in observation")
                
                # Use shared LLM executor if available for better concurrency
                if shared_llm_executor is not None:
                    llm_future = agent.get_action_from_llm_with_shared_pool(prompt, shared_llm_executor)
                    action = llm_future.result()  # This will block until LLM responds, but allows other tasks to proceed
                else:
                    action = agent.get_action_from_llm(prompt)
            
            # Store raw action for trajectory
            raw_action = action
            
            # Step environment
            obs_dict, reward_dict, done_dict, info_dict = env_manager.step_single(env_id, action)
            
            obs = obs_dict["text"][env_id]
            reward = reward_dict[env_id]
            done = done_dict[env_id]
            # info_dict is a list, get the element at env_id
            info = info_dict[env_id] if isinstance(info_dict, list) else info_dict
            
            # Ensure info is a dictionary
            if not isinstance(info, dict):
                info = {}
            
            cumulative_reward += reward
            
            # Store trajectory step (step_idx is 0-based)
            trajectory_step = {
                "step": step_idx,  # Keep 0-based indexing for consistency
                "observation": obs,
                "action": raw_action,
                "reward": float(reward),
                "done": bool(done),
                "won": bool(info.get("won", False))
            }
            trajectory_steps.append(trajectory_step)
            
            if trajectory_manager:
                trajectory_manager.add_step(env_id, trajectory_step)
            
            # Update chat history
            chat_history.append({"role": "user", "content": prompt})
            chat_history.append({"role": "assistant", "content": raw_action})
            
            # Write to dump file if specified
            if dump_fp and (save_all_attempts or retry_idx == 0 or won):
                try:
                    row = {
                        "batch_idx": batch_idx,
                        "test_idx": test_idx,
                        "retry_idx": retry_idx,
                        "step": step_idx,
                        "env_id": global_env_counter + env_id,
                        "prompt": prompt,
                        "action": raw_action,
                        "reward": float(reward),
                        "done": bool(done),
                        "won": bool(info.get("won", False)),
                        "is_action_valid": bool(info.get("is_action_valid", False)),
                        "env_type": env_type
                    }
                    
                    # Add environment-specific fields
                    if env_type == "gaia":
                        row["pid"] = info.get("pid", "unknown")
                    elif env_type == "alfworld":
                        row["gamefile"] = info.get("extra.gamefile", "")
                    elif env_type == "webshop":
                        row["task_score"] = float(info.get("task_score", 0))
                    
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                    if dump_lock is not None:
                        # Serialize writes across threads
                        with dump_lock:
                            dump_fp.write(line)
                            dump_fp.flush()
                    else:
                        dump_fp.write(line)
                        dump_fp.flush()
                except Exception as e:
                    logging.error(f"Failed to write trajectory: {e}")
            
            if done:
                env_done = True
                won = bool(info.get("won", False))
                final_info = info
                break
        
        # Save this attempt's trajectory
        attempt_final_info = info if isinstance(info, dict) else {}

        if trajectory_manager:
            trajectory_manager.save_attempt(env_id)
        
        attempt_data = {
            "retry_idx": retry_idx,
            "trajectory": trajectory_steps.copy(),
            "won": won,
            "reward": cumulative_reward,
            "steps": len(trajectory_steps)
        }
        
        attempt_metadata = {
            "environment": env_type,
            "attempt_index": retry_idx + 1,
            "max_steps": max_steps,
            "success": bool(won),
            "won": bool(won),
            "initial_observation": initial_observation,
            "final_observation": obs,
            "initial_info": _json_safe_copy(initial_info),
            "final_info": _json_safe_copy(attempt_final_info),
            "trajectory_length": len(trajectory_steps),
            "chat_history_length": len(chat_history),
            "timestamp": run_ts,
        }
        if replay_to_step >= 0:
            attempt_metadata["replay_to_step"] = replay_to_step
        attempt_data["metadata"] = attempt_metadata

        if continuous_instruction_manager and debugger_type == "continue":
            effective_guidance = guidance_list or continuous_instruction_manager.get_instructions(env_id)
            effective_history = guidance_history or continuous_instruction_manager.get_instruction_history(env_id)

            attempt_data["continuous_guidance"] = effective_guidance
            attempt_data["continuous_guidance_history"] = [
                {"attempt": attempt_idx, "instruction": guidance}
                for attempt_idx, guidance in effective_history
            ]

            overlay_snapshot = instruction_overlay or continuous_instruction_manager.format_instructions_for_observation(env_id)
            if overlay_snapshot:
                attempt_data["continuous_guidance_overlay"] = overlay_snapshot

            if follow_up_instruction:
                attempt_data["follow_up_instruction"] = follow_up_instruction

        all_attempt_trajectories.append(attempt_data)
        
        # Save individual attempt trajectory to task dir
        if task_dir:
            attempt_file = os.path.join(task_dir, f"attempt_{retry_idx + 1}_trajectory.json")
            with open(attempt_file, "w") as f:
                json.dump(attempt_data, f, indent=2)
        
        # Save current trajectory for potential debugging
        last_trajectory = trajectory_steps
        last_chat_history = list(chat_history)
        last_metadata = attempt_metadata
        final_reward = cumulative_reward
        
        # Check if this attempt was successful
        if won:
            logging.info(f"  Env {env_id} - SUCCESS on attempt {retry_idx + 1}")
            if retry_idx == 0:
                first_attempt_success = True
            break  # Success! No need to retry
        else:
            logging.info(f"  Env {env_id} - FAILED on attempt {retry_idx + 1}, will retry with debugging" if debugger and retry_idx < max_retries - 1 else f"  Env {env_id} - FAILED on attempt {retry_idx + 1}")
        
        # Clear replay mode after each attempt, but only if the attempt is complete
        # Don't clear if we're still in the middle of a replay sequence
        if debugger and analysis and env_done:
            env_manager.clear_replay(env_id)
            logging.info(f"    Cleared replay mode for env_id {env_id} after completed attempt")
        
        # If debugger is not enabled, don't retry
        if not debugger:
            break
    
    # Save final summary to task dir
    if task_dir:
        try:
            summary_file = os.path.join(task_dir, "task_summary.json")
            
            meta = {
                "batch_idx": batch_idx,
                "env_id": global_env_counter + env_id,
                "test_idx": test_idx,
                "model": agent.model_name,
                "env_type": env_type,
                "total_attempts": retry_idx + 1,
                "won": won,
                "first_attempt_success": first_attempt_success,
                "final_reward": final_reward,
                "timestamp": run_ts,
                "steps_in_final_attempt": len(last_trajectory) if last_trajectory else 0
            }
            
            # Add environment-specific metadata
            if env_type == "gaia":
                meta["pid"] = final_info.get("pid", "unknown")
            elif env_type == "alfworld":
                meta["gamefile"] = final_info.get("extra.gamefile", "")
            elif env_type == "webshop":
                meta["task_score"] = float(final_info.get("task_score", 0))
            
            # Save summary with all attempts info
            save_data = {
                "metadata": meta,
                "all_attempts_summary": [
                    {
                        "attempt": i + 1,
                        "won": att["won"],
                        "reward": att["reward"],
                        "steps": att["steps"]
                    }
                    for i, att in enumerate(all_attempt_trajectories)
                ]
            }

            if continuous_instruction_manager and debugger_type == "continue":
                save_data["continuous_guidance"] = continuous_instruction_manager.get_instructions(env_id)
                save_data["continuous_guidance_history"] = [
                    {"attempt": attempt_idx, "instruction": guidance}
                    for attempt_idx, guidance in continuous_instruction_manager.get_instruction_history(env_id)
                ]
            
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save task summary: {e}")
    
    env_manager.clear_persistent_guidance(env_id)

    result = {
        "env_id": env_id,
        "won": won,
        "first_attempt_success": first_attempt_success,
        "reward": final_reward,
        "retries": retry_idx + 1,
        "steps": len(last_trajectory) if last_trajectory else 0,
        "env_type": env_type,
        "trajectory": last_trajectory,
        "all_attempts": all_attempt_trajectories if save_all_attempts else None
    }

    if continuous_instruction_manager and debugger_type == "continue":
        result["continuous_guidance"] = continuous_instruction_manager.get_instructions(env_id)
        result["continuous_guidance_history"] = [
            {"attempt": attempt_idx, "instruction": guidance}
            for attempt_idx, guidance in continuous_instruction_manager.get_instruction_history(env_id)
        ]

    return result


def main():
    parser = argparse.ArgumentParser(description="Unified rollout script for multiple environments")
    
    # Environment selection
    parser.add_argument("--env", choices=["alfworld", "gaia", "webshop"], required=True,
                       help="Environment to run")
    
    # Common parameters
    parser.add_argument("--batch_size", type=int, default=10, 
                       help="Number of envs to process per batch")
    parser.add_argument("--total_envs", type=int, default=100, 
                       help="Total number of environments to rollout")
    parser.add_argument("--test_times", type=int, default=1,
                       help="Number of test runs per batch")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum steps per episode (default: 50 for alfworld, 30 for gaia/webshop)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=2)
    
    # Model parameters
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="Model name (OpenAI: gpt-4o, gpt-4o-mini; Together: Qwen/Qwen2.5-7B-Instruct-Turbo, etc.)")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--base_url", default=None,
                       help="OpenAI-compatible base URL (e.g., vLLM http://127.0.0.1:8000/v1)")
    
    # Execution parameters
    parser.add_argument("--concurrency", type=int, default=4,
                       help="Max concurrent task workers")
    parser.add_argument("--llm_concurrency", type=int, default=None,
                       help="Max concurrent LLM requests across all tasks (default: 3x task concurrency)")
    parser.add_argument("--retries", type=int, default=3,
                       help="Retries per request on failure")
    
    # Strategy selection
    parser.add_argument("--strategy", choices=["debugger", "bon", "tot", "dfsdt"], default="debugger",
                       help="Rollout strategy: LLM Debugger, Best-of-N, Tree-of-Thought DFS, or DFSDT")
    parser.add_argument("--bon_n", type=int, default=5, help="Best-of-N: number of independent rollouts")
    parser.add_argument("--beam_size", type=int, default=3, help="ToT/DFSDT: candidate branching per state")
    parser.add_argument("--value_threshold", type=float, default=0.15, help="ToT: prune if top value below threshold")
    parser.add_argument("--max_try", type=int, default=10, help="ToT/DFSDT/Debugger: maximum complete trajectory attempts")
    parser.add_argument("--diversity_back_steps", type=int, default=2, help="DFSDT: backtrack steps on failure")
    parser.add_argument("--diversity_back_steps_alt", type=int, default=3, help="DFSDT: alternate backtrack if needed")
    parser.add_argument("--propose_k", type=int, default=4, help="ToT/DFSDT: proposals when no explicit list")

    # Output parameters
    parser.add_argument("--dump_path", default=None,
                       help="If set, write JSONL trajectory to this file")
    parser.add_argument("--chat_root", default=None,
                       help="If set, save per-episode chat histories under this root")
    parser.add_argument("--experiment_dir", default=None,
                       help="Root directory for all experiment outputs")
    parser.add_argument("--save_per_task_trajectories", action="store_true",
                       help="Save each task's trajectories in a separate folder")
    
    # Environment-specific parameters
    parser.add_argument("--alf_env_type", default="alfworld/AlfredTWEnv",
                       help="AlfWorld environment type")
    parser.add_argument("--gaia_data_path", default="data/gaia/val.json",
                       help="Path to GAIA dataset")
    parser.add_argument("--gaia_tools", nargs='+', 
                       default=['google_search', 'wikipedia_knowledge_searcher', 'python_code_generator'],
                       help="List of available tools for GAIA")
    parser.add_argument("--webshop_train", action="store_true",
                       help="Use WebShop training set instead of test set")
    
    # Debugger options
    parser.add_argument("--enable_debugger", action="store_true",
                       help="Enable LLM debugger for failed trajectories")
    parser.add_argument("--debugger_type", choices=["naive", "advanced", "continue"], default="naive",
                       help="Select debugger implementation: naive heuristic, advanced API, or continue (cumulative guidance)")
    parser.add_argument("--max_debug_retry", type=int, default=None,
                       help="Deprecated: use --max_try instead. If set, overrides --max_try for debugger strategy.")
    parser.add_argument("--debugger_model", default="gpt-4o",
                       help="Model to use for trajectory debugging")
    parser.add_argument("--debugger_temperature", type=float, default=0.3,
                       help="Temperature for debugger model")
    parser.add_argument(
        "--debugger_base_url",
        default=None,
        help="OpenAI-compatible base URL for the debugger (defaults to --base_url if not specified)",
    )
    parser.add_argument(
        "--debugger_api_key",
        default=None,
        help="API key for the debugger client (defaults to OPENAI_API_KEY env var, use empty string for local vLLM)",
    )
    parser.add_argument("--debug_output_dir", default=None,
                       help="Directory to save debug analysis results")
    parser.add_argument("--save_all_attempts", action="store_true",
                       help="Save trajectories for all retry attempts")
    parser.add_argument("--debugger_capture_api_debug", action="store_true",
                        help="Include advanced debugger request/response payloads in outputs for troubleshooting")
    
    # Together AI routing
    parser.add_argument(
        "--together",
        choices=["rollout", "debugger", "both"],
        default=None,
        help=(
            "Route model calls to Together AI using TOGETHER_API_KEY. "
            "Choose which module to route: rollout, debugger, or both."
        ),
    )
    
    # Other options
    parser.add_argument("--unique_envs", action="store_true",
                       help="Ensure unique tasks/games across all environments")
    parser.add_argument(
        "--alfworld_task_ids_file",
        default=None,
        help=(
            "Optional path to a text file containing AlfWorld task identifiers (one per line).\n"
            "Accepted formats per line: (1) absolute path to game.tw-pddl; (2) trial directory name\n"
            "(e.g., 'trial_00003_T20190312_234237'). When provided with --unique_envs, tasks are\n"
            "loaded strictly in this order."
        ),
    )
    parser.add_argument("--start_index", type=int, default=0,
                       help="Starting offset into the task/game list for initial assignment across envs (0-based legacy)")
    parser.add_argument("--start_id", type=int, default=None,
                       help="1-based start offset into the task/game list (e.g., 2 skips the first task)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only print batch allocation without running")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set default max_steps based on environment
    if args.max_steps is None:
        args.max_steps = {
            "alfworld": 50,
            "gaia": 30,
            "webshop": 30
        }[args.env]
    
    # Unified max_try logic: use max_debug_retry if set (for backward compatibility), otherwise use max_try
    def get_max_retries():
        if args.max_debug_retry is not None:
            logging.warning("--max_debug_retry is deprecated, use --max_try instead")
            return args.max_debug_retry
        return args.max_try
    
    # Setup logging
    os.makedirs(f"logs/{args.env}", exist_ok=True)
    log_fp = os.path.join(
        f"logs/{args.env}",
        f"unified_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Redirect stdout and stderr to both terminal and log file
    stdout_tee = TeeOutput(log_fp, 'stdout')
    stderr_tee = TeeOutput(log_fp, 'stderr')
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    print(f"=== Log file: {log_fp} ===")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )
    
    logging.info(f"Starting unified rollout for {args.env}")
    logging.info(f"Model: {args.model}, Temperature: {args.temperature}")
    logging.info(f"Total envs: {args.total_envs}, Batch size: {args.batch_size}, Max steps: {args.max_steps}")
    
    # Resolve Together routing flags
    use_together_rollout = args.together in ("rollout", "both")
    use_together_debugger = args.together in ("debugger", "both")
    if args.together:
        logging.info(
            f"Together routing enabled: rollout={use_together_rollout}, debugger={use_together_debugger}"
        )
    
    # Calculate number of batches (deprecated: we keep a fixed env pool)
    num_batches = 1
    logging.info(f"Using fixed env pool; batch_size is ignored.")
    
    # Prepare experiment directory structure
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.experiment_dir:
        # Use the provided experiment directory
        experiment_root = args.experiment_dir
        os.makedirs(experiment_root, exist_ok=True)
        
        # Create subdirectories
        trajectories_dir = os.path.join(experiment_root, "trajectories")
        summaries_dir = os.path.join(experiment_root, "summaries")
        os.makedirs(trajectories_dir, exist_ok=True)
        os.makedirs(summaries_dir, exist_ok=True)
        
        # Set up dump file path
        if not args.dump_path:
            args.dump_path = os.path.join(summaries_dir, "all_trajectories.jsonl")
        
        logging.info(f"Experiment directory: {experiment_root}")
    else:
        trajectories_dir = None
        summaries_dir = None
    
    # Prepare output files
    dump_fp = None
    if args.dump_path:
        os.makedirs(os.path.dirname(args.dump_path) or ".", exist_ok=True)
        dump_fp = open(args.dump_path, "a", encoding="utf-8")
        logging.info(f"Dumping trajectories to: {args.dump_path}")

    # Pre-initialize Ray (once) for Ray-based envs to avoid thread-race on ray.init
    if args.env in ("alfworld", "webshop"):
        try:
            import os as _os
            _os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
            import ray as _ray
            if not _ray.is_initialized():
                _ray.init(ignore_reinit_error=True, include_dashboard=False)
        except Exception as e:
            logging.warning(f"Ray pre-initialization skipped or failed: {e}")
    
    def _sanitize(s: str) -> str:
        """Sanitize string for filename"""
        return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '-' for c in str(s))[:200]

    def _compute_start_offset(seq_len: int) -> Optional[int]:
        """Resolve the effective rotation offset for task sequences."""
        if seq_len <= 0:
            return None

        if args.start_id is not None:
            if args.start_id <= 0:
                logging.warning(f"start_id={args.start_id} is invalid; using 1 instead")
                return 0
            # start_id is 1-based; convert to 0-based offset
            offset = (args.start_id - 1) % seq_len
            logging.info(f"Applied start_id offset: {args.start_id} (effective {offset})")
            return offset

        if args.start_index:
            offset = args.start_index % seq_len
            logging.info(f"Applied start_index offset: {args.start_index} (effective {offset})")
            return offset

        return None

    # Prepare environment-specific data
    gaia_tasks = None
    alfworld_game_files = None
    
    if args.env == "gaia":
        logging.info(f"Loading GAIA tasks from {args.gaia_data_path}")
        gaia_tasks = load_gaia_tasks(args.gaia_data_path)
        logging.info(f"Loaded {len(gaia_tasks)} tasks")
        
        # Trim to what will actually be used by a fixed pool of envs
        pool_size = max(1, int(args.total_envs))
        rounds = max(1, int(args.test_times))
        max_needed = min(len(gaia_tasks), pool_size * rounds)
        if len(gaia_tasks) > max_needed:
            logging.info(f"Trimming GAIA tasks to {max_needed} (pool_size={pool_size}, rounds={rounds})")
            gaia_tasks = gaia_tasks[:max_needed]
        elif len(gaia_tasks) < max_needed:
            logging.warning(f"Only {len(gaia_tasks)} GAIA tasks available, reducing rounds to fit availability")
            rounds = max(1, len(gaia_tasks) // pool_size)
            args.test_times = rounds
        
        # Shuffle tasks for random sampling variety, then apply start offset rotation
        rng = random.Random(args.seed)
        rng.shuffle(gaia_tasks)
        offset = _compute_start_offset(len(gaia_tasks))
        if offset:
            gaia_tasks = gaia_tasks[offset:] + gaia_tasks[:offset]

    elif args.env == "alfworld" and args.unique_envs:
        # Helper to collect all available AlfWorld game files once
        def _collect_all_alfworld_gamefiles() -> List[str]:
            try:
                from openmanus_rl.environments.env_package.alfworld.envs import load_config_file as _alf_load_cfg
                from openmanus_rl.environments.env_package.alfworld.alfworld.agents.environment import get_environment as _alf_get_env
                _alf_cfg_path = os.path.join(
                    os.path.dirname(__file__),
                    '../../openmanus_rl/environments/env_package/alfworld/configs/config_tw.yaml'
                )
                _cfg = _alf_load_cfg(_alf_cfg_path)
                _BaseEnvCls = _alf_get_env(_cfg['env']['type'])
                _tmp_env = _BaseEnvCls(_cfg, train_eval='train')
                _tmp_env.collect_game_files()
                return list(getattr(_tmp_env, 'game_files', []) or [])
            except Exception as _e:
                logging.error(f"Failed to collect AlfWorld game files: {_e}")
                return []

        # If a task-IDs file is provided, honor its order.
        if args.alfworld_task_ids_file:
            logging.info(f"Loading AlfWorld task IDs from {args.alfworld_task_ids_file}")
            all_files = _collect_all_alfworld_gamefiles()
            # Map trial dir name -> gamefile path (unique)
            trial_to_path = {os.path.basename(os.path.dirname(p)): p for p in all_files}
            # Read requested IDs/paths
            req_ids: List[str] = []
            try:
                with open(args.alfworld_task_ids_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            req_ids.append(s)
            except Exception as e:
                logging.error(f"Failed to read task IDs file: {e}")
                sys.exit(1)

            resolved: List[str] = []
            for s in req_ids:
                if os.path.isfile(s):
                    resolved.append(s)
                else:
                    # Treat as trial directory name
                    p = trial_to_path.get(s)
                    if p is not None:
                        resolved.append(p)
                    else:
                        logging.warning(f"Unresolved AlfWorld task identifier: {s}")

            if not resolved:
                logging.error("No valid AlfWorld tasks resolved from the provided IDs. Aborting.")
                sys.exit(1)

            alfworld_game_files = resolved

            # Fit rounds to available tasks for the fixed env pool size
            pool_size_est = max(1, int(args.total_envs))
            max_rounds_by_files = len(alfworld_game_files) // pool_size_est
            if max_rounds_by_files <= 0:
                logging.error("Not enough AlfWorld game files from IDs to allocate one per env. Aborting.")
                sys.exit(1)
            if args.test_times > max_rounds_by_files:
                logging.warning(
                    f"Reducing test_times from {args.test_times} to {max_rounds_by_files} to fit provided IDs"
                )
                args.test_times = max_rounds_by_files
            logging.info(f"Prepared {len(alfworld_game_files)} game files from IDs (no rotation applied)")

        else:
            # Default path: collect, then slice deterministically with optional rotation
            total_needed = max(1, int(args.total_envs)) * max(1, int(args.test_times))
            alfworld_game_files = prepare_alfworld_game_files(args.env, total_needed, args.seed)
            if alfworld_game_files:
                # Apply start offset rotation for initial assignment if requested
                offset = _compute_start_offset(len(alfworld_game_files))
                if offset:
                    alfworld_game_files = alfworld_game_files[offset:] + alfworld_game_files[:offset]
                logging.info(f"Prepared {len(alfworld_game_files)} unique game files")
                # If not enough files for requested rounds, reduce rounds to avoid repetition
                pool_size_est = max(1, int(args.total_envs))
                max_rounds_by_files = len(alfworld_game_files) // pool_size_est
                if max_rounds_by_files <= 0:
                    logging.error("Not enough AlfWorld game files to allocate one per env. Aborting.")
                    sys.exit(1)
                if args.test_times > max_rounds_by_files:
                    logging.warning(
                        f"Reducing test_times from {args.test_times} to {max_rounds_by_files} to avoid task repetition"
                    )
                    args.test_times = max_rounds_by_files
    
    # Dry run mode
    if args.dry_run:
        logging.info(f"[Dry-Run] Environment: {args.env}")
        logging.info(f"[Dry-Run] Total envs: {args.total_envs}, Batches: {num_batches}")
        
        for b in range(num_batches):
            start = b * args.batch_size
            end = min(start + args.batch_size, args.total_envs)
            batch_size = end - start
            
            if args.env == "gaia" and gaia_tasks:
                batch_tasks = gaia_tasks[start:end]
                pids = [t.get('pid', f'task_{i}') for i, t in enumerate(batch_tasks)]
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} tasks; PIDs: {', '.join(pids[:3])}...")
            elif args.env == "alfworld" and alfworld_game_files:
                batch_files = alfworld_game_files[start:end]
                # Show trial directory names for clarity instead of the repeated filename 'game.tw-pddl'
                examples = [os.path.basename(os.path.dirname(f)) for f in batch_files[:3]]
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} files; Trials: {', '.join(examples)}")
            else:
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} environments")
        
        sys.exit(0)
    
    # Initialize agent (defer until after potential dry-run exit to avoid requiring API keys)
    agent = UnifiedAgent(
        model_name=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        env_type=args.env,
        use_together=use_together_rollout,
    )
    
    # Create shared LLM executor pool for better concurrency across all tasks
    # Use more workers than task concurrency to handle LLM I/O wait time
    if args.llm_concurrency is not None:
        llm_pool_size = args.llm_concurrency
    else:
        llm_pool_size = min(50, args.concurrency * 3)  # 3x task concurrency for LLM calls
    
    shared_llm_executor = ThreadPoolExecutor(max_workers=llm_pool_size)
    logging.info(f"Created shared LLM executor pool with {llm_pool_size} workers")
    
    # Initialize debugger and trajectory manager if enabled
    debugger = None
    trajectory_manager = None
    continuous_instruction_manager = None
    if args.enable_debugger and args.strategy == "debugger":
        debugger_type_label = args.debugger_type
        # Use debugger_base_url if provided; otherwise fall back to rollout --base_url
        debugger_base_url = args.debugger_base_url
        if args.debugger_type == "advanced":
            try:
                debugger = AdvancedDebugger(
                    model_name=args.debugger_model,
                    temperature=args.debugger_temperature,
                    base_url=debugger_base_url,
                    api_key=args.debugger_api_key,
                    analysis_model=args.debugger_model,
                    capture_debug_data=args.debugger_capture_api_debug,
                )
            except Exception as exc:
                logging.error(f"Failed to initialize advanced debugger: {exc}")
                raise
        else:
            debugger = LLMDebugger(
                model_name=args.debugger_model,
                temperature=args.debugger_temperature,
                base_url=debugger_base_url,
                api_key=args.debugger_api_key,
                use_together=use_together_debugger,
            )
            if args.debugger_type != "continue":
                debugger_type_label = "naive"
        
        trajectory_manager = TrajectoryManager()
        
        # Initialize continuous instruction manager for continue mode
        if args.debugger_type == "continue":
            continuous_instruction_manager = ContinuousInstructionManager()
            debugger_type_label = "continue (cumulative guidance)"
        logging.info(
            "Debugger enabled (%s)\n"
            "  Rollout: model=%s, base_url=%s\n"
            "  Debugger: model=%s, base_url=%s\n"
            "  Max retries: %s",
            debugger_type_label,
            args.model,
            args.base_url or "(default OpenAI)",
            args.debugger_model,
            debugger_base_url or "(default OpenAI)",
            get_max_retries(),
        )
        
        # Create debug output directory if specified
        if args.debug_output_dir:
            os.makedirs(args.debug_output_dir, exist_ok=True)
            logging.info(f"Debug analysis will be saved to: {args.debug_output_dir}")

    # Statistics tracking
    all_overall_success_rates = []
    all_first_attempt_success_rates = []  # Track first attempt success rates
    all_debugger_success_rates = []  # Track success rates after debugger assistance
    all_task_success_history = defaultdict(list)
    global_env_counter = 0
    
    # Track overall statistics
    total_first_attempt_successes = 0
    total_debugger_successes = 0
    total_tasks = 0
    
    # Main rollout loop
    try:
        # Legacy batch loop disabled — we use a fixed env pool below.
        for batch_idx in range(0):
            # Calculate actual batch size
            current_batch_size = min(args.batch_size, args.total_envs - batch_idx * args.batch_size)
            logging.info(f"\n========== Starting Batch {batch_idx + 1}/{num_batches} with {current_batch_size} envs ==========")
            
            # Prepare environment-specific kwargs
            env_kwargs = {
                "env_num": current_batch_size,
                "seed": args.seed + batch_idx,
                "history_length": args.history_length,
            }
            
            if args.env == "gaia":
                start = batch_idx * args.batch_size
                end = start + current_batch_size
                env_kwargs["tasks_data"] = gaia_tasks[start:end]
                env_kwargs["available_tools"] = args.gaia_tools
                env_kwargs["max_steps"] = args.max_steps
                
            elif args.env == "alfworld":
                env_kwargs["alf_env_type"] = args.alf_env_type
                if alfworld_game_files:
                    start = batch_idx * args.batch_size
                    end = start + current_batch_size
                    env_kwargs["game_files"] = alfworld_game_files[start:end]
                    
            elif args.env == "webshop":
                env_kwargs["use_train_set"] = args.webshop_train
            
            # Batch-level statistics
            batch_overall_success_rates = []
            batch_task_success_history = defaultdict(list)
            
            try:
                # Test loop for this batch
                for test_idx in range(args.test_times):
                    logging.info(f"\n========== Start Batch {batch_idx + 1} Test {test_idx} ==========")
                    start_time = time.time()
                    
                    # Run with retry logic if debugger is enabled
                    if args.enable_debugger:
                        # Build per-env managers (env_num == 1) so each rollout can reset independently
                        per_env_build_args = []
                        for i in range(current_batch_size):
                            single_kwargs = dict(env_kwargs)
                            single_kwargs["env_num"] = 1
                            # Slice GAIA tasks to a single task
                            if args.env == "gaia" and "tasks_data" in single_kwargs:
                                # Select the i-th task within this batch
                                start = batch_idx * args.batch_size
                                single_kwargs["tasks_data"] = [gaia_tasks[start + i]]
                            # Pin one gamefile for AlfWorld if provided
                            if args.env == "alfworld" and "game_files" in single_kwargs and single_kwargs["game_files"]:
                                single_kwargs["game_files"] = [single_kwargs["game_files"][i]]
                            per_env_build_args.append(single_kwargs)

                        # Helper to run a single rollout in its own env
                        def _run_one(env_idx: int):
                            task_start_time = time.time()
                            thread_id = threading.get_ident()
                            logging.info(f"[PARALLEL] Task {env_idx + 1} starting on thread {thread_id} at {task_start_time:.3f}")

                            # Create one-env manager for this rollout
                            env_init_start = time.time()
                            local_env = EnvironmentFactory.build_env(
                                args.env,
                                with_debugger=True,
                                **per_env_build_args[env_idx]
                            )
                            env_init_time = time.time() - env_init_start
                            logging.info(f"[PARALLEL] Task {env_idx + 1} env init took {env_init_time:.3f}s")

                            try:
                                # Reset once to compute task id and make task dir
                                reset_start = time.time()
                                init_obs, init_infos = local_env.reset()
                                info0 = init_infos[0] if isinstance(init_infos, list) else init_infos
                                task_id_local = get_task_id(args.env, env_idx, info0, batch_idx)
                                task_dir_local = None
                                if args.save_per_task_trajectories and trajectories_dir:
                                    task_dir_local = os.path.join(trajectories_dir, _sanitize(task_id_local))
                                    os.makedirs(task_dir_local, exist_ok=True)
                                reset_time = time.time() - reset_start
                                logging.info(f"[PARALLEL] Task {env_idx + 1} reset took {reset_time:.3f}s")

                                logging.info(f"[PARALLEL] Task {env_idx + 1}/{current_batch_size} in Batch {batch_idx + 1} - {task_id_local}")

                                # Execute rollout per strategy; env_id is 0 for single-env managers
                                rollout_start = time.time()
                                if args.strategy == "debugger":
                                    res = run_environment_with_retry(
                                        env_id=0,
                                        env_manager=local_env,
                                        agent=agent,
                                        max_steps=args.max_steps,
                                        env_type=args.env,
                                        debugger=debugger,
                                        trajectory_manager=trajectory_manager,
                                        max_retries=get_max_retries(),
                                        dump_fp=dump_fp,
                                        dump_lock=dump_lock,
                                        chat_base_dir=None,
                                        batch_idx=batch_idx,
                                        test_idx=test_idx,
                                        global_env_counter=global_env_counter + env_idx,
                                        run_ts=run_ts,
                                        debug_output_dir=None,
                                        save_all_attempts=args.save_all_attempts,
                                        task_dir=task_dir_local,
                                        shared_llm_executor=shared_llm_executor,
                                        continuous_instruction_manager=continuous_instruction_manager,
                                        debugger_type=args.debugger_type,
                                    )
                                elif args.strategy == "bon":
                                    # Helper closure to attempt a single rollout (no debugger, one attempt)
                                    def _single_attempt(attempt_idx: int):
                                        # Create attempt-specific task directory
                                        attempt_task_dir = None
                                        if task_dir_local:
                                            attempt_task_dir = os.path.join(task_dir_local, f"attempt_{attempt_idx}")
                                            os.makedirs(attempt_task_dir, exist_ok=True)
                                        
                                        return run_environment_with_retry(
                                            env_id=0,
                                            env_manager=local_env,
                                            agent=agent,
                                            max_steps=args.max_steps,
                                            env_type=args.env,
                                            debugger=None,
                                            trajectory_manager=None,
                                            max_retries=1,
                                            dump_fp=dump_fp,
                                            dump_lock=dump_lock,
                                            chat_base_dir=None,
                                            batch_idx=batch_idx,
                                            test_idx=test_idx,
                                            global_env_counter=global_env_counter + env_idx,
                                            run_ts=run_ts,
                                            debug_output_dir=None,
                                            save_all_attempts=False,
                                            task_dir=attempt_task_dir,
                                            shared_llm_executor=shared_llm_executor,
                                            continuous_instruction_manager=None,
                                            debugger_type="naive",
                                        )
                                    res = run_best_of_n(
                                        N=args.bon_n,
                                        env_manager=local_env,
                                        agent=agent,
                                        max_steps=args.max_steps,
                                        env_type=args.env,
                                        single_attempt_fn=_single_attempt,
                                        task_dir=task_dir_local,
                                    )
                                else:
                                    # ToT-DFS or DFSDT: run up to max_try independent search attempts (one file per attempt)
                                    sp = SearchParams(
                                        beam_size=args.beam_size,
                                        value_threshold=args.value_threshold,
                                        max_try=1,  # single trajectory per call
                                        max_depth=args.max_steps,
                                        diversity_back_steps=args.diversity_back_steps,
                                        diversity_back_steps_alt=args.diversity_back_steps_alt,
                                        propose_k=args.propose_k,
                                    )
                                    mode = "tot" if args.strategy == "tot" else "dfsdt"
                                    res = None
                                    # Optionally clear task dir (already cleared earlier when created)
                                    for attempt_idx in range(1, args.max_try + 1):
                                        logging.info(f"[{mode.upper()}] Starting trajectory attempt {attempt_idx}/{args.max_try}")
                                        sr = run_tree_search(
                                            env_manager=local_env,
                                            agent=agent,
                                            max_steps=args.max_steps,
                                            env_type=args.env,
                                            params=sp,
                                            mode=mode,
                                            task_dir=task_dir_local if args.save_per_task_trajectories else None,
                                            attempt_id=attempt_idx if args.save_per_task_trajectories else None,
                                        )
                                        if res is None:
                                            res = sr.copy()
                                            res["search_attempts"] = []
                                        res["search_attempts"].append({
                                            "won": sr.get("won", False),
                                            "steps": sr.get("steps", 0),
                                            "strategy": mode,
                                        })
                                        if sr.get("won", False):
                                            logging.info(f"[{mode.upper()}] SUCCESS on attempt {attempt_idx}")
                                            res = sr
                                            break
                                rollout_time = time.time() - rollout_start
                                total_time = time.time() - task_start_time
                                logging.info(f"[PARALLEL] Task {env_idx + 1} rollout took {rollout_time:.3f}s, total time: {total_time:.3f}s")

                                res["task_id"] = task_id_local
                                res["timing"] = {
                                    "env_init_time": env_init_time,
                                    "reset_time": reset_time,
                                    "rollout_time": rollout_time,
                                    "total_time": total_time,
                                    "thread_id": thread_id,
                                }
                                # If ToT/DFSDT, refresh summary file to include search result
                                if args.strategy in ("tot", "dfsdt") and task_dir_local:
                                    try:
                                        summary_file = os.path.join(task_dir_local, "task_summary.json")
                                        meta = {
                                            "model": agent.model_name,
                                            "env_type": args.env,
                                            "strategy": args.strategy,
                                            "total_attempts": len(res.get("search_attempts", [])),
                                            "won": bool(res.get("won", False)),
                                            "timestamp": run_ts,
                                        }
                                        with open(summary_file, "w", encoding="utf-8") as f:
                                            json.dump({
                                                "metadata": meta,
                                                "search_attempts": res.get("search_attempts", []),
                                                "final_result": {"won": res.get("won", False), "steps": res.get("steps", 0)},
                                            }, f, ensure_ascii=False, indent=2)
                                    except Exception as e:
                                        logging.debug(f"Failed to write updated summary: {e}")

                                # Preserve original batch env_id in result for consistency
                                res["env_id"] = env_idx
                                return res
                            finally:
                                # Ensure resources for this single env are released
                                cleanup_start = time.time()
                                try:
                                    local_env.envs.close()
                                    cleanup_time = time.time() - cleanup_start
                                    logging.info(f"[PARALLEL] Task {env_idx + 1} cleanup took {cleanup_time:.3f}s")
                                except Exception as e:
                                    logging.warning(f"[PARALLEL] Task {env_idx + 1} cleanup failed: {e}")

                        # Run all envs in parallel using a thread pool (LLM calls are also IO-bound)
                        dump_lock = threading.Lock() if dump_fp is not None else None
                        env_results = []
                        
                        batch_parallel_start = time.time()
                        logging.info(f"[PARALLEL] Starting parallel execution of {current_batch_size} tasks with {args.concurrency} workers")
                        
                        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                            futures = [ex.submit(_run_one, i) for i in range(current_batch_size)]
                            logging.info(f"[PARALLEL] All {len(futures)} tasks submitted to thread pool")
                            
                            completed_count = 0
                            for fut in as_completed(futures):
                                result = fut.result()
                                env_results.append(result)
                                completed_count += 1
                                logging.info(f"[PARALLEL] Task completed ({completed_count}/{current_batch_size}): Task {result['env_id'] + 1} in {result['timing']['total_time']:.3f}s")
                        
                        batch_parallel_time = time.time() - batch_parallel_start
                        logging.info(f"[PARALLEL] All {current_batch_size} tasks completed in {batch_parallel_time:.3f}s")
                        
                        # Analyze parallel performance
                        if env_results and "timing" in env_results[0]:
                            total_task_times = [r["timing"]["total_time"] for r in env_results]
                            avg_task_time = np.mean(total_task_times)
                            max_task_time = np.max(total_task_times)
                            min_task_time = np.min(total_task_times)
                            theoretical_sequential_time = sum(total_task_times)
                            parallel_efficiency = theoretical_sequential_time / (batch_parallel_time * current_batch_size) if batch_parallel_time > 0 else 0
                            
                            logging.info(f"[PARALLEL] Performance Analysis:")
                            logging.info(f"  Average task time: {avg_task_time:.3f}s")
                            logging.info(f"  Max task time: {max_task_time:.3f}s (bottleneck)")
                            logging.info(f"  Min task time: {min_task_time:.3f}s")
                            logging.info(f"  Theoretical sequential time: {theoretical_sequential_time:.3f}s")
                            logging.info(f"  Actual parallel time: {batch_parallel_time:.3f}s")
                            logging.info(f"  Speedup: {theoretical_sequential_time/batch_parallel_time:.2f}x")
                            logging.info(f"  Parallel efficiency: {parallel_efficiency:.2f} ({parallel_efficiency*100:.1f}%)")
                        
                        # Collect statistics from results
                        overall_success_this_round = np.array([r['won'] for r in env_results])
                        first_attempt_success_this_round = np.array([r['first_attempt_success'] for r in env_results])
                        task_success_cnt = defaultdict(int)
                        task_total_cnt = defaultdict(int)
                        
                        # Update overall statistics
                        total_tasks += len(env_results)
                        total_first_attempt_successes += first_attempt_success_this_round.sum()
                        total_debugger_successes += overall_success_this_round.sum()
                        
                        # Process results for task-specific statistics
                        for result in env_results:
                            task_id = result.get('task_id', f"task_{result['env_id']}")
                            task_total_cnt[task_id] = 1
                            if result['won']:
                                task_success_cnt[task_id] = 1
                        
                        # Calculate success rates
                        round_success_rate = overall_success_this_round.mean()
                        first_attempt_rate = first_attempt_success_this_round.mean()
                        
                        batch_overall_success_rates.append(round_success_rate)
                        all_first_attempt_success_rates.append(first_attempt_rate)
                        all_debugger_success_rates.append(round_success_rate)
                        
                        logging.info(f"Batch {batch_idx + 1} Test {test_idx} Results:")
                        logging.info(f"  First attempt success rate: {first_attempt_rate:.4f}")
                        logging.info(f"  Success rate after debugger: {round_success_rate:.4f}")
                        
                        # Log per-task results if needed
                        for task, total in task_total_cnt.items():
                            if total > 0:
                                rate = task_success_cnt.get(task, 0) / total
                                batch_task_success_history[task].append(rate)
                        
                        logging.info(f"Batch {batch_idx + 1} Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n")
                        continue  # Skip the normal rollout code below
                    # Normal rollout without debugger (original code)
                    env_manager = EnvironmentFactory.build_env(
                        args.env,
                        with_debugger=False,
                        **env_kwargs
                    )
                    obs, infos = env_manager.reset()
                    env_dones = [False] * current_batch_size
                    
                    # Set chat_base_dir from args
                    chat_base_dir = args.chat_root
                    
                    # Per-env chat buffers
                    chats = [[] for _ in range(current_batch_size)]
                    saved_flags = [False] * current_batch_size
                    last_infos = infos
                    
                    # Statistics for single round
                    overall_success_this_round = np.zeros(current_batch_size, dtype=bool)
                    task_success_cnt = defaultdict(int)
                    task_total_cnt = defaultdict(int)
                    
                    for step_idx in range(args.max_steps):
                        logging.info(f"Batch {batch_idx + 1} Step {step_idx}; Dones ({np.array(env_dones).sum()}/{current_batch_size}); SR {overall_success_this_round.mean():.3f}")
                        
                        # Assemble actions
                        prompts = []
                        idx_map = []
                        for i in range(current_batch_size):
                            if not env_dones[i]:
                                prompts.append(obs["text"][i])
                                idx_map.append(i)
                        
                        if not prompts:
                            break
                        
                        batch_actions = agent.get_actions_batch(
                            prompts, 
                            concurrency=args.concurrency, 
                            retries=args.retries
                        )
                        
                        actions = ["None"] * current_batch_size
                        for k, i in enumerate(idx_map):
                            actions[i] = batch_actions[k]
                        
                        # Environment stepping
                        prev_prompts = obs["text"]
                        raw_actions = actions.copy()
                        obs, rewards, dones, infos = env_manager.step(actions.copy())
                        last_infos = infos
                        
                        # Process results
                        for i in range(current_batch_size):
                            if env_dones[i]:
                                continue
                            
                            # Append chat history
                            if prev_prompts and i < len(prev_prompts):
                                chats[i].append({"role": "user", "content": prev_prompts[i]})
                            chats[i].append({"role": "assistant", "content": raw_actions[i]})
                            
                            # Dump trajectory
                            if args.dump_path and (i in idx_map):
                                try:
                                    row = {
                                        "batch_idx": batch_idx,
                                        "test_idx": test_idx,
                                        "step": step_idx,
                                        "env_id": global_env_counter + i,
                                        "prompt": prev_prompts[i],
                                        "action": raw_actions[i],
                                        "reward": float(rewards[i]) if i < len(rewards) else None,
                                        "done": bool(dones[i]) if i < len(dones) else None,
                                        "won": bool(infos[i].get("won", False)),
                                        "is_action_valid": bool(infos[i].get("is_action_valid", False)),
                                    }
                                    
                                    # Add environment-specific fields
                                    if args.env == "gaia":
                                        row["pid"] = infos[i].get("pid", "unknown")
                                    elif args.env == "alfworld":
                                        row["gamefile"] = infos[i].get("extra.gamefile", "")
                                    elif args.env == "webshop":
                                        row["task_score"] = float(infos[i].get("task_score", 0))
                                    
                                    dump_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                                except Exception as e:
                                    logging.debug(f"Dump error: {e}")
                            
                            # Check if done
                            if dones[i]:
                                env_dones[i] = True
                                won = bool(infos[i].get("won", False))
                                overall_success_this_round[i] = won
                                
                                # Track task success
                                if args.env == "gaia":
                                    task_id = infos[i].get("pid", f"task_{i}")
                                elif args.env == "alfworld":
                                    gamefile = infos[i].get("extra.gamefile", "")
                                    # Extract task type from gamefile
                                    task_types = ["pick_and_place", "pick_two_obj_and_place", 
                                                 "look_at_obj_in_light", "pick_heat_then_place_in_recep",
                                                 "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"]
                                    task_id = "other"
                                    for t in task_types:
                                        if t in gamefile:
                                            task_id = t
                                            break
                                else:  # webshop
                                    task_id = f"task_{i}"
                                
                                task_total_cnt[task_id] = 1
                                if won:
                                    task_success_cnt[task_id] = 1
                                
                                # Save chat history
                                if chat_base_dir and not saved_flags[i]:
                                    try:
                                        task_hash = hashlib.sha1(str(task_id).encode()).hexdigest()[:8]
                                        unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}-{task_hash}"
                                        out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")
                                        
                                        meta = {
                                            "batch_idx": batch_idx,
                                            "env_id": global_env_counter + i,
                                            "test_idx": test_idx,
                                            "model": args.model,
                                            "steps": step_idx + 1,
                                            "won": won,
                                            "timestamp": run_ts,
                                            "environment": args.env,
                                        }
                                        
                                        with open(out_path, "w", encoding="utf-8") as f:
                                            json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                        saved_flags[i] = True
                                    except Exception as e:
                                        logging.debug(f"Failed to save chat: {e}")
                        
                        if all(env_dones):
                            logging.info("All environments finished early!")
                            break
                    
                    # Save any unfinished chats
                    if chat_base_dir:
                        for i in range(current_batch_size):
                            if not saved_flags[i]:
                                try:
                                    task_hash = hashlib.sha1(f"unfinished_{i}".encode()).hexdigest()[:8]
                                    unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}-{task_hash}"
                                    out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")
                                    
                                    meta = {
                                        "batch_idx": batch_idx,
                                        "env_id": global_env_counter + i,
                                        "test_idx": test_idx,
                                        "model": args.model,
                                        "steps": len(chats[i]) // 2,
                                        "won": False,
                                        "timestamp": run_ts,
                                        "environment": args.env,
                                    }
                                    
                                    with open(out_path, "w", encoding="utf-8") as f:
                                        json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                    saved_flags[i] = True
                                except Exception as e:
                                    logging.debug(f"Failed to save unfinished chat: {e}")
                    
                    # Round statistics
                    round_success_rate = overall_success_this_round.mean()
                    batch_overall_success_rates.append(round_success_rate)
                    
                    logging.info(f"Batch {batch_idx + 1} Test {test_idx} overall success: {round_success_rate:.4f}")
                    
                    # Calculate and store per-task success rates for this test
                    for task, total in task_total_cnt.items():
                        if total > 0:
                            rate = task_success_cnt.get(task, 0) / total
                            batch_task_success_history[task].append(rate)
                            
                            # Log task-specific results for alfworld
                            if args.env == "alfworld":
                                logging.info(f"    {task:<35s}: {rate:.4f} ({task_success_cnt.get(task, 0)}/{task_total_cnt[task]})")
                    
                    logging.info(f"Batch {batch_idx + 1} Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n")
                
            finally:
                # Accumulate batch results
                all_overall_success_rates.extend(batch_overall_success_rates)
                for task, rates in batch_task_success_history.items():
                    all_task_success_history[task].extend(rates)
                
                # Update global counter
                global_env_counter += current_batch_size
                
                # Clean up resources for non-debugger batch manager (if any)
                try:
                    if 'env_manager' in locals() and hasattr(env_manager, 'envs'):
                        env_manager.envs.close()
                        logging.info(f"Released resources for Batch {batch_idx + 1}")
                except Exception as e:
                    logging.warning(f"Failed to release resources: {e}")
                
                logging.info(f"========== Finished Batch {batch_idx + 1}/{num_batches}, processed {global_env_counter}/{args.total_envs} envs ==========\n")

        # ===== Fixed-pool parallel rollout (preferred path) =====
        pool_size = max(1, int(args.total_envs))
        rounds = max(1, int(args.test_times))

        logging.info(f"\n========== Fixed Env Pool ==========")
        logging.info(f"Parallel envs: {pool_size} | Rounds per env: {rounds}")

        if args.strategy != "debugger" or args.enable_debugger:
            # Build per-env managers (single-env each) once and reuse with reset()
            common_kwargs = {"env_num": 1, "seed": args.seed, "history_length": args.history_length}
            if args.env == "gaia":
                common_kwargs["available_tools"] = args.gaia_tools
                common_kwargs["max_steps"] = args.max_steps
                # Distribute trimmed tasks across envs
                per_env_tasks: List[List[Dict]] = [[] for _ in range(pool_size)]
                for k, task in enumerate(gaia_tasks or []):
                    per_env_tasks[k % pool_size].append(task)
            elif args.env == "alfworld":
                common_kwargs["alf_env_type"] = args.alf_env_type
                # For AlfWorld with unique_envs, we allocate per-round files on the fly
                # so we skip building a persistent env_pool below.
            elif args.env == "webshop":
                common_kwargs["use_train_set"] = args.webshop_train

            env_pool = []
            # For GAIA/WEBSHOP (and AlfWorld without unique_envs), prebuild persistent envs
            if not (args.env == "alfworld" and args.unique_envs):
                for i in range(pool_size):
                    kwargs_i = dict(common_kwargs)
                    if args.env == "gaia":
                        kwargs_i["tasks_data"] = per_env_tasks[i] if gaia_tasks is not None else []
                    if args.env == "alfworld":
                        # Non-unique mode: use default sampling within env
                        pass
                    mgr = EnvironmentFactory.build_env(args.env, with_debugger=True, **kwargs_i)
                    env_pool.append(mgr)

            # Prepare per-env task assignments and persist to disk
            assignments_root: Optional[str] = None
            if args.experiment_dir:
                assignments_root = os.path.join(args.experiment_dir, "assignments")
                os.makedirs(assignments_root, exist_ok=True)

            per_env_assigned_ids: List[List[str]] = [[] for _ in range(pool_size)]
            per_env_assigned_payloads: List[List[Any]] = [[] for _ in range(pool_size)]

            if args.env == "alfworld" and args.unique_envs and alfworld_game_files:
                total_jobs = min(len(alfworld_game_files), pool_size * rounds)
                files = alfworld_game_files[:total_jobs]
                for r in range(rounds):
                    start_idx = r * pool_size
                    end_idx = start_idx + pool_size
                    slice_files = files[start_idx:end_idx]
                    if not slice_files:
                        break
                    for i, gf in enumerate(slice_files):
                        per_env_assigned_payloads[i].append(gf)
                        per_env_assigned_ids[i].append(os.path.basename(gf))
            elif args.env == "gaia" and gaia_tasks is not None:
                # per_env_tasks already built; record IDs
                for i in range(pool_size):
                    tasks_i = per_env_tasks[i]
                    per_env_assigned_payloads[i] = tasks_i
                    per_env_assigned_ids[i] = [str(t.get("pid", f"task_{k}")) for k, t in enumerate(tasks_i)]
            elif args.env == "webshop":
                # WebShop does not expose task slicing; record placeholders
                for i in range(pool_size):
                    per_env_assigned_ids[i] = [f"round_{r+1}" for r in range(rounds)]
                    per_env_assigned_payloads[i] = [None] * rounds

            # Save task_ids.txt per env
            if assignments_root is not None:
                for i in range(pool_size):
                    env_dir = os.path.join(assignments_root, f"env_{i+1:03d}")
                    try:
                        os.makedirs(env_dir, exist_ok=True)
                        # Prefer human-readable unique IDs (trial directory names) for AlfWorld
                        if args.env == "alfworld":
                            per_env_assigned_ids[i] = [
                                os.path.basename(os.path.dirname(p)) for p in per_env_assigned_payloads[i]
                            ]
                        # Write IDs
                        fp_ids = os.path.join(env_dir, "task_ids.txt")
                        with open(fp_ids, "w", encoding="utf-8") as f:
                            for tid in per_env_assigned_ids[i]:
                                f.write(str(tid) + "\n")
                        # Also write full paths for reproducibility
                        if per_env_assigned_payloads[i]:
                            fp_paths = os.path.join(env_dir, "task_paths.txt")
                            with open(fp_paths, "w", encoding="utf-8") as f:
                                for p in per_env_assigned_payloads[i]:
                                    f.write(str(p) + "\n")
                    except Exception as exc:
                        logging.warning(f"Failed to write assignments for env {i}: {exc}")

            def _run_one_round(env_idx: int, round_idx: int, override_gamefile: Optional[str] = None):
                gamefile: Optional[str] = None

                # For AlfWorld unique mode, build a fresh single-env with a distinct gamefile per run
                if args.env == "alfworld" and args.unique_envs and alfworld_game_files:
                    if override_gamefile is not None:
                        gamefile = override_gamefile
                    else:
                        file_idx = round_idx * pool_size + env_idx
                        if file_idx >= len(alfworld_game_files):
                            logging.warning(
                                f"Requested AlfWorld game file index {file_idx} but only {len(alfworld_game_files)} available"
                            )
                            return {}
                        gamefile = alfworld_game_files[file_idx]

                    kwargs_i = dict(common_kwargs)
                    kwargs_i["game_files"] = [gamefile]
                    local_mgr = EnvironmentFactory.build_env("alfworld", with_debugger=True, **kwargs_i)
                    init_obs, init_infos = local_mgr.reset()
                else:
                    # Reset to get task id and ensure fresh episode
                    local_mgr = env_pool[env_idx]
                    init_obs, init_infos = local_mgr.reset()
                info0 = init_infos[0] if isinstance(init_infos, list) else init_infos
                task_id = get_task_id(args.env, env_idx, info0, round_idx)

                task_dir = None
                if trajectories_dir and args.save_per_task_trajectories:
                    task_dir = os.path.join(trajectories_dir, _sanitize(task_id))
                    os.makedirs(task_dir, exist_ok=True)

                if args.strategy == "debugger":
                    res = run_environment_with_retry(
                        env_id=0,
                        env_manager=local_mgr,
                        agent=agent,
                        max_steps=args.max_steps,
                        env_type=args.env,
                        debugger=debugger,
                        trajectory_manager=trajectory_manager,
                        max_retries=get_max_retries(),
                        dump_fp=dump_fp,
                        dump_lock=(threading.Lock() if dump_fp is not None else None),
                        chat_base_dir=None,
                        batch_idx=0,
                        test_idx=round_idx,
                        global_env_counter=env_idx,
                        run_ts=run_ts,
                        debug_output_dir=None,
                        save_all_attempts=args.save_all_attempts,
                        task_dir=task_dir,
                        shared_llm_executor=shared_llm_executor,
                        continuous_instruction_manager=continuous_instruction_manager,
                        debugger_type=args.debugger_type,
                    )
                elif args.strategy == "bon":
                    def _single_attempt(attempt_idx: int):
                        # Create attempt-specific task directory
                        attempt_task_dir = None
                        if task_dir:
                            attempt_task_dir = os.path.join(task_dir, f"attempt_{attempt_idx}")
                            os.makedirs(attempt_task_dir, exist_ok=True)
                        
                        return run_environment_with_retry(
                            env_id=0,
                            env_manager=local_mgr,
                            agent=agent,
                            max_steps=args.max_steps,
                            env_type=args.env,
                            debugger=None,
                            trajectory_manager=None,
                            max_retries=1,
                            dump_fp=dump_fp,
                            dump_lock=(threading.Lock() if dump_fp is not None else None),
                            chat_base_dir=None,
                            batch_idx=0,
                            test_idx=round_idx,
                            global_env_counter=env_idx,
                            run_ts=run_ts,
                            debug_output_dir=None,
                            save_all_attempts=False,
                            task_dir=attempt_task_dir,
                            shared_llm_executor=shared_llm_executor,
                            continuous_instruction_manager=None,
                            debugger_type="naive",
                        )
                    res = run_best_of_n(
                        N=args.bon_n,
                        env_manager=local_mgr,
                        agent=agent,
                        max_steps=args.max_steps,
                        env_type=args.env,
                        single_attempt_fn=_single_attempt,
                        task_dir=task_dir,
                    )
                else:
                    # ToT/DFSDT direct: run up to max_try independent trajectories (one file per attempt)
                    sp = SearchParams(
                        beam_size=args.beam_size,
                        value_threshold=args.value_threshold,
                        max_try=1,
                        max_depth=args.max_steps,
                        diversity_back_steps=args.diversity_back_steps,
                        diversity_back_steps_alt=args.diversity_back_steps_alt,
                        propose_k=args.propose_k,
                    )
                    mode = "tot" if args.strategy == "tot" else "dfsdt"
                    res = None
                    # Clear task dir for fresh attempts
                    if task_dir and os.path.exists(task_dir):
                        try:
                            shutil.rmtree(task_dir)
                        except Exception as e:
                            logging.warning(f"Failed to clear task dir {task_dir}: {e}")
                        os.makedirs(task_dir, exist_ok=True)
                    for attempt_idx in range(1, args.max_try + 1):
                        logging.info(f"[{mode.upper()}] Starting trajectory attempt {attempt_idx}/{args.max_try}")
                        sr = run_tree_search(
                        env_manager=local_mgr,
                        agent=agent,
                        max_steps=args.max_steps,
                        env_type=args.env,
                        params=sp,
                            mode=mode,
                            task_dir=task_dir if args.save_per_task_trajectories else None,
                            attempt_id=attempt_idx if args.save_per_task_trajectories else None,
                        )
                        if res is None:
                            res = sr.copy()
                            res["search_attempts"] = []
                        res["search_attempts"].append({
                            "won": sr.get("won", False),
                            "steps": sr.get("steps", 0),
                            "strategy": mode,
                        })
                        if sr.get("won", False):
                            res = sr
                            break
                    # Write summary
                    if task_dir and args.strategy in ("tot", "dfsdt"):
                        try:
                            summary_file = os.path.join(task_dir, "task_summary.json")
                            meta = {
                                "model": agent.model_name,
                                "env_type": args.env,
                                "strategy": args.strategy,
                                "total_attempts": len(res.get("search_attempts", [])) if res.get("search_attempts") else 0,
                                "won": bool(res.get("won", False)),
                                "timestamp": run_ts,
                            }
                            with open(summary_file, "w", encoding="utf-8") as f:
                                json.dump({
                                    "metadata": meta,
                                    "search_attempts": res.get("search_attempts", []),
                                }, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            logging.debug(f"Failed to write updated summary: {e}")
                res["task_id"] = task_id
                res["env_id"] = env_idx
                res["round_idx"] = round_idx
                if gamefile is not None:
                    res["game_file"] = gamefile
                
                # Close per-round manager for AlfWorld unique mode to free resources
                if args.env == "alfworld" and args.unique_envs and alfworld_game_files:
                    try:
                        local_mgr.envs.close()
                    except Exception:
                        pass
                return res

            if args.env == "alfworld" and args.unique_envs and alfworld_game_files:
                logging.info("\n========== AlfWorld unique-env per‑env assignment ==========")

                def _infer_task_category(game_file: Optional[str]) -> str:
                    if not game_file:
                        return "other"
                    known_types = [
                        "pick_and_place",
                        "pick_two_obj_and_place",
                        "look_at_obj_in_light",
                        "pick_heat_then_place_in_recep",
                        "pick_cool_then_place_in_recep",
                        "pick_clean_then_place_in_recep",
                    ]
                    for t in known_types:
                        if t in game_file:
                            return t
                    return "other"

                def _run_env_worker(env_idx: int) -> List[Dict[str, Any]]:
                    assigned_files = per_env_assigned_payloads[env_idx]
                    slot_results: List[Dict[str, Any]] = []
                    for r, game_file in enumerate(assigned_files):
                        res = _run_one_round(env_idx, r, override_gamefile=game_file)
                        if res:
                            if "task_category" not in res:
                                res["task_category"] = _infer_task_category(res.get("game_file"))
                            res["logical_env_idx"] = env_idx
                            res["job_index"] = r
                            slot_results.append(res)
                    return slot_results

                all_slot_results: List[Dict[str, Any]] = []
                with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                    futures = [ex.submit(_run_env_worker, i) for i in range(pool_size)]
                    for fut in as_completed(futures):
                        slot_results = fut.result()
                        all_slot_results.extend(slot_results)

                if not all_slot_results:
                    logging.warning("No AlfWorld results were produced. Check task preparation.")
                else:
                    # Aggregate statistics
                    for res in all_slot_results:
                        total_tasks += 1
                        if res.get("first_attempt_success"):
                            total_first_attempt_successes += 1
                        if res.get("won"):
                            total_debugger_successes += 1

                        task_cat = res.get("task_category", "other")
                        all_task_success_history[task_cat].append(1.0 if res.get("won") else 0.0)

                    overall_rate = float(np.mean([1.0 if r.get("won") else 0.0 for r in all_slot_results])) if all_slot_results else 0.0
                    first_attempt_rate = float(np.mean([1.0 if r.get("first_attempt_success") else 0.0 for r in all_slot_results])) if all_slot_results else 0.0
                    all_first_attempt_success_rates.append(first_attempt_rate)
                    all_debugger_success_rates.append(overall_rate)

                    logging.info(
                        f"Overall: First attempt success {first_attempt_rate:.4f}, Debugger success {overall_rate:.4f}"
                    )
            else:
                # GAIA/WEBSHOP or AlfWorld without unique_envs
                logging.info("\n========== Per‑env assignment scheduling ==========")

                def _run_env_worker(env_idx: int) -> List[Dict[str, Any]]:
                    assigned_rounds = len(per_env_assigned_payloads[env_idx]) or rounds
                    actual_pool_size = len(env_pool) if env_pool else pool_size
                    if env_pool and env_idx >= actual_pool_size:
                        return []
                    slot_results: List[Dict[str, Any]] = []
                    for r in range(assigned_rounds):
                        res = _run_one_round(env_idx, r)
                        if res:
                            slot_results.append(res)
                    return slot_results

                all_slot_results: List[Dict[str, Any]] = []
                with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                    actual_pool_size = len(env_pool) if env_pool else pool_size
                    futures = [ex.submit(_run_env_worker, i) for i in range(actual_pool_size)]
                    for fut in as_completed(futures):
                        slot_results = fut.result()
                        all_slot_results.extend(slot_results)

                # Update stats
                if all_slot_results:
                    overall = np.array([1 if rr.get('won') else 0 for rr in all_slot_results], dtype=float)
                    first_attempt = np.array([1 if rr.get('first_attempt_success') else 0 for rr in all_slot_results], dtype=float)
                    total_tasks += len(all_slot_results)
                    total_first_attempt_successes += int(first_attempt.sum())
                    total_debugger_successes += int(overall.sum())
                    all_first_attempt_success_rates.append(float(first_attempt.mean() if len(first_attempt) else 0.0))
                    all_debugger_success_rates.append(float(overall.mean() if len(overall) else 0.0))

            # Close pool (if any)
            if env_pool:
                for mgr in env_pool:
                    try:
                        mgr.envs.close()
                    except Exception:
                        pass

            global_env_counter = pool_size

        else:
            # Without debugger: build a single multi-env manager once and reuse
            # In no‑debugger path, for AlfWorld unique mode we rebuild per‑round with new files to avoid repetition
            if args.env == "alfworld" and args.unique_envs and alfworld_game_files:
                for test_idx in range(rounds):
                    round_slice_start = test_idx * pool_size
                    round_slice_end = round_slice_start + pool_size
                    files_this_round = alfworld_game_files[round_slice_start:round_slice_end]
                    env_kwargs = {
                        "env_num": pool_size,
                        "seed": args.seed,
                        "history_length": args.history_length,
                        "alf_env_type": args.alf_env_type,
                        "game_files": files_this_round,
                    }
                    env_manager = EnvironmentFactory.build_env("alfworld", with_debugger=False, **env_kwargs)
                    obs, infos = env_manager.reset()
                    env_dones = [False] * pool_size
                    overall_success_this_round = np.zeros(pool_size, dtype=bool)
                    
                    for step_idx in range(args.max_steps):
                        prompts, idx_map = [], []
                        for i in range(pool_size):
                            if not env_dones[i]:
                                prompts.append(obs["text"][i])
                                idx_map.append(i)
                        if not prompts:
                            break
                        batch_actions = agent.get_actions_batch(prompts, concurrency=args.concurrency, retries=args.retries)
                        actions = ["None"] * pool_size
                        for k, i in enumerate(idx_map):
                            actions[i] = batch_actions[k]
                        obs, rewards, dones, infos = env_manager.step(actions.copy())
                        for i in range(pool_size):
                            if env_dones[i]:
                                continue
                            if dones[i]:
                                env_dones[i] = True
                                overall_success_this_round[i] = bool(infos[i].get("won", False))
                        if all(env_dones):
                            break
                    all_overall_success_rates.append(overall_success_this_round.mean())
                    try:
                        env_manager.envs.close()
                    except Exception:
                        pass
                global_env_counter = pool_size
            else:
                env_kwargs = {
                    "env_num": pool_size,
                    "seed": args.seed,
                    "history_length": args.history_length,
                }
                if args.env == "gaia":
                    env_kwargs["tasks_data"] = gaia_tasks
                    env_kwargs["available_tools"] = args.gaia_tools
                    env_kwargs["max_steps"] = args.max_steps
                elif args.env == "alfworld":
                    env_kwargs["alf_env_type"] = args.alf_env_type
                    if args.unique_envs and alfworld_game_files:
                        env_kwargs["game_files"] = alfworld_game_files[:pool_size]
                elif args.env == "webshop":
                    env_kwargs["use_train_set"] = args.webshop_train

                env_manager = EnvironmentFactory.build_env(args.env, with_debugger=False, **env_kwargs)

                # Repeat for a number of rounds; each round calls reset() and steps to done
                for test_idx in range(rounds):
                    obs, infos = env_manager.reset()
                env_dones = [False] * pool_size
                overall_success_this_round = np.zeros(pool_size, dtype=bool)

                for step_idx in range(args.max_steps):
                    # Collect prompts for active envs
                    prompts, idx_map = [], []
                    for i in range(pool_size):
                        if not env_dones[i]:
                            prompts.append(obs["text"][i])
                            idx_map.append(i)
                    if not prompts:
                        break

                    batch_actions = agent.get_actions_batch(prompts, concurrency=args.concurrency, retries=args.retries)
                    actions = ["None"] * pool_size
                    for k, i in enumerate(idx_map):
                        actions[i] = batch_actions[k]

                    prev_prompts = obs["text"]
                    raw_actions = actions.copy()
                    obs, rewards, dones, infos = env_manager.step(actions.copy())

                    for i in range(pool_size):
                        if env_dones[i]:
                            continue
                        if dones[i]:
                            env_dones[i] = True
                            overall_success_this_round[i] = bool(infos[i].get("won", False))
                    if all(env_dones):
                        break

                # Update simple aggregate for non-debugger path
                all_overall_success_rates.append(overall_success_this_round.mean())

            try:
                env_manager.envs.close()
            except Exception:
                pass
            global_env_counter = pool_size

    finally:
        # Clean up shared LLM executor
        if 'shared_llm_executor' in locals():
            shared_llm_executor.shutdown(wait=True)
            logging.info("Shared LLM executor pool shut down")
        
        if dump_fp is not None:
            dump_fp.flush()
            dump_fp.close()
            logging.info(f"Trajectories saved to: {args.dump_path}")
    
    strategy_label_lookup = {
        "debugger": "Debugger",
        "bon": "Best-of-N",
        "tot": "Tree-of-Thought",
        "dfsdt": "DFSDT",
    }
    strategy_descriptor = strategy_label_lookup.get(args.strategy, args.strategy)
    if args.strategy == "debugger" and getattr(args, "debugger_type", None):
        strategy_descriptor = f"{strategy_descriptor} ({args.debugger_type})"
    if args.strategy == "bon" and getattr(args, "bon_n", None):
        strategy_descriptor = f"{strategy_descriptor} (N={args.bon_n})"

    # Final summary
    logging.info("=============== Final Summary ===============")
    logging.info(f"Environment: {args.env}")
    logging.info(f"Total batches: {num_batches} | Parallel envs: {max(1, int(args.total_envs))} | Total tasks run: {total_tasks}")
    
    # Report both first attempt and post-strategy success rates
    if args.enable_debugger and total_tasks > 0:
        first_attempt_success_rate = total_first_attempt_successes / total_tasks
        strategy_success_rate = total_debugger_successes / total_tasks
        improvement = strategy_success_rate - first_attempt_success_rate
        
        logging.info("\n========== Success Rate Analysis ==========")
        logging.info(f"First Attempt Success Rate: {first_attempt_success_rate:.4f} ({total_first_attempt_successes}/{total_tasks})")
        logging.info(f"Success Rate after {strategy_descriptor}: {strategy_success_rate:.4f} ({total_debugger_successes}/{total_tasks})")
        logging.info(f"Improvement over First Attempt: +{improvement:.4f} ({improvement*100:.2f}%)")
        
        if all_first_attempt_success_rates:
            logging.info(
                f"First Attempt (avg ± std): "
                f"{np.mean(all_first_attempt_success_rates):.4f} ± {np.std(all_first_attempt_success_rates):.4f}"
            )
        if all_debugger_success_rates:
            logging.info(
                f"After {strategy_descriptor} (avg ± std): "
                f"{np.mean(all_debugger_success_rates):.4f} ± {np.std(all_debugger_success_rates):.4f}"
            )
    elif all_overall_success_rates:
        logging.info(
            f"Overall success avg ± std: "
            f"{np.mean(all_overall_success_rates):.4f} ± {np.std(all_overall_success_rates):.4f}"
        )
    
    # Save final experiment summary to file if experiment_dir is set
    if summaries_dir:
        summary_file = os.path.join(summaries_dir, "experiment_summary.json")
        experiment_summary = {
            "experiment_info": {
                "environment": args.env,
                "model": args.model,
                "temperature": args.temperature,
                "debugger_enabled": args.enable_debugger,
                "debugger_model": args.debugger_model if args.enable_debugger else None,
                "debugger_temperature": args.debugger_temperature if args.enable_debugger else None,
                "max_retries": get_max_retries() if args.enable_debugger else 1,
                "total_tasks": total_tasks,
                "total_batches": num_batches,
                "batch_size": args.batch_size,
                "max_steps": args.max_steps,
                "timestamp": run_ts,
                "strategy": args.strategy,
                "strategy_descriptor": strategy_descriptor
            },
            "results": {
                "first_attempt_success_rate": float(total_first_attempt_successes) / total_tasks if total_tasks > 0 else 0,
                "strategy_success_rate": float(total_debugger_successes) / total_tasks if total_tasks > 0 else 0,
                "debugger_success_rate": float(total_debugger_successes) / total_tasks if total_tasks > 0 else 0,
                "improvement": (float(total_debugger_successes) - float(total_first_attempt_successes)) / total_tasks if total_tasks > 0 else 0,
                "first_attempt_successes": int(total_first_attempt_successes),
                "strategy_successes": int(total_debugger_successes),
                "debugger_successes": int(total_debugger_successes),
                "total_tasks": int(total_tasks)
            },
            "statistics": {
                "first_attempt_mean": float(np.mean(all_first_attempt_success_rates)) if all_first_attempt_success_rates else 0,
                "first_attempt_std": float(np.std(all_first_attempt_success_rates)) if all_first_attempt_success_rates else 0,
                "strategy_mean": float(np.mean(all_debugger_success_rates)) if all_debugger_success_rates else 0,
                "strategy_std": float(np.std(all_debugger_success_rates)) if all_debugger_success_rates else 0,
                "debugger_mean": float(np.mean(all_debugger_success_rates)) if all_debugger_success_rates else 0,
                "debugger_std": float(np.std(all_debugger_success_rates)) if all_debugger_success_rates else 0
            }
        }
        
        with open(summary_file, "w") as f:
            json.dump(experiment_summary, f, indent=2)
        logging.info(f"\nExperiment summary saved to: {summary_file}")
    
    # Environment-specific summaries
    if args.env == "alfworld":
        task_types = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
                     "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", 
                     "pick_clean_then_place_in_recep", "other"]
        for task in task_types:
            if task in all_task_success_history and all_task_success_history[task]:
                rates = [r for r in all_task_success_history[task] if r is not None]
                if rates:
                    logging.info(f"{task:<35s}: {np.mean(rates):.4f} ± {np.std(rates):.4f}")
    
    elif args.env == "gaia":
        successful_tasks = sum(1 for rates in all_task_success_history.values() if any(r > 0 for r in rates))
        logging.info(f"Successfully completed {successful_tasks} out of {len(all_task_success_history)} unique tasks")


if __name__ == "__main__":
    main()
