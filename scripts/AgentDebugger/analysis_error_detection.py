#!/usr/bin/env python3
"""
Phase 1: Error Type Detection System V5
Using latest error_type_definition.md with V3 loader
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
from error_definitions_loader import ErrorDefinitionsLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModuleError:
    """Error detection for a single module"""
    module_name: str
    error_type: str
    error_detected: bool
    evidence: str
    reasoning: str


@dataclass
class StepAnalysis:
    """Analysis for a single step"""
    step: int
    memory_error: Optional[ModuleError]
    reflection_error: Optional[ModuleError]
    planning_error: Optional[ModuleError]
    action_error: Optional[ModuleError]
    step_summary: str


class ErrorTypeDetector:
    """Detects error types without scoring"""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.config = api_config
        self.headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # Load error definitions
        self.error_loader = ErrorDefinitionsLoader()
        
        # Define error types based on latest error_type_definition.md
        self.error_types = {
            'memory': self.error_loader.get_valid_error_types('memory'),
            'reflection': self.error_loader.get_valid_error_types('reflection'),
            'planning': self.error_loader.get_valid_error_types('planning'),
            'action': self.error_loader.get_valid_error_types('action'),
            'system': self.error_loader.get_valid_error_types('system'),
            'others': self.error_loader.get_valid_error_types('others')
        }
    
    def parse_trajectory_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse trajectory from dictionary (for API use)"""
        metadata = data.get('metadata', {})
        # Support both old format (chat_history) and new format (messages)
        chat_history = data.get('messages', data.get('chat_history', []))

        # Extract task description
        task_description = ""
        for msg in chat_history:
            if msg['role'] == 'user' and 'task' in msg['content'].lower():
                # Look for task description patterns
                if 'Your task is:' in msg['content']:
                    task_description = msg['content'].split('Your task is:')[1].split('\n')[0].strip()
                    break
                elif 'put a clean' in msg['content'].lower() or 'find' in msg['content'].lower():
                    # Extract the task from the first user message
                    lines = msg['content'].split('\n')
                    for line in lines:
                        if 'task' in line.lower() or 'put' in line.lower() or 'find' in line.lower():
                            task_description = line.strip()
                            break

        # Extract steps
        steps = []
        step_num = 0

        for i, msg in enumerate(chat_history):
            if msg['role'] == 'assistant':
                step_num += 1

                # Get environment response
                env_response = ""
                if i + 1 < len(chat_history) and chat_history[i + 1]['role'] == 'user':
                    env_response = chat_history[i + 1]['content']

                # Get the user input for this step (from previous message)
                current_input = ""
                if i > 0 and chat_history[i-1]['role'] == 'user':
                    current_input = chat_history[i-1]['content']

                steps.append({
                    'step': step_num,
                    'content': msg['content'],
                    'env_response': env_response,
                    'current_input': current_input
                })

        return {
            'task_id': metadata.get('task_id', 'unknown'),
            'task_description': task_description or metadata.get('task', ''),
            'success': metadata.get('success', metadata.get('won', False)),
            'steps': steps,
            'total_steps': step_num,
            'environment': metadata.get('environment', 'alfworld')
        }

    def parse_trajectory(self, file_path: str) -> Dict[str, Any]:
        """Parse trajectory file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        # Support both old format (chat_history) and new format (messages)
        chat_history = data.get('messages', data.get('chat_history', []))
        
        # Extract task description
        task_description = ""
        for msg in chat_history:
            if msg['role'] == 'user' and 'task' in msg['content'].lower():
                # Look for task description patterns
                if 'Your task is:' in msg['content']:
                    task_description = msg['content'].split('Your task is:')[1].split('\n')[0].strip()
                    break
                elif 'put a clean' in msg['content'].lower() or 'find' in msg['content'].lower():
                    # Extract the task from the first user message
                    lines = msg['content'].split('\n')
                    for line in lines:
                        if 'task' in line.lower() or 'put' in line.lower() or 'find' in line.lower():
                            task_description = line.strip()
                            break
        
        # Extract steps
        steps = []
        step_num = 0
        
        for i, msg in enumerate(chat_history):
            if msg['role'] == 'assistant':
                step_num += 1
                
                # Get environment response
                env_response = ""
                if i + 1 < len(chat_history) and chat_history[i + 1]['role'] == 'user':
                    env_response = chat_history[i + 1]['content']
                
                # Get the user input for this step (from previous message)
                current_input = ""
                if i > 0 and chat_history[i-1]['role'] == 'user':
                    current_input = chat_history[i-1]['content']
                
                steps.append({
                    'step': step_num,
                    'content': msg['content'],
                    'env_response': env_response,
                    'current_input': current_input
                })
        
        return {
            'task_id': metadata.get('task_id', 'unknown'),
            'task_description': task_description or metadata.get('task', ''),
            'success': metadata.get('success', metadata.get('won', False)),
            'steps': steps,
            'total_steps': step_num,
            'environment': metadata.get('environment', 'alfworld')
        }
    
    def extract_module_content_from_step(self, content: str, module_name: str, env: str) -> str:
        """Extract specific module content from step"""
        modules = self.extract_modules_from_content(content, env)
        return modules.get(module_name, "")
    
    def extract_modules_from_content(self, content: str, env: str) -> Dict[str, str]:
        """Extract module content based on environment and prompt format"""
        modules = {}
        
        if env == 'alfworld':
            # ALFWorld uses specific tags from prompt.py
            # Memory analysis
            memory_match = re.search(r'<memory>(.*?)</memory>', content, re.DOTALL)
            if memory_match:
                modules['memory'] = memory_match.group(1).strip()
            
            # Reflection
            reflection_match = re.search(r'<reflection>(.*?)</reflection>', content, re.DOTALL)
            if reflection_match:
                modules['reflection'] = reflection_match.group(1).strip()
            
            # Plan
            plan_match = re.search(r'<plan>(.*?)</plan>', content, re.DOTALL)
            if plan_match:
                modules['planning'] = plan_match.group(1).strip()
            
            # Action
            action_match = re.search(r'<action>(.*?)</action>', content, re.DOTALL)
            if action_match:
                modules['action'] = action_match.group(1).strip()
        
        else:  # webshop or other
            # WebShop might use different format
            modules['memory'] = ""  # WebShop might not have explicit memory
            modules['reflection'] = ""  # WebShop might not have explicit reflection
            
            # Look for planning/thinking
            if 'think' in content.lower() or 'plan' in content.lower():
                modules['planning'] = content
            
            # Look for action patterns
            action_patterns = [
                r'search\[(.*?)\]',
                r'click\[(.*?)\]',
                r'buy\[(.*?)\]'
            ]
            for pattern in action_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    modules['action'] = match.group(0)
                    break
        
        return modules
    
    async def detect_module_errors(
        self,
        module_name: str,
        module_content: str,
        step_num: int,
        step_data: Dict,
        task_description: str,
        env_response: str,
        previous_steps: List[Dict],
        current_step_input: str,
        task_success: bool,
        environment: str
    ) -> ModuleError:
        """Detect errors in a specific module"""
        
        # Build detection prompt
        prompt = self._build_error_detection_prompt(
            module_name,
            module_content,
            step_num,
            step_data,
            task_description,
            env_response,
            previous_steps,
            current_step_input,
            environment
        )
        
        # Call LLM for error detection
        response = await self.call_llm(prompt)
        
        # Parse detection result
        error = self._parse_error_detection(response, module_name)
        
        return error
    
    def _build_error_detection_prompt(
        self,
        module_name: str,
        module_content: str,
        step_num: int,
        step_data: Dict,
        task_description: str,
        env_response: str,
        previous_steps: List[Dict],
        current_step_input: str,
        environment: str
    ) -> str:
        """Build prompt for error detection"""
        
        # Get detailed error definitions for this module
        error_definitions = self.error_loader.format_for_phase1_prompt(module_name)
        
        # Build context - ALL modules need current step input (user message which includes history)
        context = f"Current Step Input (user message with history):\n{current_step_input}\n\n"
        
        # Add previous module outputs from THIS SAME STEP for evaluation
        if module_name == 'memory':
            # Memory only needs the current step input
            pass
        elif module_name == 'reflection':
            # Reflection needs to see Memory output from this step
            memory_content = self.extract_module_content_from_step(step_data['content'], 'memory', environment)
            if memory_content:
                context += f"Memory Module Output (from this step):\n{memory_content}\n\n"
        elif module_name == 'planning':
            # Planning needs to see Memory and Reflection outputs from this step
            memory_content = self.extract_module_content_from_step(step_data['content'], 'memory', environment)
            reflection_content = self.extract_module_content_from_step(step_data['content'], 'reflection', environment)
            if memory_content:
                context += f"Memory Module Output (from this step):\n{memory_content}\n\n"
            if reflection_content:
                context += f"Reflection Module Output (from this step):\n{reflection_content}\n\n"
        elif module_name == 'action':
            # Action needs to see Planning output from this step
            planning_content = self.extract_module_content_from_step(step_data['content'], 'planning', environment)
            if planning_content:
                context += f"Planning Module Output (from this step):\n{planning_content}\n\n"
        
        prompt = f"""
You are an expert at detecting errors in agent trajectories.

TASK: {task_description}
ENVIRONMENT: {environment}
CURRENT STEP: {step_num}

INPUT AND CONTEXT:
{context}

MODULE TO ANALYZE: {module_name}
MODULE OUTPUT (What the agent produced for this module):
{module_content if module_content else "No content found for this module"}

ENVIRONMENT RESPONSE AFTER THIS STEP:
{env_response[:500] if env_response else "No response"}

{error_definitions}

Based on the SPECIFIC error definitions provided above:
1. Identify if there is an error in this module by checking if the output matches any error definition
2. If yes, specify which exact error type based on the definitions
3. Provide evidence from the content that directly relates to the definition
4. Explain your reasoning showing how it matches the specific definition criteria

SPECIAL RULES:
- The "Current Step Input" contains the full user message including conversation history
- Evaluation criteria for each module:
  * Memory: Should correctly summarize/recall from the current step input only
  * Reflection: Should correctly reflect based on current input + this step's Memory output
  * Planning: Should plan reasonably based on current input + this step's Memory & Reflection outputs
  * Action: Should execute correctly based on current input + this step's Planning output
- Each module builds on previous modules' outputs FROM THE SAME STEP
- System errors (step_limit, tool_execution_error, etc.) should be identified separately

REQUIRED OUTPUT FORMAT (JSON):
{{
    "error_detected": true/false,
    "error_type": "specific_error_type or no_error",
    "evidence": "Quote or description from module content supporting the detection",
    "reasoning": "Explanation of why this is (or isn't) an error based on the definition"
}}

Be precise and base your detection on the actual content and error definitions.
"""
        return prompt
    
    async def _check_system_errors(
        self,
        step_num: int,
        step_data: Dict,
        task_description: str,
        env_response: str,
        environment: str
    ) -> Optional[ModuleError]:
        """Check for system-level errors"""
        
        # Check for step limit
        if step_num >= 30:  # Common step limit in ALFWorld
            return ModuleError(
                module_name='system',
                error_type='step_limit',
                error_detected=True,
                evidence=f"Reached step {step_num}, hitting system step limit",
                reasoning="Task failed due to exceeding maximum allowed steps"
            )
        
        # Check for environment errors in response
        if env_response and any(keyword in env_response.lower() for keyword in ['error', 'crashed', 'exception', 'timeout']):
            # Call LLM to verify if this is a system error
            prompt = f"""
Analyze if this environment response indicates a system error:

Environment Response: {env_response[:500]}

System error types:
- tool_execution_error: External tool or API returned error
- llm_limit: LLM response limitations (timeout, max tokens)
- environment_error: Environment bug or crash

Is this a system error? If yes, which type?

REQUIRED OUTPUT FORMAT (JSON):
{{
    "error_detected": true/false,
    "error_type": "specific_system_error_type or no_error",
    "evidence": "Quote from response",
    "reasoning": "Why this is/isn't a system error"
}}
"""
            response = await self.call_llm(prompt)
            return self._parse_error_detection(response, 'system')
        
        return None
    
    def _parse_error_detection(self, response: str, module_name: str) -> ModuleError:
        """Parse LLM error detection response"""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                detection = json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found in response")
            
            return ModuleError(
                module_name=module_name,
                error_type=detection.get('error_type', 'unknown'),
                error_detected=detection.get('error_detected', False),
                evidence=detection.get('evidence', 'No evidence provided'),
                reasoning=detection.get('reasoning', 'No reasoning provided')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse error detection for {module_name}: {e}")
            return ModuleError(
                module_name=module_name,
                error_type='parse_error',
                error_detected=False,
                evidence=f"Parse error: {str(e)}",
                reasoning="Failed to parse LLM response"
            )
    
    async def analyze_step(
        self,
        step_data: Dict,
        task_description: str,
        previous_steps: List[Dict],
        task_success: bool,
        environment: str
    ) -> StepAnalysis:
        """Analyze all modules in a step for errors"""
        
        step_num = step_data['step']
        content = step_data['content']
        env_response = step_data['env_response']
        
        # Extract module content
        modules = self.extract_modules_from_content(content, environment)
        
        # Detect errors in each module
        module_errors = {}
        
        # Get current step input (user message) for Memory module
        current_step_input = ""
        if 'current_input' in step_data:
            current_step_input = step_data['current_input']
        
        # Check for system errors first (these apply to the whole step, not specific modules)
        system_error = await self._check_system_errors(
            step_num, 
            step_data, 
            task_description,
            env_response,
            environment
        )
        if system_error and system_error.error_detected:
            module_errors['system'] = system_error
        
        for module_name in ['memory', 'reflection', 'planning', 'action']:
            module_content = modules.get(module_name, "")
            
            # Skip memory/reflection for WebShop if not present
            if environment == 'webshop' and module_name in ['memory', 'reflection'] and not module_content:
                module_errors[module_name] = None
                continue
            
            error = await self.detect_module_errors(
                module_name,
                module_content,
                step_num,
                step_data,
                task_description,
                env_response,
                previous_steps,
                current_step_input,
                task_success,
                environment
            )
            module_errors[module_name] = error
        
        # Generate step summary
        errors_found = [
            f"{err.module_name}:{err.error_type}" 
            for err in module_errors.values() 
            if err and err.error_detected
        ]
        
        step_summary = f"Step {step_num}: "
        if errors_found:
            step_summary += f"Errors detected - {', '.join(errors_found)}"
        else:
            step_summary += "No errors detected"
        
        return StepAnalysis(
            step=step_num,
            memory_error=module_errors.get('memory'),
            reflection_error=module_errors.get('reflection'),
            planning_error=module_errors.get('planning'),
            action_error=module_errors.get('action'),
            step_summary=step_summary
        )
    
    async def call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        payload = {
            "model": self.config['model'],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at detecting errors in agent trajectories based on specific error type definitions."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.get('temperature', 0.0),
            "response_format": {"type": "json_object"}
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
    
    async def analyze_trajectory(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze complete trajectory for error types"""
        
        task_description = trajectory_data['task_description']
        task_success = trajectory_data['success']
        steps = trajectory_data['steps']
        environment = trajectory_data['environment']
        
        # Analyze each step
        step_analyses = []
        previous_steps = []
        
        for step_data in steps:
            analysis = await self.analyze_step(
                step_data,
                task_description,
                previous_steps,
                task_success,
                environment
            )
            step_analyses.append(analysis)
            previous_steps.append(step_data)
        
        # Convert to serializable format
        analyses_dict = []
        for analysis in step_analyses:
            step_dict = {
                'step': analysis.step,
                'errors': {},
                'summary': analysis.step_summary
            }
            
            for module_name in ['memory', 'reflection', 'planning', 'action']:
                error = getattr(analysis, f"{module_name}_error")
                if error:
                    step_dict['errors'][module_name] = {
                        'error_type': error.error_type,
                        'error_detected': error.error_detected,
                        'evidence': error.evidence,
                        'reasoning': error.reasoning
                    }
            
            analyses_dict.append(step_dict)
        
        return {
            'task_id': trajectory_data['task_id'],
            'task_description': task_description,
            'task_success': task_success,
            'environment': environment,
            'total_steps': len(steps),
            'step_analyses': analyses_dict
        }
    
    async def process_file(self, file_path: str, output_dir: str, output_filename: str = None) -> Dict[str, Any]:
        """Process a single trajectory file"""
        try:
            # Parse trajectory
            trajectory_data = self.parse_trajectory(file_path)
            
            # Analyze for errors
            analysis = await self.analyze_trajectory(trajectory_data)
            
            # Save results
            if output_filename:
                output_file = Path(output_dir) / f"{output_filename}_error_detection.json"
            else:
                output_file = Path(output_dir) / f"{Path(file_path).stem}_error_detection.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {analysis['task_id']}: {analysis['total_steps']} steps analyzed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
