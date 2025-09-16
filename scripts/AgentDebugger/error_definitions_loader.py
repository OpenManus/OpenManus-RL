#!/usr/bin/env python3
"""
Error Definitions Loader V3 - Updated based on latest error_type_definition.md
All definitions and examples in English
"""

from typing import Dict, Any

class ErrorDefinitionsLoader:
    """Loads and manages error type definitions for prompts"""
    
    def __init__(self):
        self.definitions = self._load_definitions()
    
    def _load_definitions(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Load error definitions with English translations
        Based on latest error_type_definition.md
        """
        definitions = {
            'memory': {},
            'reflection': {},
            'planning': {},
            'action': {},
            'system': {},
            'others': {}
        }
        
        # Memory Module Errors
        definitions['memory'] = {
            'over_simplification': {
                'definition': 'Agent oversimplifies complex information from previous N steps, ignoring details and key factors, leading to decisions based on partial or oversimplified summaries',
                'example': 'Agent simplifies multiple product selection criteria to just "item found", ignoring price, features, and inventory factors'
            },
            'memory_retrieval_failure': {
                'definition': 'Relevant information exists in agent memory but fails to be retrieved when needed',
                'example': 'Scenario: Step 3 - Agent explored kitchen and observed "You see a knife on the countertop". Error: Step 10 - When needing to cut object, agent fails to recall knife location and aimlessly searches other rooms (bedroom, bathroom) for knife'
            },
            'hallucination': {
                'definition': 'Agent "recalls" events that never happened, object states never observed, or actions never executed, and uses these as basis for reasoning. Agent fills in missing parts without solid memory basis, generating false memory information',
                'example': 'When needing a knife, memory module generates "I remember seeing a knife in the first drawer" even though "open drawer 1" was never successfully executed'
            }
        }
        
        # Reflection Module Errors
        definitions['reflection'] = {
            'progress_misjudge': {
                'definition': 'Agent incorrectly evaluates progress toward completing the overall task goal, being either overly optimistic or pessimistic',
                'example': 'Overly optimistic: Task is "find a cup and fill it with water". Agent just entered kitchen without finding cup yet, but reflects "progress is smooth, task nearly complete". Overly pessimistic: Agent successfully filled cup with water, only needs to "put it on microwave", but reflects "huge difficulties, no progress" and abandons correct plan'
            },
            'outcome_misinterpretation': {
                'definition': 'Agent correctly executes an action but incorrectly interprets the direct result or environment feedback from that action',
                'example': 'Scenario: Agent executes Put(Apple, Microwave) but due to distance, environment returns "Nothing happens". Error reflection: Agent reflects "Successfully placed apple in microwave. Next step is to start it" (apple is still in hand)'
            },
            'causal_misattribution': {
                'definition': 'Agent correctly identifies a failure phenomenon but attributes it to the wrong cause',
                'example': 'Scenario: Take(Key) fails because key is in locked safe. Error reflection: Agent reflects "Cannot pick up key, probably mechanical arm malfunction" instead of "key is in safe, need to open safe first"'
            },
            'hallucination': {
                'definition': 'Agent believes it performed actions that never actually occurred',
                'example': 'Agent in step 6 interprets the plan generated in step 1 as operations it has already completed'
            }
        }
        
        # Planning Module Errors
        definitions['planning'] = {
            'constraint_ignorance': {
                'definition': 'Planning ignores task constraints, not considering resource limits (time, budget, space) or other relevant restrictions',
                'example': 'Budget is $40 but selects $55 product, not considering time/interaction round limits'
            },
            'impossible_action': {
                'definition': 'Agent plans to execute an action or subtask that is fundamentally impossible under current physical or logical conditions. May include decomposition problems',
                'example': 'Physical impossibility: Agent plans Slice(Desk) (cut desk with knife). Logical/prerequisite impossibility: Empty inventory but plans Put(Mug, Sink) (planning to put mug when having no mug)'
            },
            'inefficient_plan': {
                'definition': 'Agent creates plan that can theoretically complete task but is extremely inefficient, lengthy, or illogical',
                'example': 'Task: "Take apple from living room to kitchen". Error plan: Agent plans to go upstairs to bedroom, then bathroom, then downstairs to kitchen, instead of direct path from living room to kitchen'
            }
        }
        
        # Action Module Errors  
        definitions['action'] = {
            'misalignment': {
                'definition': 'Generated specific action completely contradicts the intention stated in current plan module (wrong action choice)',
                'example': 'Plan: "I found the knife, next I plan to slice the apple". Action: <action>GoTo(Bedroom 1)</action> (executes unrelated movement)'
            },
            'invalid_action': {
                'definition': 'Uses action that does not exist',
                'example': 'Action is not in the available action list'
            },
            'format_error': {
                'definition': 'Generated action has invalid format causing parse failure',
                'example': 'click"product" (correct format is click["product"])'
            },
            'parameter_error': {
                'definition': 'Action parameters are unreasonable or incorrectly chosen',
                'example': 'search[query repeated 100 times]'
            }
        }
        
        # System Errors
        definitions['system'] = {
            'step_limit': {
                'definition': 'Agent executes reasonably but fails due to reaching system maximum step limit',
                'example': 'Find two items task: first item found and placed, searching for second item when 30-step limit reached'
            },
            'tool_execution_error': {
                'definition': 'External tool or API called by agent returns error or exhibits unpredictable behavior',
                'example': 'Object recognition tool misidentifies apple as tomato, causing subsequent failures from tool error'
            },
            'llm_limit': {
                'definition': 'Agent response limitations cause failure',
                'example': 'API call timeout, max token exceeded'
            },
            'environment_error': {
                'definition': 'Simulation environment itself has bugs, network issues cause unexpected behavior against expected rules',
                'example': 'Agent executes valid Open(Drawer) command but environment crashes or drawer model disappears'
            }
        }
        
        # Others
        definitions['others'] = {
            'others': {
                'definition': 'All remaining problems not previously defined or discussed',
                'example': 'Issues not covered by any of the above categories'
            }
        }
        
        return definitions
    
    def get_module_definitions(self, module_name: str) -> Dict[str, Dict[str, str]]:
        """Get all error definitions for a specific module"""
        return self.definitions.get(module_name, {})
    
    def format_for_phase1_prompt(self, module_name: str) -> str:
        """
        Format error definitions for Phase 1 prompt
        Returns formatted string with definitions and examples
        """
        module_defs = self.get_module_definitions(module_name)
        
        if not module_defs:
            # If module not found, check if it's a system/others category
            if module_name == 'system':
                module_defs = self.definitions.get('system', {})
            elif module_name == 'others':
                return "• **others**: All remaining problems not previously defined or discussed\n"
            else:
                return f"No error definitions found for module: {module_name}"
        
        formatted = f"DETAILED ERROR TYPE DEFINITIONS FOR {module_name.upper()} MODULE:\n\n"
        
        for error_type, details in module_defs.items():
            formatted += f"• **{error_type}**:\n"
            formatted += f"  - Definition: {details['definition']}\n"
            if 'example' in details and details['example']:
                formatted += f"  - Example: {details['example']}\n"
            formatted += "\n"
        
        formatted += "• **no_error**: No error detected in this module\n"
        
        return formatted
    
    def format_for_phase2_prompt(self) -> str:
        """
        Format all error definitions for Phase 2 critical error identification
        """
        reference = "COMPLETE ERROR TYPE REFERENCE WITH DEFINITIONS:\n\n"
        
        module_order = ['memory', 'reflection', 'planning', 'action', 'system', 'others']
        
        for module in module_order:
            reference += f"━━━ {module.upper()} MODULE ERRORS ━━━\n"
            module_defs = self.definitions.get(module, {})
            
            for error_type, details in module_defs.items():
                reference += f"• {error_type}: {details['definition']}\n"
            
            reference += "\n"
        
        return reference
    
    def get_valid_error_types(self, module_name: str) -> list:
        """Get list of valid error type names for a module"""
        module_defs = self.get_module_definitions(module_name)
        if module_name == 'others':
            return ['others', 'no_error']
        return list(module_defs.keys()) + ['no_error']
    
    def get_all_modules(self) -> list:
        """Get list of all modules"""
        return list(self.definitions.keys())