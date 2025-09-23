"""
Tool Use Projection function for processing text actions into tool calls or regular actions.
"""

import json
import re
from typing import List, Tuple


def tool_use_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    Process text actions for tool use environment.
    
    Args:
        actions: List of text actions from the agent
        
    Returns:
        Tuple of (processed_actions, valids) where:
        - processed_actions: List of processed action strings
        - valids: List of 1s and 0s indicating valid/invalid actions
    """
    valids = [0] * len(actions)
    processed_actions = []
    
    for i, action in enumerate(actions):
        try:
            # Start with assuming action is valid
            is_valid = True
            original_action = action
            
            # Check for Chinese characters - mark as invalid if found
            if re.search(r'[\u4e00-\u9fff]', action):
                is_valid = False
            
            # Check if action contains tool call
            if _has_tool_call(action):
                # Parse and validate tool call
                tool_action, tool_valid = _parse_tool_call(action)
                processed_actions.append(tool_action)
                is_valid = is_valid and tool_valid
                
            # Check if action contains final answer
            elif _has_answer(action):
                # Extract answer and mark as completion action
                answer = _extract_answer(action)
                processed_actions.append(f"FINAL_ANSWER: {answer}")
                is_valid = True  # Answer actions are always valid
                
            else:
                # Regular reasoning action - just pass through
                processed_actions.append(action)
                # Regular actions are valid as long as they don't have Chinese
            
            valids[i] = 1 if is_valid else 0
            
        except Exception as e:
            # If any error occurs, mark as invalid
            processed_actions.append(action)
            valids[i] = 0
    
    return processed_actions, valids


def _has_tool_call(action: str) -> bool:
    """Return True if the text contains a tool call block.

    Supports both legacy <tool_call>...</tool_call> and main-style
    <action>...</action> wrappers.
    """
    if re.search(r"<tool_call>.*?</tool_call>", action, re.DOTALL):
        return True
    # Accept <action> wrapper as a tool-call container for GAIA prompts
    return bool(re.search(r"<action>.*?</action>", action, re.DOTALL))


def _has_answer(action: str) -> bool:
    """Check if action contains a final answer"""
    return bool(re.search(r'<answer>.*?</answer>', action, re.DOTALL))


def _parse_tool_call(action: str) -> Tuple[str, bool]:
    """
    Parse tool call from action text.
    
    Returns:
        Tuple of (parsed_action, is_valid)
    """
    try:
        # Extract tool call content from either <tool_call> or <action>
        tool_match = re.search(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
        if tool_match is None:
            tool_match = re.search(r"<action>(.*?)</action>", action, re.DOTALL)
        if tool_match is None:
            return action, False
        tool_content = tool_match.group(1).strip()
        
        # Parse tool name and parameters
        tool_name = None
        params = {}
        
        lines = tool_content.split('\n')
        for line in lines:
            line = line.strip()
            # Accept either 'tool:' (preferred) or 'action:' as the tool name line
            if line.lower().startswith('tool:') or line.lower().startswith('action:'):
                tool_name = line.split(':', 1)[1].strip()
            elif line.lower().startswith('parameters:'):
                try:
                    params_str = line.split(':', 1)[1].strip()
                    # Try to parse as JSON
                    params = json.loads(params_str)
                except (json.JSONDecodeError, IndexError):
                    # Fallback to treating the whole thing as a query
                    params = {'query': params_str}
            elif ':' in line and not tool_name:
                # Handle simple key:value format
                key, value = line.split(':', 1)
                params[key.strip()] = value.strip()
        
        if not tool_name:
            return action, False
        
        # Format as structured action
        formatted_action = json.dumps({
            'type': 'tool_call',
            'tool': tool_name,
            'parameters': params,
            'original': action
        })
        
        return formatted_action, True
        
    except Exception:
        return action, False


def _extract_answer(action: str) -> str:
    """Extract final answer from action text"""
    answer_match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""
