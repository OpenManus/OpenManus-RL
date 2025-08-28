"""
Modular Stage Processing System
Each stage is independent and communicates through memory.md
Integrates with the new octotools tool system.
"""

import re
import json
from typing import Dict, List, Optional, Any

from openmanus_rl.memory.file_memory import FileMemory
from .tool_integration import GLOBAL_TOOL_REGISTRY, create_simple_tool_wrappers

class PlanningModule:
    """Planning stage - reads from memory, outputs plan."""
    
    def __init__(self, memory: FileMemory):
        self.memory = memory
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process planning stage with memory queries."""
        # Extract plan
        plan_match = re.search(r'<plan>(.*?)</plan>', text, re.DOTALL)
        if not plan_match:
            return {'plan': None, 'augmented_text': text}
        
        plan_content = plan_match.group(1).strip()
        
        # Find and process memory queries
        queries = re.findall(r'<memory query>(.*?)</memory query>', plan_content, re.DOTALL)
        
        augmented = text
        for query in queries:
            result = self.memory.query(query.strip())
            # Inject result
            augmented = augmented.replace(
                f'<memory query>{query}</memory query>',
                f'<memory query>{query}</memory query>\n<memory result>{result}</memory result>',
                1
            )
        
        # Extract clean plan (without memory tags)
        clean_plan = re.sub(r'<memory query>.*?</memory query>', '', plan_content, flags=re.DOTALL)
        clean_plan = re.sub(r'<memory result>.*?</memory result>', '', clean_plan, flags=re.DOTALL)
        
        return {
            'plan': clean_plan.strip(),
            'queries': queries,
            'augmented_text': augmented
        }


class ActionModule:
    """Action stage - executes tools or returns environment actions."""
    
    def __init__(self, memory: FileMemory):
        self.memory = memory
        self.tools = {}
        
        self._register_tools()
    
    def _register_tools(self):
        """Auto-register discovered tools."""
        simple_wrappers = create_simple_tool_wrappers(GLOBAL_TOOL_REGISTRY)
        for name, wrapper in simple_wrappers.items():
            self.tools[name] = wrapper
        print(f"Registered {len(simple_wrappers)} octotools: {list(simple_wrappers.keys())}")
    
    def register_tool(self, name: str, func):
        """Register a tool function."""
        self.tools[name] = func
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process action stage and execute tools."""
        # Extract action
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        if not action_match:
            return {'action': None, 'result': None}
        
        action_content = action_match.group(1).strip()
        
        # Check if it's a tool call
        if 'tool:' in action_content.lower():
            tool_name = None
            params = {}
            
            for line in action_content.split('\n'):
                if line.lower().startswith('tool:'):
                    tool_name = line.split(':', 1)[1].strip()
                elif line.lower().startswith('parameters:'):
                    try:
                        params = json.loads(line.split(':', 1)[1].strip())
                    except:
                        params = {'query': line.split(':', 1)[1].strip()}
            
            # Execute tool
            if tool_name in self.tools:
                result = self.tools[tool_name](params)
                
                # Inject result into text
                augmented = text.replace(
                    '</action>',
                    f'</action>\n<action results>{result}</action results>',
                    1
                )
                
                return {
                    'action': action_content,
                    'tool': tool_name,
                    'result': result,
                    'augmented_text': augmented,
                    'for_env': ""  # Empty action for environment
                }
        
        # Regular environment action
        # Extract the actual action from "action_choice: xxx" format
        env_action = action_content
        if 'action_choice:' in action_content:
            # Extract the part after "action_choice:"
            parts = action_content.split('action_choice:', 1)
            if len(parts) > 1:
                env_action = parts[1].strip()
                # Remove any action_parameters line
                if '\n' in env_action:
                    env_action = env_action.split('\n')[0].strip()
        
        return {
            'action': action_content,
            'tool': None,
            'result': None,
            'augmented_text': text,
            'for_env': env_action
        }


class MemoryStoreModule:
    """Memory storage stage - saves important information."""
    
    def __init__(self, memory: FileMemory):
        self.memory = memory
    
    def process(self, text: str, episode: str = "", step: int = 0) -> Dict[str, Any]:
        """Process memory store stage."""
        # Extract memory store content
        store_match = re.search(r'<memory store>(.*?)</memory store>', text, re.DOTALL)
        if not store_match:
            return {'stored': None}
        
        content = store_match.group(1).strip()
        
        # Store with metadata
        self.memory.store_to_file(content, episode, step)
        metadata = f"E:{episode}|S:{step}"
        
        return {
            'stored': content,
            'metadata': metadata
        }


class ReflectionModule:
    """Reflection stage - analyzes results and queries memory."""
    
    def __init__(self, memory: FileMemory):
        self.memory = memory
    
    def process(self, text: str, episode: str = "", step: int = 0) -> Dict[str, Any]:
        """Process reflection stage with memory queries."""
        # Extract reflection
        reflection_match = re.search(r'<reflection\s*>(.*?)</reflection>', text, re.DOTALL | re.IGNORECASE)
        if not reflection_match:
            return {'reflection': None}
        
        reflection_content = reflection_match.group(1).strip()
        
        # Process memory queries in reflection
        queries = re.findall(r'<memory query>(.*?)</memory query>', reflection_content, re.DOTALL)
        
        augmented = text
        for query in queries:
            result = self.memory.query(query.strip())
            augmented = augmented.replace(
                f'<memory query>{query}</memory query>',
                f'<memory query>{query}</memory query>\n<memory result>{result}</memory result>',
                1
            )
        
        # Clean reflection
        clean_reflection = re.sub(r'<memory query>.*?</memory query>', '', reflection_content, flags=re.DOTALL)
        clean_reflection = re.sub(r'<memory result>.*?</memory result>', '', clean_reflection, flags=re.DOTALL)
        
        # Store reflection as memory
        if clean_reflection:
            self.memory.store_to_file(f"[Reflection] {clean_reflection.strip()}", episode, step)
        
        return {
            'reflection': clean_reflection.strip(),
            'queries': queries,
            'augmented_text': augmented
        }


class ModularStageProcessor:
    """Main processor that orchestrates all stages."""
    
    def __init__(self, memory_file: str = "memory.md"):
        # Shared memory interface
        self.memory = FileMemory(memory_file)
        
        # Initialize all modules with shared memory
        self.planning = PlanningModule(self.memory)
        self.action = ActionModule(self.memory)
        self.memory_store = MemoryStoreModule(self.memory)
        self.reflection = ReflectionModule(self.memory)
    
    def register_tool(self, name: str, func):
        """Register tool in action module."""
        self.action.register_tool(name, func)
    
    def query_memory(self, query: str, top_k: int = 3) -> str:
        """Query memory - delegate to memory module."""
        return self.memory.query(query, top_k)
    
    def store_memory(self, content: str, episode: str = "", step: int = 0):
        """Store memory - delegate to memory module."""
        self.memory.store_to_file(content, episode, step)
    
    def process_response(self, text: str, episode: str = "", step: int = 0) -> Dict[str, Any]:
        """Process all stages in sequence."""
        results = {
            'original': text,
            'augmented': text
        }
        
        # Process each stage independently
        # 1. Planning
        plan_result = self.planning.process(results['augmented'])
        results['plan'] = plan_result
        if 'augmented_text' in plan_result:
            results['augmented'] = plan_result['augmented_text']
        
        # 2. Action
        action_result = self.action.process(results['augmented'])
        results['action'] = action_result
        if 'augmented_text' in action_result:
            results['augmented'] = action_result['augmented_text']
        
        # 3. Memory Store
        store_result = self.memory_store.process(results['augmented'], episode, step)
        results['memory_store'] = store_result
        
        # 4. Reflection
        reflection_result = self.reflection.process(results['augmented'], episode, step)
        results['reflection'] = reflection_result
        if 'augmented_text' in reflection_result:
            results['augmented'] = reflection_result['augmented_text']
        
        # Extract environment action
        results['env_action'] = action_result.get('for_env', '')
        
        return results
    
    def parse_simple(self, text: str) -> Dict[str, Optional[str]]:
        """Simple parsing for all tags (utility function)."""
        tags = ['plan', 'action', 'memory store', 'reflection', 'think']
        result = {}
        
        for tag in tags:
            # Handle both with and without spaces
            tag_pattern = tag.replace(' ', r'\s*')
            pattern = f'<{tag_pattern}>(.*?)</{tag_pattern}>'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            content = match.group(1).strip() if match else None
            
            # Special handling for action tag to extract action_choice
            if tag == 'action' and content:
                if 'action_choice:' in content:
                    parts = content.split('action_choice:')
                    if len(parts) > 1:
                        action = parts[1].split('\n')[0].strip()
                        # Remove quotes if present
                        action = action.strip("'\"")
                        content = action
                else:
                    # Remove quotes from first line
                    content = content.split('\n')[0].strip().strip("'\"")
            
            result[tag.replace(' ', '_')] = content
        
        return result


# Default tools
def search_tool(params: dict) -> str:
    """Simple search tool."""
    return f"Found: {params.get('query', 'nothing')}"

def calculate_tool(params: dict) -> str:
    """Simple calculator."""
    try:
        result = eval(params.get('expression', '0'), {"__builtins__": {}})
        return f"Result: {result}"
    except:
        return "Error: Invalid expression"

DEFAULT_TOOLS = {
    'search': search_tool,
    'calculate': calculate_tool
}