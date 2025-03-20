# tool_manager.py
from typing import Any, Dict, List, Callable, Optional, Union
import inspect

class ToolManager:
    """
    Manages the registration and execution of tools for the agent.
    """
    
    def __init__(self):
        self.tools = {}
        self.tool_descriptions = {}
    
    def register_tool(self, func: Callable, name: Optional[str] = None, description: str = ""):
        """
        Register a tool function.
        
        Args:
            func: The function to register
            name: Optional name for the tool (defaults to function name)
            description: Description of what the tool does
        """
        tool_name = name if name else func.__name__
        self.tools[tool_name] = func
        
        # Create description with parameter info
        sig = inspect.signature(func)
        params = []
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                params.append(f"{param_name}")
            else:
                params.append(f"{param_name}={param.default}")
        
        param_str = ", ".join(params)
        full_desc = f"{tool_name}({param_str}): {description}"
        
        self.tool_descriptions[tool_name] = full_desc
    
    def get_tool_descriptions(self) -> List[str]:
        """Get descriptions of all registered tools."""
        return list(self.tool_descriptions.values())
    
    def get_tool_names(self) -> List[str]:
        """Get names of all registered tools."""
        return list(self.tools.keys())
    
    def call_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Call a tool by name with given arguments.
        
        Args:
            tool_name: Name of the tool to call
            *args, **kwargs: Arguments to pass to the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If the tool doesn't exist or arguments are invalid
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool_func = self.tools[tool_name]
        
        try:
            result = tool_func(*args, **kwargs)
            return result
        except Exception as e:
            raise ValueError(f"Error calling tool {tool_name}: {str(e)}")
    
    def get_prompt_instructions(self) -> str:
        """
        Get formatted instructions for including in the agent prompt.
        """
        tool_descs = "\n".join([f"- {desc}" for desc in self.get_tool_descriptions()])
        
        instructions = f"""You have access to the following tools:

{tool_descs}

To use a tool, respond with:
<action>tool_name(arg1, arg2, ...)</action>

After gathering information and completing your reasoning, provide your final response with:
<response>Your final response here</response>

Follow the ReAct format:
1. Reason: Think through the problem step by step
2. Action: Use tools when you need information
3. Observation: Review the tool's output
4. Repeat steps 1-3 as needed
5. Conclude with your final response
"""
        return instructions