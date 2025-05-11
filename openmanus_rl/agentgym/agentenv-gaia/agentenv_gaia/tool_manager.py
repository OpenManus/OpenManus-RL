# Import real tools from gaia directory
from typing import Dict, List
from gaia.base import ToolResult
from gaia.bash import Bash
from gaia.browser_use_tool import BrowserUseTool
from gaia.python_execute import PythonExecute
from gaia.web_search import WebSearch


# Tool Initialization
class ToolManager:
    """Manager for GAIA tools"""

    def __init__(self):
        # Initialize tool instances
        self.web_search = WebSearch()
        self.bash = Bash()
        self.python_execute = PythonExecute()
        self.browser_use = BrowserUseTool()

        # Define terminate as a special handler (not a real tool class)
        self.terminate = self._terminate_handler

    async def _terminate_handler(self, **kwargs) -> Dict:
        """Handle terminate action"""
        return {"observation": "Task terminated.", "success": True}

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict:
        """Execute a tool by name with given parameters"""
        tool_name = tool_name.lower().replace(" ", "_")

        if tool_name == "web_search" or tool_name == "search_web":
            return await self.web_search.execute(**kwargs)
        elif tool_name == "bash":
            return await self.bash.execute(**kwargs)
        elif tool_name == "python_execute":
            return await self.python_execute.execute(**kwargs)
        elif tool_name == "browser_use":
            return await self.browser_use.execute(**kwargs)
        elif tool_name == "terminate":
            return await self.terminate(**kwargs)
        else:
            return {"observation": f"Unknown tool: {tool_name}", "success": False}

    def get_tool_names(self) -> List[str]:
        """Get list of available tool names"""
        return ["web_search", "bash", "python_execute", "browser_use", "terminate"]

    def get_tool_by_name(self, name: str):
        """Get tool instance by name"""
        name = name.lower().replace(" ", "_")
        if name == "web_search" or name == "search_web":
            return self.web_search
        elif name == "bash":
            return self.bash
        elif name == "python_execute":
            return self.python_execute
        elif name == "browser_use":
            return self.browser_use
        return None

    def get_tool_description(self, name: str) -> str:
        """Get tool description by name"""
        tool = self.get_tool_by_name(name)
        if tool:
            return tool.description
        elif name == "terminate":
            return "Terminate the current task and submit a final answer."
        return "Unknown tool"
