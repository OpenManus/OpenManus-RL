#!/usr/bin/env python
"""
Test module for testing GAIA Environment Server functionality
with a focus on BrowserUseTool integration
"""

import json
import pytest
import asyncio
from unittest import mock

# Import server components to test
from agentenv_gaia.server import GaiaEnvServer, ToolManager
from gaia.browser_use_tool import BrowserUseTool
from gaia.base import ToolResult

class TestToolManager:
    """Tests for the ToolManager class"""
    
    def test_init(self):
        """Test ToolManager initialization"""
        manager = ToolManager()
        assert hasattr(manager, 'browser_use')
        assert isinstance(manager.browser_use, BrowserUseTool)
        
    def test_get_tool_names(self):
        """Test tool name listing"""
        manager = ToolManager()
        names = manager.get_tool_names()
        assert "browser_use" in names
        
    def test_get_tool_by_name(self):
        """Test getting tool by name"""
        manager = ToolManager()
        tool = manager.get_tool_by_name("browser_use")
        assert tool is not None
        assert isinstance(tool, BrowserUseTool)
        
    def test_get_tool_description(self):
        """Test getting tool description"""
        manager = ToolManager()
        description = manager.get_tool_description("browser_use")
        assert description is not None
        assert len(description) > 0

    @pytest.mark.asyncio
    async def test_execute_browser_use_tool(self):
        """Test executing browser_use tool"""
        manager = ToolManager()
        
        # Mock the BrowserUseTool.execute method
        original_execute = manager.browser_use.execute
        
        # Create our mock execute function
        async def mock_execute(**kwargs):
            return ToolResult(output=f"Mocked browser action executed with: {json.dumps(kwargs)}")
        
        # Replace with mock
        manager.browser_use.execute = mock_execute
        
        try:
            # Test execution
            result = await manager.execute_tool("browser_use", action="go_to_url", url="https://example.com")
            assert result is not None
            assert isinstance(result, ToolResult) or isinstance(result, dict)
            
            # Check if our mocked function was called with the right parameters
            if isinstance(result, ToolResult):
                assert "go_to_url" in result.output
                assert "https://example.com" in result.output
            else:  # dict
                assert "go_to_url" in str(result)
                assert "https://example.com" in str(result)
        finally:
            # Restore original method
            manager.browser_use.execute = original_execute

class TestGaiaEnvServer:
    """Tests for the GaiaEnvServer class with BrowserUseTool"""
    
    def setup_method(self):
        """Set up the test environment"""
        self.server = GaiaEnvServer(max_envs=5)
    
    def test_browser_use_in_default_tools(self):
        """Test browser_use is in default tools"""
        from agentenv_gaia.server import DEFAULT_TOOLS
        assert "browser_use" in DEFAULT_TOOLS
    
    def test_create_environment_with_browser_use(self):
        """Test creating environment with browser_use tool"""
        env_id = self.server.create(tool_list=["browser_use"])
        assert env_id is not None
        assert "browser_use" in self.server.env_instances[env_id]["available_tools"]
    
    def test_parse_browser_use_action(self):
        """Test parsing browser_use action"""
        # Test standard format
        action = "Action: browser_use Action Input: {\"action\": \"go_to_url\", \"url\": \"https://example.com\"}"
        action_type, action_input = self.server._parse_action(action)
        assert action_type == "browser_use"
        
        # Test JSON format
        action = json.dumps({
            "tool_name": "browser_use",
            "action": "click_element",
            "index": 1
        })
        action_type, action_input = self.server._parse_action(action)
        assert action_type == "browser_use"
        assert "action" in action_input
        assert action_input["action"] == "click_element"
        assert action_input["index"] == 1
        
        # Test direct format
        action = "browser_use: {\"action\": \"scroll_down\", \"scroll_amount\": 100}"
        action_type, action_input = self.server._parse_action(action)
        assert action_type == "browser_use"
    
    @pytest.mark.asyncio
    async def test_process_browser_use_action(self):
        """Test processing browser_use action with mocked tool"""
        # Create environment
        env_id = self.server.create(tool_list=["browser_use"])
        
        # Mock the execute_tool method to avoid actual browser operations
        original_execute = self.server.tool_manager.execute_tool
        
        async def mock_execute_tool(tool_name, **kwargs):
            if tool_name == "browser_use":
                return ToolResult(output=f"Mocked browser action: {json.dumps(kwargs)}")
            return await original_execute(tool_name, **kwargs)
        
        # Replace with mock
        self.server.tool_manager.execute_tool = mock_execute_tool
        
        try:
            # Test processing a browser action
            observation, reward, done = self.server._process_action(
                env_id, 
                "browser_use", 
                {"action": "go_to_url", "url": "https://example.com"}
            )
            
            assert observation is not None
            assert "Mocked browser action" in observation
            assert reward > 0  # Should get positive reward for successful tool use
            assert not done  # Should not be done
            
            # Verify the action was added to memory
            memory = self.server.env_instances[env_id]["state"]["memory"]
            assert len(memory) > 0
            last_action = memory[-1]
            assert "browser_use" in str(last_action["action"])
            
        finally:
            # Restore original method
            self.server.tool_manager.execute_tool = original_execute

if __name__ == "__main__":
    pytest.main(["-xvs", "test_server.py"]) 