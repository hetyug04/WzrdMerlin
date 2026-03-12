import pytest
import os
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from src.core.base_agent import BaseAgentActor
from src.core.mcp.manager import MCPManager
from src.core.mcp.forage import ForageManager

@pytest.mark.asyncio
async def test_mcp_integration():
    # Mock MCP Manager
    with patch("src.core.mcp.manager.stdio_client") as mock_stdio:
        # Set up a mock session with one tool
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = MagicMock(tools=[
            MagicMock(name="test_tool", description="A test tool", inputSchema={})
        ])
        mock_session.call_tool.return_value = "Success"
        
        # This is a bit complex to mock because of the nested context managers in manager.py
        # For a simpler test, let's just mock the MCPManager itself in the agent
        
        agent = BaseAgentActor(nats_url="nats://localhost:4222")
        agent.mcp_manager = MagicMock()
        agent.mcp_manager.tools = {
            "test_server__test_tool": {
                "server": "test_server",
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {}
            }
        }
        agent.mcp_manager.call_tool = AsyncMock(return_value="MCP Success")

        # Test execute_tool with an MCP tool
        result = await agent.execute_tool("test_server__test_tool", {})
        assert result == "MCP Success"
        agent.mcp_manager.call_tool.assert_called_with("test_server__test_tool", {})

@pytest.mark.asyncio
async def test_forage_tools():
    agent = BaseAgentActor(nats_url="nats://localhost:4222")
    agent.forage_manager = MagicMock()
    
    # Test forage_search
    agent.forage_manager.forage_search = AsyncMock(return_value=[{"name": "playwright"}])
    res = await agent.tool_forage_search({"query": "browser"})
    assert "playwright" in res
    
    # Test forage_install
    agent.forage_manager.forage_install = AsyncMock(return_value=True)
    res = await agent.tool_forage_install({"name": "playwright", "config": {"command": "npx"}})
    assert "Successfully installed" in res

@pytest.mark.asyncio
async def test_python_sandbox_tool():
    agent = BaseAgentActor(nats_url="nats://localhost:4222")
    agent.sandbox = MagicMock()
    agent.sandbox.execute_python = AsyncMock(return_value={"status": "success", "stdout": "hello"})
    
    res = await agent.tool_python_sandbox({"code": "print('hello')"})
    assert '"status": "success"' in res
    assert '"stdout": "hello"' in res
