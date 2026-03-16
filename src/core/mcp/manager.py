"""
WzrdMerlin v3 — MCPManager

v3 fix: replaced `while name in self.sessions: await asyncio.sleep(1)`
(busy-polling loop) with `asyncio.Event()` to avoid wasting a coroutine
spinning every second.
"""
import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPManager:
    """
    Manages multiple MCP server connections over stdio.
    Acts as the central registry for all tools available to the agent.
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(".merlin", "mcp_config.json")
        self.sessions: Dict[str, ClientSession] = {}
        self.server_params: Dict[str, StdioServerParameters] = {}
        self.tools: Dict[str, Any] = {}
        # Per-session close events — used to keep the stdio_client alive without polling
        self._close_events: Dict[str, asyncio.Event] = {}

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                json.dump({"servers": {}}, f)

    async def load_config(self):
        """Load configured MCP servers from disk."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            for name, params in config.get("servers", {}).items():
                self.server_params[name] = StdioServerParameters(
                    command=params["command"],
                    args=params.get("args", []),
                    env=params.get("env"),
                )
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")

    async def connect_all(self):
        """Connect to all configured MCP servers."""
        await self.load_config()
        for name in self.server_params:
            try:
                await self.connect_server(name)
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {name}: {e}")

    async def connect_server(self, name: str):
        """Connect to a specific MCP server by name."""
        if name not in self.server_params:
            raise ValueError(f"Server {name} not found in config.")

        params = self.server_params[name]
        # Per-server close event — set when we want the background task to exit
        close_event = asyncio.Event()
        self._close_events[name] = close_event

        async def _run_client():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.sessions[name] = session

                    tools_resp = await session.list_tools()
                    for tool in tools_resp.tools:
                        full_name = f"{name}__{tool.name}"
                        self.tools[full_name] = {
                            "server": name,
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema,
                        }

                    logger.info(
                        f"Connected to MCP server '{name}' with {len(tools_resp.tools)} tools."
                    )

                    # v3 fix: wait on an Event instead of polling every 1s
                    await close_event.wait()

        asyncio.create_task(_run_client())

        # Wait for session to be initialized (up to 5s)
        for _ in range(50):
            if name in self.sessions:
                return
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Timed out connecting to MCP server {name}")

    async def disconnect_server(self, name: str):
        """Gracefully disconnect a server by signalling its close event."""
        event = self._close_events.pop(name, None)
        if event:
            event.set()
        self.sessions.pop(name, None)
        # Remove all tools for this server
        to_remove = [k for k, v in self.tools.items() if v["server"] == name]
        for k in to_remove:
            del self.tools[k]

    async def call_tool(self, full_name: str, args: Dict[str, Any]) -> Any:
        """Call a tool by its full name (server__tool)."""
        if full_name not in self.tools:
            raise ValueError(f"Tool {full_name} not found.")

        tool_info = self.tools[full_name]
        server_name = tool_info["server"]
        tool_name = tool_info["name"]

        session = self.sessions.get(server_name)
        if not session:
            await self.connect_server(server_name)
            session = self.sessions.get(server_name)
            if not session:
                raise RuntimeError(f"Server {server_name} is not connected.")

        result = await session.call_tool(tool_name, args)
        return result

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return list of tool definitions for LLM context."""
        defs = []
        for full_name, info in self.tools.items():
            defs.append({
                "name": full_name,
                "description": info["description"],
                "parameters": info["input_schema"],
            })
        return defs
