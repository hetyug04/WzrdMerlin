import asyncio
import logging
import json
import os
import subprocess
import httpx
from typing import Dict, Any, List, Optional
from src.core.mcp.manager import MCPManager

logger = logging.getLogger(__name__)

class ForageManager:
    """
    Implements the Forage protocol for autonomous tool discovery and installation.
    Search registries, install new MCP servers, and persist their configuration.
    """
    def __init__(self, mcp_manager: MCPManager):
        self.mcp_manager = mcp_manager
        self.registry_url = "https://mcp-registry.com/api" # Placeholder registry

    async def forage_search(self, query: str) -> List[Dict[str, Any]]:
        """Search for MCP servers matching the query."""
        logger.info(f"Foraging for tools matching: {query}")
        
        # Real-world search would query community registries like Smithery or MCP Registry
        # For now, we simulate a search result.
        
        # Simulate an external search
        results = [
            {
                "name": "playwright",
                "description": "Full browser automation with Playwright",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-playwright"],
                "repository": "https://github.com/modelcontextprotocol/servers"
            },
            {
                "name": "postgres",
                "description": "Query and manage PostgreSQL databases",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres"],
                "repository": "https://github.com/modelcontextprotocol/servers"
            },
            {
                "name": "filesystem",
                "description": "Safe read/write access to specified directories",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()],
                "repository": "https://github.com/modelcontextprotocol/servers"
            }
        ]
        
        # Filter results based on query (simple substring match)
        filtered = [r for r in results if query.lower() in r["name"].lower() or query.lower() in r["description"].lower()]
        return filtered

    async def forage_install(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Install an MCP server and add it to the active config.
        Requires user authorization in a real scenario.
        """
        logger.info(f"Installing MCP server: {name}")
        
        # In a real environment, we'd verify dependencies (npx, python, etc.)
        # and possibly download/compile code.
        
        # Add to mcp_config.json
        try:
            with open(self.mcp_manager.config_path, "r") as f:
                data = json.load(f)
            
            data["servers"][name] = {
                "command": config["command"],
                "args": config.get("args", []),
                "env": config.get("env", {})
            }
            
            with open(self.mcp_manager.config_path, "w") as f:
                json.dump(data, f, indent=2)
            
            # Re-initialize mcp_manager to pickup new config
            await self.mcp_manager.connect_server(name)
            
            # Persist tool documentation for the 'learn' phase
            await self.forage_learn(name)
            
            return True
        except Exception as e:
            logger.error(f"Failed to install MCP server {name}: {e}")
            return False

    async def forage_learn(self, server_name: str):
        """
        Persist tool usage instructions and schemas to .merlin/rules/.
        This allows future context retrieval to quickly reload tool knowledge.
        """
        rules_dir = os.path.join(".merlin", "rules")
        os.makedirs(rules_dir, exist_ok=True)
        
        server_tools = [t for t in self.mcp_manager.tools.values() if t["server"] == server_name]
        
        rule_path = os.path.join(rules_dir, f"{server_name}.json")
        with open(rule_path, "w") as f:
            json.dump({
                "server": server_name,
                "tools": server_tools,
                "timestamp": asyncio.get_event_loop().time()
            }, f, indent=2)
        
        logger.info(f"Learned tool rules for {server_name} and saved to {rule_path}")
