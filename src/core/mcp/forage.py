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
    Forage protocol — autonomous tool discovery.

    forage_search queries the npm registry for community MCP servers so the
    agent can discover what exists externally.  The real power comes from the
    self-improvement pipeline (request_capability → ImprovementManager) which
    lets the agent synthesize brand-new tools from scratch.
    """
    def __init__(self, mcp_manager: MCPManager):
        self.mcp_manager = mcp_manager
        self._npm_search_url = "https://registry.npmjs.org/-/v1/search"

    async def forage_search(self, query: str) -> List[Dict[str, Any]]:
        """Search npm + PyPI for MCP server packages matching the query."""
        logger.info(f"Foraging for tools matching: {query}")

        results = await self._search_npm(query)
        if not results:
            results = await self._search_pypi(query)

        logger.info(f"Forage search returned {len(results)} result(s) for '{query}'")
        return results[:10]

    async def _search_npm(self, query: str) -> List[Dict[str, Any]]:
        """Search npm registry for MCP server packages."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    self._npm_search_url,
                    params={"text": f"modelcontextprotocol server {query}", "size": 15},
                )
                if resp.status_code != 200:
                    logger.warning(f"npm search returned {resp.status_code}")
                    return []
                data = resp.json()
                results = []
                for obj in data.get("objects", []):
                    pkg = obj.get("package", {})
                    name = pkg.get("name", "")
                    if "mcp" not in name.lower() and "modelcontextprotocol" not in name.lower():
                        continue
                    results.append({
                        "name": name,
                        "description": pkg.get("description", ""),
                        "command": "npx",
                        "args": ["-y", name],
                    })
                return results
        except Exception as e:
            logger.warning(f"npm registry search failed: {e}")
            return []

    async def _search_pypi(self, query: str) -> List[Dict[str, Any]]:
        """Search PyPI for MCP-related packages as a fallback."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://pypi.org/search/",
                    params={"q": f"mcp server {query}"},
                    headers={"Accept": "application/json"},
                )
                # PyPI search doesn't have a clean JSON API, so this is best-effort.
                if resp.status_code != 200:
                    return []
                return []
        except Exception:
            return []

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
