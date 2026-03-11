import logging
import asyncio
import os
import subprocess
import json
import re
import time
import httpx
from pathlib import Path
from typing import Dict, Any, List, Callable, Awaitable, Optional
from src.core.actor import BaseActor
from src.core.events import Event, EventType

logger = logging.getLogger(__name__)

# Persistent workspace — mounted as a Docker volume so it survives restarts
WORKSPACE = os.getenv("MERLIN_WORKSPACE", "/workspace")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
AGENT_REQUIREMENTS = os.path.join(WORKSPACE, "requirements-agent.txt")


class BaseAgentActor(BaseActor):
    """
    The Alita-pattern foundation agent. Ships with exactly 8 core tools.
    Any capability gap encountered is organically queued for self-improvement.

    Persistence:
      - Memory: JSON files in /workspace/memory/ (survives restarts)
      - Packages: tracked in /workspace/requirements-agent.txt (auto-installed on startup)
      - State: NATS KV via StateStore (crash recovery)
    """
    def __init__(self, nats_url: str = None, role: str = "implementer"):
        super().__init__(name=f"agent-{role}", nats_url=nats_url)
        self.role = role
        self._ui_broadcast: Optional[Callable[[Event], Awaitable[None]]] = None
        self.tools = {
            "shell": self.tool_shell,
            "read_file": self.tool_read_file,
            "write_file": self.tool_write_file,
            "search_memory": self.tool_search_memory,
            "write_memory": self.tool_write_memory,
            "fetch_url": self.tool_fetch_url,
            "done": self.tool_done,
            "request_human": self.tool_request_human,
        }
        # Ensure memory directory exists
        os.makedirs(MEMORY_DIR, exist_ok=True)

    async def connect(self):
        await super().connect()
        self.on(f"events.actor.{self.role}", self.handle_action_requested)
        await self.listen(f"events.actor.{self.role}")
        logger.info(f"{self.name} listening for actions.")

    async def handle_action_requested(self, event: Event):
        task_id = event.payload.get("task_id")
        instruction = event.payload.get("instruction")
        logger.info(f"{self.name} executing task {task_id}: {instruction}")

        if "playwright" in instruction.lower() or "browser" in instruction.lower():
            logger.warning(f"Task {task_id} requires browser tools. Emitting CAPABILITY_GAP.")
            await self._emit_capability_gap(task_id, "Requires DOM/Browser interaction (Playwright)")
            return

        await self._execute_react_loop(task_id, instruction)

    async def _emit_capability_gap(self, task_id: str, desc: str):
        gap_event = Event(
            type=EventType.CAPABILITY_GAP,
            source_actor=self.name,
            correlation_id=task_id,
            payload={
                "gap_description": desc,
                "triggering_task": task_id,
                "priority": 1,
            },
        )
        await self.publish("events.capability.gap", gap_event)

    async def _ui_emit(self, event_type: EventType, task_id: str, payload: dict):
        """Send an event directly to UI queues (no NATS overhead for live narration)."""
        if self._ui_broadcast:
            evt = Event(
                type=event_type,
                source_actor=self.name,
                correlation_id=task_id,
                payload=payload,
            )
            await self._ui_broadcast(evt)

    async def _execute_react_loop(self, task_id: str, instruction: str, max_iterations: int = 10):
        from src.core.llm import ModelInterface

        llm = ModelInterface()
        system_prompt = """You are WzrdMerlin Base Agent. Select the right tool for each step.
Tools: shell(cmd), read_file(path), write_file(path,content), search_memory(query), write_memory(content,tags), fetch_url(url), request_human(question), done(summary).

RULES:
- Reply with ONLY a JSON object. No prose before or after.
- All strings in JSON must use double quotes and properly escape special chars.
- For shell commands with quotes, use single quotes inside the command or escape with backslash.
- Keep shell commands simple. Prefer python one-liners over complex grep/sed/awk.

Example: {"tool": "shell", "args": {"cmd": "ls -la"}}
When complete: {"tool": "done", "args": {"summary": "what was accomplished"}}
"""
        history = []
        final_result = None

        for iteration in range(max_iterations):
            logger.info(f"ReAct iteration {iteration + 1}/{max_iterations} for {task_id}")

            await self._ui_emit(EventType.AGENT_THINKING, task_id, {
                "iteration": iteration + 1,
                "status": "generating",
                "text": f"[iteration {iteration + 1}] Deciding next action…",
            })

            think_buf = []
            content_buf = []

            async for chunk_type, text in llm.generate_action_streaming(system_prompt, history, instruction):
                if chunk_type == "think":
                    think_buf.append(text)
                    await self._ui_emit(EventType.AGENT_THINKING, task_id, {"text": text})
                else:
                    content_buf.append(text)
                    await self._ui_emit(EventType.AGENT_STREAMING, task_id, {"text": text})

            full_response = "".join(content_buf)
            think_text = "".join(think_buf)

            if think_text:
                logger.debug(f"Think block ({len(think_text)} chars) for {task_id}")

            action_payload = llm.parse_action(full_response)

            if not action_payload:
                logger.error(f"No valid JSON action in response: {full_response[:200]!r}")
                await self.publish("events.action.completed", Event(
                    type=EventType.ACTION_FAILED,
                    source_actor=self.name,
                    correlation_id=task_id,
                    payload={"task_id": task_id, "status": "failed",
                             "reason": "LLM returned no valid JSON action",
                             "raw": full_response[:500]},
                ))
                return

            tool_name = action_payload.get("tool")
            tool_args = action_payload.get("args", {})
            logger.info(f"Action: {tool_name}({tool_args})")

            await self._ui_emit(EventType.AGENT_TOOL_START, task_id, {
                "tool": tool_name,
                "args": tool_args,
            })

            result = await self.execute_tool(tool_name, tool_args)

            await self._ui_emit(EventType.AGENT_TOOL_END, task_id, {
                "tool": tool_name,
                "result": result[:500] if isinstance(result, str) else str(result)[:500],
            })

            # Cap result in history to protect the context window.
            capped = result[:1500] if isinstance(result, str) else str(result)[:1500]
            if len(str(result)) > 1500:
                capped += f"\n... ({len(str(result))} chars total, truncated)"
            history.append({"action": action_payload, "result": capped})
            final_result = result

            if tool_name == "done":
                break
        else:
            logger.warning(f"ReAct hit max iterations for {task_id}")

        await self.publish("events.action.completed", Event(
            type=EventType.ACTION_COMPLETED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={"task_id": task_id, "status": "success",
                     "iterations": len(history), "result": final_result},
        ))

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self.tools:
            return f"Error: Tool {name} not found. Suggest emitting a CAPABILITY_GAP."

        try:
            return await self.tools[name](args)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    # -------------------------------------------------------------------------
    # The 8 Core Tools
    # -------------------------------------------------------------------------

    async def tool_shell(self, args: Dict[str, Any]) -> str:
        cmd = args.get("cmd", "")
        logger.info(f"Executing: {cmd}")

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=120,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = ((stdout or b"") + (stderr or b"")).decode(errors="replace").strip()
            returncode = proc.returncode or 0
        except asyncio.TimeoutError:
            return "Error: command timed out after 120s"

        # Track pip/apt installs so they survive container restarts
        self._track_package_install(cmd, output, returncode)

        return output if output else "(no output)"

    def _track_package_install(self, cmd: str, output: str, returncode: int):
        """
        If the agent ran a pip install and it succeeded, record the package
        in /workspace/requirements-agent.txt so the entrypoint script can
        reinstall it on container restart.
        """
        if returncode != 0:
            return

        # Match: pip install <packages>, pip3 install <packages>
        pip_match = re.match(r"(?:pip3?|python -m pip)\s+install\s+(.+)", cmd)
        if not pip_match:
            return

        raw_args = pip_match.group(1).split()
        # Filter out flags like -U, --upgrade, --user, -q, etc.
        packages = [a for a in raw_args if not a.startswith("-")]
        if not packages:
            return

        # Read existing tracked packages
        existing = set()
        if os.path.exists(AGENT_REQUIREMENTS):
            with open(AGENT_REQUIREMENTS, "r") as f:
                existing = {line.strip() for line in f if line.strip() and not line.startswith("#")}

        # Append new ones
        new_packages = [p for p in packages if p not in existing]
        if new_packages:
            with open(AGENT_REQUIREMENTS, "a") as f:
                for pkg in new_packages:
                    f.write(f"{pkg}\n")
            logger.info(f"Tracked new packages for persistence: {new_packages}")

    async def tool_read_file(self, args: Dict[str, Any]) -> str:
        path = args.get("path", "")
        with open(path, "r") as f:
            content = f.read()
        # Truncate large files to fit in the model's context window
        if len(content) > 8000:
            content = content[:8000] + f"\n\n... (truncated, {len(content)} chars total)"
        return content

    async def tool_write_file(self, args: Dict[str, Any]) -> str:
        path = args.get("path", "")
        content = args.get("content", "")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to {path}"

    async def tool_search_memory(self, args: Dict[str, Any]) -> str:
        """
        Search persistent episodic memory stored in /workspace/memory/.
        Each memory is a JSON file with content, tags, and timestamp.
        Returns the top matches by substring match on content and tags.
        """
        query = args.get("query", "").lower()
        if not query:
            return "Error: query is required"

        matches = []
        memory_path = Path(MEMORY_DIR)
        if not memory_path.exists():
            return "No memories stored yet."

        for fp in memory_path.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
                content = data.get("content", "")
                tags = data.get("tags", [])
                tag_str = " ".join(tags)
                searchable = f"{content} {tag_str}".lower()

                # Score: count how many query words appear
                query_words = query.split()
                score = sum(1 for w in query_words if w in searchable)

                if score > 0:
                    matches.append((score, data))
            except (json.JSONDecodeError, OSError):
                continue

        if not matches:
            return f"No memories matching '{query}' found."

        # Sort by score descending, take top 5
        matches.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, data in matches[:5]:
            tags = ", ".join(data.get("tags", []))
            ts = data.get("timestamp", "?")
            content = data.get("content", "")[:300]
            results.append(f"[{ts}] (tags: {tags}) {content}")

        return "\n---\n".join(results)

    async def tool_write_memory(self, args: Dict[str, Any]) -> str:
        """
        Write to persistent episodic memory in /workspace/memory/.
        Stored as timestamped JSON files so they survive restarts.
        """
        content = args.get("content", "")
        tags = args.get("tags", [])
        if not content:
            return "Error: content is required"

        os.makedirs(MEMORY_DIR, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        memory_id = f"{int(time.time())}_{hash(content) & 0xFFFFFF:06x}"

        entry = {
            "id": memory_id,
            "content": content,
            "tags": tags,
            "timestamp": timestamp,
        }

        filepath = os.path.join(MEMORY_DIR, f"{memory_id}.json")
        with open(filepath, "w") as f:
            json.dump(entry, f, indent=2)

        logger.info(f"Memory written: {filepath} (tags: {tags})")
        return f"Written to episodic memory (id: {memory_id}, tags: {tags})"

    async def tool_fetch_url(self, args: Dict[str, Any]) -> str:
        """Fetch a URL and return its content."""
        url = args.get("url", "")
        if not url:
            return "Error: url is required"

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
                content_type = resp.headers.get("content-type", "")

                if "text" in content_type or "json" in content_type or "xml" in content_type:
                    body = resp.text
                    if len(body) > 8000:
                        body = body[:8000] + f"\n\n... (truncated, {len(body)} chars total)"
                    return f"HTTP {resp.status_code}\n{body}"
                else:
                    return f"HTTP {resp.status_code} ({content_type}, {len(resp.content)} bytes binary)"
        except httpx.ConnectError:
            return f"Error: Could not connect to {url}"
        except httpx.TimeoutException:
            return f"Error: Request to {url} timed out"
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"

    async def tool_done(self, args: Dict[str, Any]) -> str:
        summary = args.get("summary", "")
        logger.info(f"Task Completed: {summary}")
        return "Task marked complete."

    async def tool_request_human(self, args: Dict[str, Any]) -> str:
        question = args.get("question", "")
        logger.info(f"Interrupting for human: {question}")
        return "Human input recorded."
