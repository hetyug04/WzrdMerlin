#!/usr/bin/env python3
"""
merlin_mcp.py — MCP server for interacting with a running WzrdMerlin v3 instance.

Exposes the full Merlin REST API as MCP tools so Claude Code can submit tasks,
observe events, inspect memory, read logs, and cancel stuck tasks.

Setup (one-time):
  claude mcp add merlin-agent -- python C:/CS/sideProjects/WzrdMerlinV3/tools/merlin_mcp.py

Environment variables:
  MERLIN_API              Backend URL (default: http://localhost:8000)
  MERLIN_POLL_TIMEOUT     Max seconds for poll_task to wait (default: 25)
"""
import asyncio
import json
import os
import sys

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = os.getenv("MERLIN_API", "http://localhost:8000").rstrip("/")
POLL_TIMEOUT = float(os.getenv("MERLIN_POLL_TIMEOUT", "25"))

mcp = FastMCP("merlin-agent")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compact(data, max_chars: int = 2500) -> str:
    """
    Convert any response to a context-efficient string.
    - Dicts: keep all keys but truncate long string values
    - Lists: summarise count + first few items
    - Strings: hard cap
    """
    if isinstance(data, str):
        return data[:max_chars] + (f"\n... [{len(data) - max_chars} chars truncated]" if len(data) > max_chars else "")

    if isinstance(data, list):
        count = len(data)
        preview = json.dumps(data[:3], indent=2) if data else "[]"
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "..."
        suffix = f"\n... [{count - 3} more items]" if count > 3 else ""
        return f"[{count} items]\n{preview}{suffix}"

    if isinstance(data, dict):
        # Truncate long string values in-place (shallow)
        truncated = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 300:
                truncated[k] = v[:300] + f"... [{len(v) - 300} chars]"
            elif isinstance(v, list) and len(v) > 10:
                truncated[k] = v[:5] + [f"... {len(v) - 5} more"]
            else:
                truncated[k] = v
        raw = json.dumps(truncated, indent=2)
        if len(raw) > max_chars:
            raw = raw[:max_chars] + "\n... [truncated]"
        return raw

    raw = json.dumps(data, indent=2) if not isinstance(data, str) else data
    if len(raw) > max_chars:
        raw = raw[:max_chars] + "\n... [truncated]"
    return raw


def _fmt_events(events: list) -> str:
    lines = []
    for evt in events:
        t = evt.get("type", "?")
        p = evt.get("payload", {})
        if t in ("agent.streaming", "agent.thinking"):
            snippet = p.get("text", "")[:80].replace("\n", " ")
            lines.append(f"  [{t}] {snippet}")
        elif t in ("agent.tool_start", "agent.tool_end"):
            lines.append(f"  [{t}] tool={p.get('tool', '?')}")
        else:
            lines.append(f"  [{t}]")
    return "\n".join(lines) or "  (none)"


# ── Health ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_health() -> str:
    """Check backend NATS connectivity and overall health."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{API_BASE}/api/health")
            return _compact(r.json())
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend — is docker compose up?"


@mcp.tool()
async def get_llm_health() -> str:
    """Check Ollama model status, active model name, and circuit breaker state."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{API_BASE}/api/llm/health")
            return _compact(r.json())
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."
    except httpx.TimeoutException:
        return "ERROR: LLM health check timed out — Ollama may be unresponsive."


# ── Task submission & polling ─────────────────────────────────────────────────

@mcp.tool()
async def submit_task(description: str) -> str:
    """
    Submit a task to the Merlin agent. Returns immediately with task_id.

    Use poll_task(task_id) to check completion status.
    Also handles the human-in-the-loop case: if the agent is waiting for human
    input, submitting a task resumes it with your text as the response.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{API_BASE}/api/task",
                params={"description": description},
            )
            if r.status_code != 200:
                return f"ERROR HTTP {r.status_code}: {r.text}"
            data = r.json()
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."

    task_id = data.get("task_id", "")
    status = data.get("status", "")

    if status == "responded":
        return f"Resumed waiting task {task_id}. Use poll_task('{task_id}') to watch it."
    if not task_id:
        return f"ERROR: No task_id returned: {data}"
    return f"Task submitted: {task_id}\nUse poll_task('{task_id}') to check completion."


@mcp.tool()
async def poll_task(task_id: str, timeout: int = 25) -> str:
    """
    Poll a submitted task until it completes, fails, or timeout (seconds) is reached.

    Returns the task status + result. Call repeatedly if the task is still running.
    timeout is capped at 25s to avoid MCP client abort errors.
    """
    timeout = min(timeout, 25)
    deadline = asyncio.get_event_loop().time() + timeout
    poll_interval = 2.0

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                r = await client.get(f"{API_BASE}/api/task/{task_id}")
                if r.status_code == 200:
                    data = r.json()
                    status = data.get("status", "unknown")
                    iteration = data.get("iteration", 0)
                    result = data.get("result", "")

                    if status in ("completed", "failed", "cancelled"):
                        return (
                            f"status: {status}\n"
                            f"iterations: {iteration}\n"
                            f"result: {result[:1000]}"
                        )

                    # Still running — wait and retry
                    remaining = deadline - asyncio.get_event_loop().time()
                    wait = min(poll_interval, max(0, remaining))
                    if wait > 0:
                        await asyncio.sleep(wait)
                else:
                    return f"ERROR {r.status_code}: {r.text[:200]}"

        # Timed out — return last known state
        try:
            r = await client.get(f"{API_BASE}/api/task/{task_id}")
            data = r.json()
            return (
                f"TIMEOUT after {timeout}s — task still running\n"
                f"status: {data.get('status', 'unknown')}\n"
                f"iterations: {data.get('iteration', 0)}\n"
                f"Call poll_task('{task_id}') again to continue watching."
            )
        except Exception:
            return f"TIMEOUT after {timeout}s. Call poll_task('{task_id}') again."

    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


@mcp.tool()
async def cancel_task(task_id: str) -> str:
    """
    Cancel an in-flight task. The agent will stop at the next step boundary.
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(f"{API_BASE}/api/task/{task_id}/cancel")
            return _compact(r.json())
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


# ── Logs & Events ─────────────────────────────────────────────────────────────

@mcp.tool()
async def get_logs(lines: int = 50) -> str:
    """
    Return the last N lines from the backend's in-process log buffer.
    Useful for debugging without needing docker exec.
    lines is capped at 100 for context efficiency.
    """
    lines = min(lines, 100)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{API_BASE}/api/logs", params={"lines": lines})
            data = r.json()
            log_lines = data.get("lines", [])
            total = data.get("total_buffered", 0)
            header = f"[{len(log_lines)} of {total} buffered lines]\n"
            return header + "\n".join(log_lines)
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


@mcp.tool()
async def get_task_events(task_id: str = "", limit: int = 20) -> str:
    """
    Return the last N SSE events from the in-memory event snapshot.
    Optionally filter by task_id. Much faster than watching the SSE stream.
    """
    limit = min(limit, 50)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            params = {"limit": limit}
            if task_id:
                params["task_id"] = task_id
            r = await client.get(f"{API_BASE}/api/events/recent", params=params)
            data = r.json()
            events = data.get("events", [])
            count = data.get("count", 0)
            lines = [f"[{count} events" + (f" for task {task_id}" if task_id else "") + "]"]
            for e in events:
                summary = e.get("payload_summary", "")[:100].replace("\n", " ")
                lines.append(f"  [{e['type']}] {e['correlation_id'][:20]}… {summary}")
            return "\n".join(lines)
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


# ── Actor debug ───────────────────────────────────────────────────────────────

@mcp.tool()
async def get_actors() -> str:
    """
    Get actor registry: NATS connection state, subscribed handlers,
    and loaded MCP tools for each actor.
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{API_BASE}/api/debug/actors")
            data = r.json()
            # Compact summary
            lines = []
            for name, info in data.items():
                if isinstance(info, dict):
                    connected = info.get("connected", "?")
                    tools = info.get("mcp_tools", [])
                    handlers = info.get("handlers", [])
                    lines.append(
                        f"{name}: {'✓' if connected else '✗'} connected | "
                        f"{len(handlers)} handlers | {len(tools)} mcp_tools"
                        + (f": {tools}" if tools else "")
                    )
                else:
                    lines.append(f"{name}: {info}")
            return "\n".join(lines)
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


# ── Models ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def list_models() -> str:
    """List all model profiles from merlin.config.yaml and the currently active one."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{API_BASE}/api/models")
            return _compact(r.json())
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


@mcp.tool()
async def switch_model(model_name: str) -> str:
    """
    Hot-swap the active model without restarting.
    model_name must match a profile key in merlin.config.yaml (e.g. 'merlin-brain').
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{API_BASE}/api/models/switch",
                params={"model_name": model_name},
            )
            return _compact(r.json())
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


@mcp.tool()
async def reload_config() -> str:
    """Reload merlin.config.yaml from disk without restarting the container."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{API_BASE}/api/config/reload")
            return _compact(r.json())
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


# ── Memory ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_memory_stats() -> str:
    """Get episodic memory and trajectory document counts."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{API_BASE}/api/memory/stats")
            return _compact(r.json())
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


@mcp.tool()
async def search_memory(query: str, top_k: int = 5, collection: str = "episodic") -> str:
    """
    Full-text search of the agent's memory store.

    Args:
        query: Search query string
        top_k: Number of results to return (default 5)
        collection: 'episodic' (general facts) or 'trajectories' (task traces)
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{API_BASE}/api/memory/search",
                params={"query": query, "top_k": top_k, "collection": collection},
            )
            data = r.json()
            results = data.get("results", [])
            lines = [f"[{len(results)} results for '{query}']"]
            for item in results:
                content = str(item.get("content", ""))[:200]
                score = item.get("score", 0)
                lines.append(f"  [{score:.2f}] {content}")
            return "\n".join(lines)
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


# ── Workspace files ───────────────────────────────────────────────────────────

@mcp.tool()
async def list_workspace_files(path: str = "") -> str:
    """
    List files and directories inside the agent's /workspace volume.
    Pass a relative path (e.g. '.merlin') to browse subdirectories.
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{API_BASE}/api/files", params={"path": path})
            data = r.json()
            entries = data.get("entries", [])
            lines = [f"[{data.get('path', path)} — {len(entries)} entries]"]
            for e in entries:
                size = f"  {e['size']}B" if e.get("size") else ""
                lines.append(f"  {'📁' if e['type'] == 'directory' else '📄'} {e['name']}{size}")
            return "\n".join(lines)
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


@mcp.tool()
async def read_workspace_file(path: str) -> str:
    """
    Read a file from the agent's /workspace volume (max 1 MB).
    Path is relative to /workspace (e.g. '.merlin/mcp_config.json').
    Response is capped at 3000 chars.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{API_BASE}/api/files/read", params={"path": path})
            if r.status_code != 200:
                return f"ERROR {r.status_code}: {r.text}"
            content = r.text
            if len(content) > 3000:
                return content[:3000] + f"\n... [{len(content) - 3000} chars truncated]"
            return content
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


# ── Gardener & self-improvement ───────────────────────────────────────────────

@mcp.tool()
async def get_gardener_status() -> str:
    """Get memory consolidation status: last run time, cooldown remaining, idle duration."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{API_BASE}/api/gardener/status")
            return _compact(r.json())
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


@mcp.tool()
async def trigger_rollback() -> str:
    """Revert the most recent self-improvement merge (git-based rollback)."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(f"{API_BASE}/api/rollback")
            return _compact(r.json())
    except httpx.ConnectError:
        return "ERROR: Cannot reach Merlin backend."


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Merlin MCP server")
    p.add_argument(
        "--api",
        default=os.getenv("MERLIN_API", "http://localhost:8000"),
        help="Merlin backend URL (default: http://localhost:8000)",
    )
    args, _ = p.parse_known_args()
    API_BASE = args.api.rstrip("/")

    mcp.run()
