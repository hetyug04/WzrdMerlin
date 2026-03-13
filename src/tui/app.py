"""
WzrdMerlin v2 — Claude Code-style CLI
Plain rich + httpx streaming. No Textual widgets, no event-bubbling issues.

Uses /api/task (POST) to submit tasks and /api/stream (SSE) for live events.
"""
import asyncio
import json
import httpx
import textwrap
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.markup import escape
from rich.text import Text
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML

API_BASE = "http://localhost:8000"
console = Console()


@dataclass
class CliSession:
    show_thinking: bool = True
    show_raw_tokens: bool = False
    # Accumulated state from SSE stream
    telemetry: Dict[str, Any] = field(default_factory=lambda: {
        "vram": 0.0, "vram_total": 12.0, "ram": 0.0, "cpu": 0.0,
        "tokens_per_sec": 0.0, "latency_ms": 0.0, "temperature": 0,
    })
    capability_events: List[dict] = field(default_factory=list)
    sandbox_logs: List[str] = field(default_factory=list)
    # Per-task event queues: task_id -> asyncio.Queue
    task_queues: Dict[str, asyncio.Queue] = field(default_factory=dict)
    # Most recent actor debug snapshot
    actors: dict = field(default_factory=dict)
    # SSE background task
    sse_task: Optional[asyncio.Task] = None
    # Whether a task is waiting for human input
    waiting_for_human: bool = False
    waiting_task_id: Optional[str] = None
    # SSE connection readiness
    sse_connected: bool = False
    # Circular event buffer for replay (handles race between POST and queue registration)
    _event_buffer: List[dict] = field(default_factory=list)
    _event_buffer_max: int = 200


# ── Panel builders ────────────────────────────────────────────────────────────

def _build_thinking_panel(thinking_text: str, max_lines: int = 8) -> Panel:
    wrapped: List[str] = []
    for raw_line in thinking_text.splitlines() or [thinking_text]:
        wrapped.extend(textwrap.wrap(raw_line, width=90) or [""])
    tail = wrapped[-max_lines:] if wrapped else ["Thinking…"]
    body = "\n".join(escape(line) for line in tail if line is not None)
    if not body.strip():
        body = "Thinking…"
    return Panel(body, title="Thinking", border_style="blue", padding=(0, 1))


def _build_observation_panel(result_text: str, max_lines: int = 10) -> Panel:
    wrapped: List[str] = []
    for raw_line in (result_text or "").splitlines() or [result_text]:
        wrapped.extend(textwrap.wrap(raw_line, width=90) or [""])
    tail = wrapped[-max_lines:] if wrapped else ["(no output)"]
    body = "\n".join(escape(line) for line in tail)
    if not body.strip():
        body = "(no output)"
    return Panel(body, title="Observation", border_style="green", padding=(0, 1))


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _build_context_panel(context_window_tokens: int, used_tokens: int) -> Panel:
    if context_window_tokens <= 0:
        context_window_tokens = 8192
    used = max(0, min(used_tokens, context_window_tokens))
    remaining = max(0, context_window_tokens - used)
    ratio = used / context_window_tokens
    bar_width = 36
    filled = int(bar_width * ratio)
    bar = "█" * filled + "░" * (bar_width - filled)
    body = (
        f"[cyan]{bar}[/]  "
        f"used [bold]{used}[/]/{context_window_tokens} tok  "
        f"left [bold]{remaining}[/] tok"
    )
    return Panel(body, title="Context", border_style="cyan", padding=(0, 1))


def _summarize_thinking(think_text: str) -> str:
    cleaned = " ".join((think_text or "").split())
    if not cleaned:
        return "Considered options and selected the next action."
    shortened = cleaned[:120].rstrip(" ,.;:-")
    if len(cleaned) > 120:
        shortened += "…"
    return shortened


def _is_iteration_marker(text: str) -> bool:
    """Detect the iteration/step marker that starts a new ReAct cycle."""
    return "[iteration" in text or "[step" in text


# ── API helpers ───────────────────────────────────────────────────────────────

async def check_health() -> str:
    """Check backend + NATS + LLM health, return status string."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{API_BASE}/api/health")
            h = r.json()
            nats = h.get("nats", "?")
            nats_s = f"[green]{nats}[/]" if nats == "connected" else f"[red]{nats}[/]"

            # Also check LLM health
            llm_s = ""
            try:
                r2 = await client.get(f"{API_BASE}/api/llm/health")
                llm = r2.json()
                loaded = llm.get("model_loaded", False)
                model = llm.get("active_model", "?")
                if loaded:
                    llm_s = f"  model [green]{model}[/]"
                else:
                    llm_s = f"  model [yellow]{model} (not loaded)[/]"
            except Exception:
                llm_s = "  model [dim]unknown[/]"

            return f"NATS {nats_s}  backend [green]ok[/]{llm_s}"
    except httpx.ConnectError:
        return "[red]backend unreachable — is docker-compose up?[/]"
    except Exception as e:
        return f"[yellow]status error: {e}[/]"


async def get_debug_actors() -> dict:
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{API_BASE}/api/debug/actors")
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}
            return resp.json()
    except httpx.ConnectError:
        return {"error": "backend unreachable"}
    except Exception as e:
        return {"error": str(e)}


async def list_files(path: str = "") -> dict:
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{API_BASE}/api/files",
                params={"path": path},
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}
            return resp.json()
    except httpx.ConnectError:
        return {"error": "backend unreachable"}
    except Exception as e:
        return {"error": str(e)}


async def read_file(path: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{API_BASE}/api/files/read",
                params={"path": path},
            )
            if resp.status_code != 200:
                return f"Error {resp.status_code}: {resp.text}"
            return resp.text
    except httpx.ConnectError:
        return "Error: backend unreachable"
    except Exception as e:
        return f"Error: {e}"


async def trigger_rollback() -> dict:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{API_BASE}/api/rollback")
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}
            return resp.json()
    except httpx.ConnectError:
        return {"error": "backend unreachable"}
    except Exception as e:
        return {"error": str(e)}


# ── Background SSE listener ───────────────────────────────────────────────────

async def sse_listener(session_cfg: CliSession) -> None:
    """Background task: subscribe to /api/stream and dispatch events to task queues.

    Buffers all events in a circular buffer so that run_task() can replay
    any events that arrived between the POST response and queue registration.
    """
    while True:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
            ) as client:
                async with client.stream("GET", f"{API_BASE}/api/stream") as resp:
                    session_cfg.sse_connected = True
                    async for raw_line in resp.aiter_lines():
                        if not raw_line.startswith("data: "):
                            continue
                        data_str = raw_line[6:].strip()
                        if not data_str or data_str == "keep-alive":
                            continue
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type", "")
                        correlation_id = event.get("correlation_id", "")
                        payload = event.get("payload", {})

                        # Buffer ALL events for replay on queue registration
                        session_cfg._event_buffer.append(event)
                        if len(session_cfg._event_buffer) > session_cfg._event_buffer_max:
                            session_cfg._event_buffer = session_cfg._event_buffer[-session_cfg._event_buffer_max:]

                        # Update telemetry from heartbeat
                        if event_type == "system.heartbeat":
                            session_cfg.telemetry.update({
                                "vram": payload.get("vram_used", session_cfg.telemetry["vram"]),
                                "vram_total": payload.get("vram_total", session_cfg.telemetry["vram_total"]),
                                "ram": payload.get("ram_usage", session_cfg.telemetry["ram"]),
                                "cpu": payload.get("cpu_usage", session_cfg.telemetry["cpu"]),
                                "tokens_per_sec": payload.get("tokens_per_sec", session_cfg.telemetry["tokens_per_sec"]),
                                "latency_ms": payload.get("latency_ms", session_cfg.telemetry["latency_ms"]),
                                "temperature": payload.get("temperature", session_cfg.telemetry["temperature"]),
                            })

                        # Accumulate capability events
                        if event_type in ("capability.gap", "improvement.queued", "improvement.deployed"):
                            session_cfg.capability_events.insert(0, event)
                            session_cfg.capability_events = session_cfg.capability_events[:20]

                        # Sandbox log feed
                        if event_type == "agent.tool_start":
                            tool = payload.get("tool", "?")
                            session_cfg.sandbox_logs.append(f"RUNNING: {tool}")
                            session_cfg.sandbox_logs = session_cfg.sandbox_logs[-50:]
                        elif event_type == "agent.tool_end":
                            tool = payload.get("tool", "?")
                            session_cfg.sandbox_logs.append(f"DONE: {tool}")
                            session_cfg.sandbox_logs = session_cfg.sandbox_logs[-50:]

                        # Route to per-task queue if registered
                        if correlation_id and correlation_id in session_cfg.task_queues:
                            await session_cfg.task_queues[correlation_id].put(event)

        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
            session_cfg.sse_connected = False
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            return
        except Exception:
            session_cfg.sse_connected = False
            await asyncio.sleep(5)


# ── Task rendering ────────────────────────────────────────────────────────────

async def run_task(description: str, session_cfg: CliSession) -> None:
    """Submit a task via /api/task and render events from the SSE stream."""

    # Wait briefly for SSE to be connected if it isn't yet
    if not session_cfg.sse_connected:
        console.print("  [dim]Waiting for event stream connection…[/]")
        for _ in range(10):
            if session_cfg.sse_connected:
                break
            await asyncio.sleep(0.5)
        if not session_cfg.sse_connected:
            console.print("  [yellow]Warning: Event stream not connected. Events may be missed.[/]")

    # Submit the task
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{API_BASE}/api/task",
                params={"description": description},
            )
            if resp.status_code != 200:
                console.print(f"  [red]HTTP {resp.status_code}: {resp.text}[/]")
                return
            data = resp.json()
            status = data.get("status", "")
            task_id = data.get("task_id", "")

            if status == "responded":
                console.print(f"  [dim]Response sent to task {task_id}[/]\n")
            elif not task_id:
                console.print(f"  [red]No task_id returned: {data}[/]")
                return

    except httpx.ConnectError:
        console.print("  [red]Cannot reach backend.[/]")
        return

    # Register a queue for this task's events
    q: asyncio.Queue = asyncio.Queue()
    session_cfg.task_queues[task_id] = q

    # Replay any buffered events that arrived before queue registration (race fix)
    for evt in session_cfg._event_buffer:
        if evt.get("correlation_id") == task_id:
            await q.put(evt)

    try:
        await _render_task_events(task_id, q, session_cfg)
    finally:
        session_cfg.task_queues.pop(task_id, None)


async def _render_task_events(
    task_id: str, q: asyncio.Queue, session_cfg: CliSession
) -> None:
    """Read from per-task queue and render events Rich-style."""
    output_blocks: List[Any] = []

    context_window_tokens = int(os.getenv("MODEL_CONTEXT_WINDOW", "8196"))
    used_tokens_est = 0
    last_context_rendered_used = -1
    context_update_step = 128

    thinking_panel_index: Optional[int] = None
    thinking_buffer = ""
    thinking_has_explicit_content = False
    shown_request_human = False

    def rebuild(live: Live, *, force_context: bool = False) -> None:
        nonlocal last_context_rendered_used
        if force_context or last_context_rendered_used < 0:
            last_context_rendered_used = used_tokens_est
        ctx = _build_context_panel(context_window_tokens, last_context_rendered_used)
        live.update(Group(ctx, *output_blocks), refresh=True)

    def push(renderable: Any, live: Live) -> None:
        output_blocks.append(renderable)
        rebuild(live, force_context=False)

    def maybe_refresh_context(live: Live) -> None:
        nonlocal last_context_rendered_used
        if (
            last_context_rendered_used < 0
            or abs(used_tokens_est - last_context_rendered_used) >= context_update_step
        ):
            last_context_rendered_used = used_tokens_est
            rebuild(live, force_context=True)

    def finish_thinking_box(live: Live) -> None:
        nonlocal thinking_panel_index, thinking_buffer, thinking_has_explicit_content
        if thinking_panel_index is not None:
            if thinking_has_explicit_content and thinking_buffer.strip():
                summary = _summarize_thinking(thinking_buffer)
                output_blocks[thinking_panel_index] = Text(f"  Thought: {summary}", style="dim")
            else:
                output_blocks[thinking_panel_index] = Text("")
            rebuild(live, force_context=False)
            thinking_panel_index = None
            thinking_buffer = ""
            thinking_has_explicit_content = False

    def ensure_thinking_box(live: Live) -> None:
        nonlocal thinking_panel_index, thinking_buffer
        if session_cfg.show_thinking and thinking_panel_index is None:
            thinking_buffer = ""
            output_blocks.append(_build_thinking_panel(thinking_buffer))
            thinking_panel_index = len(output_blocks) - 1
            rebuild(live, force_context=False)

    with Live(
        console=console, refresh_per_second=8,
        transient=False, screen=False, auto_refresh=False,
    ) as live:
        rebuild(live, force_context=True)

        IDLE_TIMEOUT = 120.0
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=IDLE_TIMEOUT)
                except asyncio.TimeoutError:
                    push(Text("  Timed out waiting for events.", style="yellow"), live)
                    break

                t = event.get("type", "")
                payload = event.get("payload", {})

                # ── agent.thinking ──────────────────────────────────────────
                if t == "agent.thinking":
                    text = payload.get("text", "")
                    if _is_iteration_marker(text):
                        # New ReAct iteration — close previous thinking box, open new one
                        finish_thinking_box(live)
                        if session_cfg.show_thinking:
                            ensure_thinking_box(live)
                        else:
                            push(Text("  Thinking…", style="dim"), live)
                    else:
                        # Actual thinking content — ensure box exists and append
                        if session_cfg.show_thinking:
                            if thinking_panel_index is None:
                                ensure_thinking_box(live)
                            thinking_has_explicit_content = True
                            thinking_buffer += text
                            used_tokens_est += _estimate_tokens(text)
                            output_blocks[thinking_panel_index] = _build_thinking_panel(thinking_buffer)
                            maybe_refresh_context(live)
                            rebuild(live, force_context=False)

                # ── agent.streaming (narrative tokens) ──────────────────────
                elif t == "agent.streaming":
                    text = payload.get("text", "")
                    used_tokens_est += _estimate_tokens(text)
                    maybe_refresh_context(live)
                    if session_cfg.show_raw_tokens and text.strip():
                        push(Text(f"    {escape(text)}", style="dim"), live)

                # ── agent.tool_start ────────────────────────────────────────
                elif t == "agent.tool_start":
                    finish_thinking_box(live)
                    tool = payload.get("tool", "?")
                    args = payload.get("args", {})

                    if tool == "request_human":
                        question = (
                            args.get("question")
                            or payload.get("question")
                            or payload.get("input", "")
                        )
                        if question and not shown_request_human:
                            shown_request_human = True
                            push(
                                Panel(
                                    escape(str(question)),
                                    title="[bold yellow]Merlin asks[/]",
                                    border_style="yellow",
                                    padding=(0, 1),
                                ),
                                live,
                            )
                        session_cfg.waiting_for_human = True
                        session_cfg.waiting_task_id = task_id
                    elif tool != "done":
                        arg_preview = ""
                        if args:
                            first_val = next(iter(args.values()), "")
                            arg_preview = str(first_val)[:100]
                        push(Text(f"  Action {tool}  {escape(arg_preview)}", style="cyan"), live)

                # ── agent.tool_end ──────────────────────────────────────────
                elif t == "agent.tool_end":
                    finish_thinking_box(live)
                    tool = payload.get("tool", "")
                    if tool != "done":
                        result = str(payload.get("result", ""))[:800]
                        push(_build_observation_panel(result), live)

                # ── task.completed / action.completed ───────────────────────
                elif t in ("task.completed", "action.completed"):
                    finish_thinking_box(live)
                    session_cfg.waiting_for_human = False
                    session_cfg.waiting_task_id = None
                    if not shown_request_human:
                        result = payload.get("result", "")
                        summary = result if isinstance(result, str) else json.dumps(result)
                        push(Text(f"\n  Done. {escape(summary)}\n", style="green"), live)
                    else:
                        push(Text("\n  Done.\n", style="green"), live)
                    break

                # ── task.failed / action.failed ─────────────────────────────
                elif t in ("task.failed", "action.failed"):
                    finish_thinking_box(live)
                    session_cfg.waiting_for_human = False
                    reason = payload.get("reason", "unknown error")
                    push(Text(f"\n  Error: {escape(str(reason))}\n", style="red"), live)
                    break

                # ── task.routed (informational) ─────────────────────────────
                elif t == "task.routed":
                    target = payload.get("target_actor", "?")
                    push(Text(f"  Routed → {escape(target)}", style="dim"), live)

        except asyncio.CancelledError:
            finish_thinking_box(live)
            push(Text("  Cancelled.\n", style="yellow"), live)


# ── Command handlers ──────────────────────────────────────────────────────────

def _fmt_size(n: Optional[int]) -> str:
    if n is None:
        return ""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n/1024:.1f}KB"
    return f"{n/1024/1024:.1f}MB"


async def cmd_hw(session_cfg: CliSession) -> None:
    # Also try to pull live data from /api/llm/health
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{API_BASE}/api/llm/health")
            if resp.status_code == 200:
                llm = resp.json()
                if llm.get("tokens_per_sec"):
                    session_cfg.telemetry["tokens_per_sec"] = llm["tokens_per_sec"]
                if llm.get("active_model"):
                    session_cfg.telemetry["active_model"] = llm["active_model"]
    except Exception:
        pass

    t = session_cfg.telemetry
    vram = t["vram"]
    vram_total = t["vram_total"]
    vram_pct = (vram / vram_total * 100) if vram_total > 0 else 0
    bar_w = 30
    filled = int(bar_w * vram_pct / 100)
    vram_bar = "█" * filled + "░" * (bar_w - filled)
    ram_pct = t["ram"]
    ram_filled = int(bar_w * ram_pct / 100) if ram_pct > 0 else 0
    ram_bar = "█" * ram_filled + "░" * (bar_w - ram_filled)
    body = (
        f"VRAM  [yellow]{vram_bar}[/]  {vram:.1f}/{vram_total}GB  ({vram_pct:.0f}%)\n"
        f"RAM   [blue]{ram_bar}[/]  {ram_pct:.0f}%\n"
        f"CPU   {t['cpu']:.1f}%\n"
        f"GPU   {t['temperature']}°C\n"
        f"Speed [green]{t['tokens_per_sec']:.1f}[/] t/s   latency [green]{t['latency_ms']:.0f}[/] ms"
    )
    model = t.get("active_model", "")
    if model:
        body += f"\nModel [bold]{escape(model)}[/]"
    console.print(Panel(body, title="Hardware Telemetry", border_style="yellow", padding=(0, 1)))
    console.print()


async def cmd_actors(session_cfg: CliSession) -> None:
    result = await get_debug_actors()
    err = result.get("error")
    if err:
        console.print(f"  [red]actors unavailable:[/] {escape(str(err))}\n")
        return

    table = Table(title="Actors", border_style="cyan", show_lines=False, padding=(0, 1))
    table.add_column("Actor", style="bold")
    table.add_column("NATS", justify="center")
    table.add_column("Handlers", style="dim")
    table.add_column("MCP Tools")

    for name, info in result.items():
        if name in ("ui_listeners",):
            continue
        if not isinstance(info, dict):
            continue
        connected = info.get("connected", False)
        conn_s = "[green]✓[/]" if connected else "[red]✗[/]"
        handlers = ", ".join(info.get("handlers", []))
        mcp = str(len(info.get("mcp_tools", []))) + " tools"
        table.add_row(name, conn_s, handlers or "-", mcp)

    ui_count = result.get("ui_listeners", 0)
    console.print(table)
    console.print(f"  [dim]UI listeners: {ui_count}[/]\n")


async def cmd_tools(session_cfg: CliSession) -> None:
    result = await get_debug_actors()
    err = result.get("error")
    if err:
        console.print(f"  [red]tools unavailable:[/] {escape(str(err))}\n")
        return

    base_tools = ["shell", "read_file", "write_file", "search_memory",
                  "write_memory", "fetch_url", "done", "request_human"]

    mcp_tools: List[str] = []
    for name, info in result.items():
        if isinstance(info, dict):
            for tool in info.get("mcp_tools", []):
                if tool not in mcp_tools:
                    mcp_tools.append(tool)

    table = Table(title="Tool Registry", border_style="magenta", padding=(0, 1))
    table.add_column("Tool", style="bold")
    table.add_column("Type")

    for t in base_tools:
        table.add_row(t, "[blue]base[/]")
    for t in mcp_tools:
        table.add_row(t, "[yellow]mcp[/]")

    console.print(table)
    console.print()


async def cmd_caps(session_cfg: CliSession) -> None:
    evts = session_cfg.capability_events
    if not evts:
        console.print("  [dim]No capability events yet (they arrive via /api/stream).[/]\n")
        return

    table = Table(title="Capability Events", border_style="yellow", padding=(0, 1))
    table.add_column("Type", style="bold")
    table.add_column("Detail")
    table.add_column("Task", style="dim")

    for evt in evts[:15]:
        t = evt.get("type", "?")
        p = evt.get("payload", {})
        detail = p.get("gap_description") or p.get("tool_name") or json.dumps(p)[:60]
        task = p.get("triggering_task", p.get("task_id", "-"))[:20]
        color = {"capability.gap": "red", "improvement.queued": "blue", "improvement.deployed": "green"}.get(t, "")
        table.add_row(f"[{color}]{t}[/]" if color else t, escape(detail), escape(task))

    console.print(table)
    console.print()


async def cmd_sandbox(session_cfg: CliSession) -> None:
    logs = session_cfg.sandbox_logs
    if not logs:
        console.print("  [dim]No sandbox logs yet.[/]\n")
        return
    body = "\n".join(escape(line) for line in logs[-40:])
    console.print(Panel(body, title="Sandbox Feed", border_style="green", padding=(0, 1)))
    console.print()


async def cmd_files(path: str) -> None:
    result = await list_files(path)
    err = result.get("error")
    if err:
        console.print(f"  [red]files error:[/] {escape(str(err))}\n")
        return

    entries = result.get("entries", [])
    display_path = result.get("path", path or "/workspace")
    if not entries:
        console.print(f"  [dim]{display_path} is empty.[/]\n")
        return

    table = Table(title=f"[cyan]{escape(display_path)}[/]", border_style="cyan", padding=(0, 1))
    table.add_column("Name", style="bold")
    table.add_column("Type", justify="center")
    table.add_column("Size", justify="right", style="dim")

    for entry in entries:
        name = entry.get("name", "?")
        etype = entry.get("type", "file")
        size = _fmt_size(entry.get("size"))
        icon = "[yellow]d[/]" if etype == "directory" else "[blue]f[/]"
        name_s = f"[yellow]{escape(name)}/[/]" if etype == "directory" else escape(name)
        table.add_row(name_s, icon, size)

    console.print(table)
    console.print()


async def cmd_read(path: str) -> None:
    if not path:
        console.print("  usage: /read <path>\n")
        return
    content = await read_file(path)
    lines = content.splitlines()
    if len(lines) > 100:
        preview = "\n".join(lines[:100]) + f"\n[dim]… ({len(lines) - 100} more lines)[/]"
    else:
        preview = content
    console.print(Panel(
        escape(preview) if "more lines" not in preview else preview,
        title=f"[cyan]{escape(path)}[/]",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print()


async def cmd_debug() -> None:
    result = await get_debug_actors()
    err = result.get("error")
    if err:
        console.print(f"  [red]debug unavailable:[/] {escape(str(err))}\n")
        return
    console.print(Panel(
        escape(json.dumps(result, indent=2)),
        title="Actor Debug",
        border_style="magenta",
        padding=(0, 1),
    ))
    console.print()


async def cmd_rollback() -> None:
    console.print("  [yellow]Triggering self-improvement rollback…[/]")
    result = await trigger_rollback()
    err = result.get("error")
    if err:
        console.print(f"  [red]rollback failed:[/] {escape(str(err))}\n")
    else:
        console.print(f"  [green]Rollback result:[/] {escape(json.dumps(result))}\n")


# ── Main REPL ─────────────────────────────────────────────────────────────────

HELP_TEXT = (
    "  [bold]Tasks[/]\n"
    "    <text>             — run a task (streaming output)\n"
    "\n"
    "  [bold]Status & Monitoring[/]\n"
    "    [bold]/status[/]          — NATS + backend + LLM health\n"
    "    [bold]/hw[/]              — hardware telemetry (VRAM/RAM/CPU/temp/t/s)\n"
    "    [bold]/actors[/]          — actor registry + NATS connection state\n"
    "    [bold]/tools[/]           — tool registry (base + MCP)\n"
    "    [bold]/caps[/]            — capability & improvement events\n"
    "    [bold]/sandbox[/]         — sandbox execution log feed\n"
    "    [bold]/debug[/]           — raw actor debug dump\n"
    "\n"
    "  [bold]Files[/]\n"
    "    [bold]/files [path][/]    — browse workspace directory\n"
    "    [bold]/read <path>[/]     — read a workspace file\n"
    "\n"
    "  [bold]Agent Control[/]\n"
    "    [bold]/rollback[/]        — revert last self-improvement merge\n"
    "\n"
    "  [bold]Display[/]\n"
    "    [bold]/think on|off[/]    — toggle live thinking panel\n"
    "    [bold]/raw on|off[/]      — toggle raw token stream\n"
    "    [bold]/mode[/]            — show current display settings\n"
    "    [bold]/api URL[/]         — switch backend endpoint\n"
    "    [bold]/clear[/]           — clear screen\n"
    "    [bold]/help[/]            — this help\n"
    "    [bold]exit[/]             — quit\n"
)


async def main_loop():
    """Main REPL loop — prompt_toolkit for input, rich for output."""
    global API_BASE
    console.clear()
    console.print(Panel(
        "[bold cyan]WzrdMerlin v2[/]  —  autonomous agent OS\n"
        "[dim]Type a task and press Enter. /help for commands. Ctrl+C or 'exit' to quit.[/]",
        border_style="cyan",
        padding=(0, 1),
    ))

    status = await check_health()
    console.print(f"  {status}\n")

    session = PromptSession()
    session_cfg = CliSession()

    # Start background SSE listener
    session_cfg.sse_task = asyncio.create_task(sse_listener(session_cfg))

    # Wait briefly for SSE to connect
    for _ in range(6):
        if session_cfg.sse_connected:
            console.print("  [green]Event stream connected.[/]\n")
            break
        await asyncio.sleep(0.5)
    else:
        console.print("  [yellow]Event stream connecting in background…[/]\n")

    try:
        while True:
            # Build prompt — show [human?] indicator when agent is waiting
            prompt_indicator = (
                HTML("<b><ansiyellow>[human?] </ansiyellow><ansigreen>&gt;</ansigreen></b> ")
                if session_cfg.waiting_for_human
                else HTML("<b><ansigreen>&gt;</ansigreen></b> ")
            )

            try:
                with patch_stdout():
                    text = await session.prompt_async(prompt_indicator)
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/]")
                break

            text = text.strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit", ":q"):
                console.print("[dim]Goodbye.[/]")
                break

            # ── commands ───────────────────────────────────────────────────

            if text.lower() == "/status":
                s = await check_health()
                console.print(f"  {s}\n")
                continue

            if text.lower() == "/clear":
                console.clear()
                continue

            if text.lower() == "/help":
                console.print(HELP_TEXT)
                continue

            if text.lower() == "/mode":
                sse_s = "[green]connected[/]" if session_cfg.sse_connected else "[red]disconnected[/]"
                console.print(
                    f"  thinking={'on' if session_cfg.show_thinking else 'off'}  "
                    f"raw_tokens={'on' if session_cfg.show_raw_tokens else 'off'}  "
                    f"sse={sse_s}  "
                    f"api={API_BASE}\n"
                )
                continue

            if text.lower() == "/hw":
                await cmd_hw(session_cfg)
                continue

            if text.lower() == "/actors":
                await cmd_actors(session_cfg)
                continue

            if text.lower() == "/tools":
                await cmd_tools(session_cfg)
                continue

            if text.lower() == "/caps":
                await cmd_caps(session_cfg)
                continue

            if text.lower() == "/sandbox":
                await cmd_sandbox(session_cfg)
                continue

            if text.lower() == "/debug":
                await cmd_debug()
                continue

            if text.lower() == "/rollback":
                await cmd_rollback()
                continue

            if text.lower().startswith("/files"):
                path_arg = text[6:].strip()
                await cmd_files(path_arg)
                continue

            if text.lower().startswith("/read ") or text.lower().startswith("/cat "):
                parts = text.split(" ", 1)
                path_arg = parts[1].strip() if len(parts) > 1 else ""
                await cmd_read(path_arg)
                continue

            if text.lower().startswith("/think "):
                val = text.split(" ", 1)[1].strip().lower()
                if val in ("on", "off"):
                    session_cfg.show_thinking = val == "on"
                    console.print(f"  thinking {'enabled' if session_cfg.show_thinking else 'disabled'}\n")
                else:
                    console.print("  usage: /think on|off\n")
                continue

            if text.lower().startswith("/raw "):
                val = text.split(" ", 1)[1].strip().lower()
                if val in ("on", "off"):
                    session_cfg.show_raw_tokens = val == "on"
                    console.print(f"  raw token stream {'enabled' if session_cfg.show_raw_tokens else 'disabled'}\n")
                else:
                    console.print("  usage: /raw on|off\n")
                continue

            if text.lower().startswith("/api "):
                val = text.split(" ", 1)[1].strip()
                if val.startswith("http://") or val.startswith("https://"):
                    API_BASE = val.rstrip("/")
                    console.print(f"  backend API set to {API_BASE}\n")
                    # Restart SSE listener with new API base
                    if session_cfg.sse_task:
                        session_cfg.sse_task.cancel()
                    session_cfg.sse_connected = False
                    session_cfg.sse_task = asyncio.create_task(sse_listener(session_cfg))
                else:
                    console.print("  usage: /api http://host:port\n")
                continue

            # Unknown slash command
            if text.startswith("/"):
                console.print(f"  [yellow]Unknown command.[/] Type /help for a list.\n")
                continue

            # ── run task ───────────────────────────────────────────────────
            await run_task(text, session_cfg)

    finally:
        if session_cfg.sse_task:
            session_cfg.sse_task.cancel()
            try:
                await session_cfg.sse_task
            except asyncio.CancelledError:
                pass


def main():
    import argparse
    p = argparse.ArgumentParser(description="WzrdMerlin CLI")
    p.add_argument("--api", default="http://localhost:8000", help="Backend URL")
    args = p.parse_args()

    global API_BASE
    API_BASE = args.api

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/]")


if __name__ == "__main__":
    main()
