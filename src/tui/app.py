"""
WzrdMerlin v2 — Claude Code-style CLI
Plain rich + httpx streaming. No Textual widgets, no event-bubbling issues.
"""
import asyncio
import json
import httpx
import textwrap
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Any
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.markup import escape
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML

API_BASE = "http://localhost:8000"
console = Console()


@dataclass
class CliSession:
    show_thinking: bool = True
    show_raw_tokens: bool = False


def _build_thinking_panel(thinking_text: str, max_lines: int = 8) -> Panel:
    wrapped: List[str] = []
    for raw_line in thinking_text.splitlines() or [thinking_text]:
        wrapped.extend(textwrap.wrap(raw_line, width=90) or [""])
    tail = wrapped[-max_lines:] if wrapped else ["Thinking…"]
    body = "\n".join(escape(line) for line in tail if line is not None)
    if not body.strip():
        body = "Thinking…"
    return Panel(
        body,
        title="Thinking",
        border_style="blue",
        padding=(0, 1),
    )


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


async def check_health() -> str:
    """Check backend + LLM health, return status string."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r1 = await client.get(f"{API_BASE}/api/health")
            r2 = await client.get(f"{API_BASE}/api/llm/health")
            h = r1.json()
            llm = r2.json()
            nats = h.get("nats", "?")
            model = llm.get("active_model", "?")
            loaded = llm.get("model_loaded", False)
            nats_s = f"[green]{nats}[/]" if nats == "connected" else f"[red]{nats}[/]"
            model_s = f"[green]{model}[/]" if loaded else f"[red]{model} (not loaded)[/]"
            return f"NATS {nats_s}  model {model_s}"
    except httpx.ConnectError:
        return "[red]backend unreachable — is docker-compose up?[/]"
    except Exception as e:
        return f"[yellow]status error: {e}[/]"


async def get_active_tasks() -> dict:
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{API_BASE}/api/tasks/active")
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}
            return resp.json()
    except httpx.ConnectError:
        return {"error": "backend unreachable"}
    except Exception as e:
        return {"error": str(e)}


def _print_narrative(pending_chunks: List[str]) -> None:
    if not pending_chunks:
        return
    narrative = "".join(pending_chunks).strip()
    pending_chunks.clear()
    if narrative:
        console.print(f"  [white]{escape(narrative)}[/]")


def _print_agent_line(text: str) -> None:
    clean = text.strip()
    if clean:
        console.print(f"  [bold magenta]Merlin[/] [white]{escape(clean)}[/]")


async def run_task(description: str, session_cfg: CliSession) -> None:
    """Stream a task through /api/run and render output Claude Code-style."""
    output_blocks: List[Any] = []

    context_window_tokens = int(os.getenv("MODEL_CONTEXT_WINDOW", "8196"))
    used_tokens_est = _estimate_tokens(description)
    last_context_rendered_used = -1
    context_update_step = 128

    thinking_panel_index: Optional[int] = None
    thinking_buffer = ""
    thinking_has_explicit_content = False

    def rebuild(live: Live, *, force_context: bool = False) -> None:
        nonlocal last_context_rendered_used
        if force_context or last_context_rendered_used < 0:
            last_context_rendered_used = used_tokens_est
        context_panel = _build_context_panel(context_window_tokens, last_context_rendered_used)
        live.update(Group(context_panel, *output_blocks), refresh=True)

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
                # Real <think> content — summarize and show
                summary = _summarize_thinking(thinking_buffer)
                output_blocks[thinking_panel_index] = Text(f"  Thought: {summary}", style="dim")
            else:
                # No real thinking content (think:false mode) — blank it out silently
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

    with Live(console=console, refresh_per_second=8, transient=False, screen=False, auto_refresh=False) as live:
        rebuild(live, force_context=True)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
            ) as client:
                async with client.stream(
                    "POST",
                    f"{API_BASE}/api/run",
                    params={"description": description},
                ) as resp:
                    if resp.status_code != 200:
                        push(Text(f"  HTTP {resp.status_code}", style="red"), live)
                        return

                    async for raw_line in resp.aiter_lines():
                        if not raw_line.strip():
                            continue
                        try:
                            msg = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue

                        t = msg.get("type", "")
                        text = msg.get("text", "")

                        if t == "thinking":
                            if "[step" in text:
                                finish_thinking_box(live)
                                if session_cfg.show_thinking:
                                    ensure_thinking_box(live)
                                else:
                                    push(Text("  Thinking…", style="dim"), live)
                            elif session_cfg.show_thinking and thinking_panel_index is not None:
                                thinking_has_explicit_content = True
                                thinking_buffer += text
                                used_tokens_est += _estimate_tokens(text)
                                output_blocks[thinking_panel_index] = _build_thinking_panel(thinking_buffer)
                                maybe_refresh_context(live)
                                rebuild(live, force_context=False)

                        elif t == "token":
                            # Tokens are raw LLM output (the JSON tool call being generated).
                            # Only show them if raw token mode is on. Never put them in the
                            # thinking panel — that causes the tool-call JSON to appear as
                            # "Thought: {"tool": ...}" which is misleading.
                            used_tokens_est += _estimate_tokens(text)
                            maybe_refresh_context(live)
                            if session_cfg.show_raw_tokens:
                                push(Text(f"    {text}", style="dim"), live)

                        elif t == "narrative":
                            finish_thinking_box(live)
                            clean = text.strip()
                            if clean:
                                push(Text(f"  Merlin {clean}", style="magenta"), live)

                        elif t == "tool_start":
                            finish_thinking_box(live)
                            tool = msg.get("tool", "?")
                            arg = str(msg.get("arg", ""))[:120]
                            if tool != "done":
                                push(Text(f"  Action {tool} {arg}", style="cyan"), live)

                        elif t == "tool_end":
                            finish_thinking_box(live)
                            tool = msg.get("tool", "")
                            if tool != "done":
                                result = str(msg.get("result", ""))[:800]
                                push(_build_observation_panel(result), live)

                        elif t == "done":
                            finish_thinking_box(live)
                            summary = text
                            push(Text(f"\n  Done. {summary}\n", style="green"), live)

                        elif t == "error":
                            finish_thinking_box(live)
                            push(Text(f"\n  Error: {text}\n", style="red"), live)

        except httpx.ConnectError:
            push(Text("  Cannot reach backend.", style="red"), live)
        except Exception as e:
            push(Text(f"  Stream error ({type(e).__name__}): {str(e)}", style="red"), live)


async def main_loop():
    """Main REPL loop — prompt_toolkit for input, rich for output."""
    global API_BASE
    with console.screen(style="default"):
        console.clear()
        console.print(Panel(
            "[bold cyan]WzrdMerlin v2[/]  —  autonomous agent OS\n"
            "[dim]Type a task and press Enter. Ctrl+C or 'exit' to quit.[/]",
            border_style="cyan",
            padding=(0, 1),
        ))

        status = await check_health()
        console.print(f"  {status}\n")

        session = PromptSession()
        session_cfg = CliSession()

        while True:
            try:
                with patch_stdout():
                    text = await session.prompt_async(
                        HTML("<b><ansigreen>&gt;</ansigreen></b> "),
                    )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/]")
                break

            text = text.strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit", ":q"):
                console.print("[dim]Goodbye.[/]")
                break

            if text.lower() == "/status":
                s = await check_health()
                console.print(f"  {s}\n")
                continue

            if text.lower() == "/clear":
                console.clear()
                continue

            if text.lower() == "/help":
                console.print(
                    "  [bold]/status[/]  — check backend + LLM health\n"
                    "  [bold]/tasks[/]   — list active tasks\n"
                    "  [bold]/think on|off[/] — toggle live thinking panel\n"
                    "  [bold]/raw on|off[/] — toggle raw token stream\n"
                    "  [bold]/api URL[/] — switch backend endpoint\n"
                    "  [bold]/mode[/]    — show current CLI display settings\n"
                    "  [bold]/clear[/]   — clear screen\n"
                    "  [bold]/help[/]    — show this help\n"
                    "  [bold]exit[/]     — quit\n"
                )
                continue

            if text.lower() == "/mode":
                console.print(
                    f"  thinking={'on' if session_cfg.show_thinking else 'off'}  "
                    f"raw_tokens={'on' if session_cfg.show_raw_tokens else 'off'}  "
                    f"api={API_BASE}\n"
                )
                continue

            if text.lower() == "/tasks":
                result = await get_active_tasks()
                err = result.get("error")
                if err:
                    console.print(f"  [red]active tasks unavailable:[/] {escape(str(err))}\n")
                    continue

                tasks = result.get("tasks", [])
                if not tasks:
                    console.print("  [dim]No active tasks.[/]\n")
                    continue

                lines = []
                for task in tasks:
                    task_id = str(task.get("task_id", "?"))
                    status = str(task.get("status", "running"))
                    iterations = str(task.get("iterations", 0))
                    desc = str(task.get("description", ""))
                    short_desc = desc if len(desc) <= 90 else desc[:87] + "..."
                    lines.append(f"• {task_id}  [{status}]  iter={iterations}  {short_desc}")

                console.print(Panel("\n".join(lines), title="Active Tasks", border_style="cyan", padding=(0, 1)))
                console.print()
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
                else:
                    console.print("  usage: /api http://host:port\n")
                continue

            # prompt_toolkit already displayed the input — don't echo it again
            await run_task(text, session_cfg)


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
