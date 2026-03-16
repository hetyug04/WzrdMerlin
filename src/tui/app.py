"""
WzrdMerlin v3 — CLI / TUI
Rich + prompt_toolkit. SSE streaming. Thinking-agent native.
"""
import asyncio
import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

API_BASE = "http://localhost:8000"
console = Console()

# ── Design tokens ────────────────────────────────────────────────────────────
# Purple/cyan/gold palette — magical + hacker-elegant
C_PRIMARY = "medium_purple1"      # main accent (purple)
C_ACCENT = "sky_blue1"            # secondary accent (cyan)
C_GOLD = "dark_goldenrod"         # warnings, hw telemetry
C_OK = "green3"                   # success states
C_ERR = "red1"                    # errors
C_DIM = "bright_black"            # metadata, secondary text
C_THINK = "medium_purple1"        # thinking border
C_TOOL = "sky_blue1"              # tool calls
C_ANSWER = "green3"               # answer border

GLYPH_MERLIN = "✦"
GLYPH_STEP = "◈"
GLYPH_TOOL = "›"
GLYPH_ARROW = "→"
GLYPH_DOT = "·"

SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# ── Session state ────────────────────────────────────────────────────────────

@dataclass
class Session:
    show_thinking: bool = True
    show_raw: bool = False
    sse_connected: bool = False
    sse_task: Optional[asyncio.Task] = None
    waiting_for_human: bool = False
    waiting_task_id: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=lambda: {
        "vram": 0.0, "vram_total": 12.0, "ram": 0.0, "cpu": 0.0,
        "tokens_per_sec": 0.0, "latency_ms": 0.0, "temperature": 0,
    })
    queues: Dict[str, asyncio.Queue] = field(default_factory=dict)
    cap_events: List[dict] = field(default_factory=list)
    _buf: List[dict] = field(default_factory=list)
    _buf_max: int = 200


# ── API helpers ──────────────────────────────────────────────────────────────

async def _get(path: str, timeout: float = 5.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{API_BASE}{path}")
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}


async def _post(path: str, timeout: float = 10.0, **params) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{API_BASE}{path}", params=params)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}


async def check_health() -> str:
    try:
        h = await _get("/api/health")
        nats = h.get("nats", "?")
        nats_s = f"[{C_OK}]✓ connected[/]" if nats == "connected" else f"[{C_ERR}]✗ {nats}[/]"
        try:
            llm = await _get("/api/llm/health", 8.0)
            loaded = llm.get("model_loaded", False)
            model = llm.get("active_model", "?").split(":")[-1][:20]  # shorten model name
            cb = llm.get("circuit_breaker_open", False)
            if cb:
                llm_s = f"  oracle [{C_ERR}]breaker open (rest, dear Merlin)[/]"
            elif loaded:
                llm_s = f"  oracle [{C_OK}]{model} ready[/]"
            else:
                llm_s = f"  oracle [{C_GOLD}]{model} loading…[/]"
        except Exception:
            llm_s = f"  oracle [{C_DIM}]consulting the void[/]"
        return f"the channels {nats_s}{llm_s}"
    except httpx.ConnectError:
        return f"[{C_ERR}]the citadel is silent — is docker running?[/]"
    except Exception as e:
        return f"[{C_GOLD}]the threads are tangled: {e}[/]"


# ── Visual components ────────────────────────────────────────────────────────

def _thinking_panel(text: str, max_lines: int = 12) -> Panel:
    lines: List[str] = []
    for raw in (text or "").splitlines():
        lines.extend(textwrap.wrap(raw, width=88) or [""])
    body = "\n".join(escape(l) for l in lines[-max_lines:])
    # Witty thinking states
    thinking_states = [
        "consulting the threads of fate…",
        "muttering incantations…",
        "peering into the void…",
        "the gears turn…",
        "ancient wisdom stirs…",
        "plotting a course most cunning…",
    ]
    import hashlib
    state_idx = int(hashlib.md5(body.encode()).hexdigest(), 16) % len(thinking_states)
    state = thinking_states[state_idx] if body else "gathering thoughts…"
    return Panel(
        body or f"[{C_DIM}]{state}[/]",
        title=f"[{C_THINK}]{GLYPH_MERLIN} inner sanctum[/]",
        border_style=C_THINK,
        padding=(0, 1),
        style="on default",
    )


def _observation_panel(text: str, max_lines: int = 6) -> Panel:
    lines = (text or "(empty)").splitlines()
    body = "\n".join(escape(l) for l in lines[:max_lines])
    if len(lines) > max_lines:
        body += f"\n[{C_DIM}]  {GLYPH_DOT}{GLYPH_DOT}{GLYPH_DOT} {len(lines) - max_lines} more lines[/]"
    return Panel(body, border_style=C_DIM, padding=(0, 1))


_PERSONALITY_ICON = {
    "writer":         (GLYPH_MERLIN, C_PRIMARY),
    "coder":          (GLYPH_STEP,   C_ACCENT),
    "researcher":     (GLYPH_TOOL,   C_GOLD),
    "installer":      (GLYPH_ARROW,  C_OK),
    "web_researcher": (GLYPH_DOT,    C_ACCENT),
}
_STATUS_ICON = {
    "running":   (SPIN[0],  C_PRIMARY),
    "completed": ("+",      C_OK),
    "failed":    ("x",      C_ERR),
    "timeout":   ("~",      C_GOLD),
}


def _worker_status_panel(workers: dict) -> Panel:
    """Persistent panel showing every spawned sub-agent for the current task."""
    lines = []
    for wid, w in workers.items():
        personality = w.get("personality", "?")
        status = w.get("status", "running")
        desc = escape(w.get("description", "")[:72])
        p_glyph, p_color = _PERSONALITY_ICON.get(personality, ("?", C_DIM))
        s_glyph, s_color = _STATUS_ICON.get(status, ("◉", C_DIM))
        lines.append(
            f"  [{p_color}]{p_glyph} {personality:<14}[/]"
            f" [{s_color}]{s_glyph} {status:<9}[/]"
            f" [{C_DIM}]{desc}[/]"
        )
    body = "\n".join(lines) if lines else f"[{C_DIM}](none)[/]"
    return Panel(
        body,
        title=f"[{C_ACCENT}]{GLYPH_TOOL} sub-agents[/]",
        border_style=C_ACCENT,
        padding=(0, 1),
    )


def _context_bar(capacity: int, used: int) -> Text:
    capacity = max(capacity, 1)
    used = max(0, min(used, capacity))
    pct = used / capacity
    w = 30
    filled = int(w * pct)
    # Color shifts as context fills: cyan → gold → red
    if pct < 0.6:
        bar_color = C_ACCENT
    elif pct < 0.85:
        bar_color = C_GOLD
    else:
        bar_color = C_ERR
    bar = "━" * filled + "╌" * (w - filled)
    left = capacity - used
    t = Text()
    t.append(f"  ctx ", style=C_DIM)
    t.append(bar, style=bar_color)
    t.append(f" {used}/{capacity}", style=C_DIM)
    t.append(f"  {left} left", style=C_DIM)
    return t


# ── Background SSE ───────────────────────────────────────────────────────────

async def _sse_loop(s: Session) -> None:
    while True:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10, read=None, write=None, pool=None)
            ) as client:
                async with client.stream("GET", f"{API_BASE}/api/stream") as resp:
                    s.sse_connected = True
                    async for raw in resp.aiter_lines():
                        if not raw.startswith("data: "):
                            continue
                        data = raw[6:].strip()
                        if not data or data == "keep-alive":
                            continue
                        try:
                            evt = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        s._buf.append(evt)
                        if len(s._buf) > s._buf_max:
                            s._buf = s._buf[-s._buf_max:]

                        t = evt.get("type", "")
                        p = evt.get("payload", {})
                        cid = evt.get("correlation_id", "")

                        if t == "system.heartbeat":
                            for k in ("vram", "vram_total", "ram_usage", "cpu_usage",
                                       "tokens_per_sec", "latency_ms", "temperature"):
                                if k in p:
                                    s.telemetry[k.replace("_usage", "").replace("ram_", "ram")] = p[k]

                        if t in ("capability.gap", "improvement.queued", "improvement.deployed"):
                            s.cap_events = ([evt] + s.cap_events)[:20]

                        if cid and cid in s.queues:
                            await s.queues[cid].put(evt)

        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
            s.sse_connected = False
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            return
        except Exception:
            s.sse_connected = False
            await asyncio.sleep(5)


# ── Task renderer ────────────────────────────────────────────────────────────

async def _render(task_id: str, q: asyncio.Queue, s: Session) -> None:
    blocks: List[Any] = []
    ctx_cap = int(os.getenv("MODEL_CONTEXT_WINDOW", "8196"))
    ctx_used = 0
    think_idx: Optional[int] = None
    think_buf = ""
    think_real = False
    step = 0
    idle_timeout = float(os.getenv("MERLIN_TUI_IDLE_TIMEOUT", "600"))
    idle = 0.0
    spin_i = 0
    _workers: dict = {}          # worker_id → {personality, status, description}
    _workers_block_idx: list = [None]  # list wrapper so nested funcs can mutate it

    def _refresh(live: Live):
        live.update(Group(_context_bar(ctx_cap, ctx_used), *blocks), refresh=True)

    def _push(r: Any, live: Live):
        blocks.append(r)
        _refresh(live)

    def _close_think(live: Live):
        nonlocal think_idx, think_buf, think_real
        if think_idx is not None:
            if think_real and think_buf.strip():
                if s.show_thinking:
                    blocks[think_idx] = _thinking_panel(think_buf, max_lines=50)
                else:
                    short = " ".join(think_buf.split())[:120].rstrip(" ,.;:-")
                    if len(think_buf) > 120:
                        short += "…"
                    blocks[think_idx] = Text(f"  [{C_DIM}]thought: {short}[/]")
                    blocks[think_idx] = Text.from_markup(f"  [{C_DIM}]thought: {escape(short)}[/]")
            else:
                blocks[think_idx] = Text("")
            _refresh(live)
            think_idx = None
            think_buf = ""
            think_real = False

    def _open_think(live: Live):
        nonlocal think_idx, think_buf
        if think_idx is None and s.show_thinking:
            think_buf = ""
            blocks.append(_thinking_panel(""))
            think_idx = len(blocks) - 1
            _refresh(live)

    with Live(console=console, refresh_per_second=8, transient=False,
              screen=False, auto_refresh=False) as live:
        _refresh(live)

        while True:
            try:
                evt = await asyncio.wait_for(q.get(), timeout=0.5)
                idle = 0.0
            except asyncio.TimeoutError:
                idle += 0.5
                if idle >= idle_timeout:
                    _push(Text.from_markup(f"  [{C_GOLD}]timed out waiting for events.[/]"), live)
                    break
                # Breathing spinner during generation
                if idle >= 1.0 and think_idx is not None:
                    elapsed = int(idle)
                    frame = SPIN[spin_i % len(SPIN)]
                    spin_i += 1
                    blocks[think_idx] = Panel(
                        f"  [{C_PRIMARY}]{frame}[/]  [{C_DIM}]generating… {elapsed}s[/]",
                        title=f"[{C_THINK}]{GLYPH_MERLIN} thinking[/]",
                        border_style=C_THINK,
                        padding=(0, 1),
                    )
                    _refresh(live)
                continue

            t = evt.get("type", "")
            p = evt.get("payload", {})

            # ── Thinking ─────────────────────────────────────────────
            if t == "agent.thinking":
                text = p.get("text", "")
                if "[iteration" in text or "[step" in text:
                    step += 1
                    _close_think(live)
                    step_words = ["the first stroke", "onwards", "deeper still", "the path narrows", "wisdom grows", "the end nears"]
                    step_word = step_words[min(step - 1, len(step_words) - 1)]
                    marker = Text.from_markup(
                        f"\n  [{C_PRIMARY}]{GLYPH_STEP}[/] [{C_DIM}]step {step} — {step_word}[/]"
                    )
                    _push(marker, live)
                    _open_think(live)
                elif s.show_thinking:
                    _open_think(live)
                    think_real = True
                    think_buf += text
                    ctx_used += max(1, len(text) // 4)
                    blocks[think_idx] = _thinking_panel(think_buf)
                    _refresh(live)

            # ── Streaming tokens ─────────────────────────────────────
            elif t == "agent.streaming":
                text = p.get("text", "")
                ctx_used += max(1, len(text) // 4)
                idle = 0.0
                if s.show_raw and text.strip():
                    _push(Text(f"    {text}", style=C_DIM), live)

            # ── Tool progress (in-place update) ─────────────────────
            elif t == "agent.tool_progress":
                line = p.get("text", "").rstrip()
                if not line:
                    pass
                elif blocks and isinstance(blocks[-1], Panel) and "running" in str(getattr(blocks[-1], "title", "")).lower():
                    prev = str(getattr(blocks[-1], "renderable", "")).splitlines()[-8:]
                    prev.append(escape(line))
                    blocks[-1] = Panel(
                        "\n".join(prev[-8:]),
                        title=f"[{C_DIM}]running…[/]",
                        border_style=C_DIM,
                        padding=(0, 1),
                    )
                    _refresh(live)
                else:
                    _push(Panel(
                        escape(line),
                        title=f"[{C_DIM}]running…[/]",
                        border_style=C_DIM,
                        padding=(0, 1),
                    ), live)

            # ── Tool start ───────────────────────────────────────────
            elif t == "agent.tool_start":
                _close_think(live)
                tool = p.get("tool", "?")
                args = p.get("args", {})

                if tool == "request_human":
                    question = args.get("question") or p.get("question") or p.get("input", "")
                    if question:
                        _push(Panel(
                            escape(str(question)),
                            title=f"[bold {C_GOLD}]{GLYPH_MERLIN} merlin asks[/]",
                            border_style=C_GOLD,
                            padding=(0, 1),
                        ), live)
                    s.waiting_for_human = True
                    s.waiting_task_id = task_id
                    _push(Text.from_markup(f"  [{C_GOLD}]{GLYPH_ARROW} respond below[/]"), live)
                    break

                if tool != "done":
                    ln = Text()
                    ln.append(f"  {GLYPH_TOOL} ", style=C_TOOL)
                    ln.append(tool, style=f"bold {C_TOOL}")
                    preview = str(next(iter(args.values()), ""))[:80] if args else ""
                    if preview:
                        ln.append(f"  {escape(preview)}", style=C_DIM)
                    _push(ln, live)

            # ── Tool end ─────────────────────────────────────────────
            elif t == "agent.tool_end":
                _close_think(live)
                if p.get("tool") != "done":
                    _push(_observation_panel(str(p.get("result", ""))), live)

            # ── Task completed ───────────────────────────────────────
            elif t == "task.completed":
                _close_think(live)
                s.waiting_for_human = False
                s.waiting_task_id = None
                result = p.get("result", "")
                if result:
                    _push(Panel(
                        escape(str(result)),
                        title=f"[bold {C_ANSWER}]{GLYPH_MERLIN} the quest concludes[/]",
                        border_style=C_ANSWER,
                        padding=(0, 1),
                    ), live)
                else:
                    _push(Text.from_markup(f"\n  [{C_OK}]{GLYPH_MERLIN} alas, silence is the answer.[/]\n"), live)
                break

            # ── Failed / Cancelled ───────────────────────────────────
            elif t in ("task.failed", "action.failed"):
                _close_think(live)
                s.waiting_for_human = False
                reason = escape(str(p.get("reason", "unknown")))
                _push(Text.from_markup(f"\n  [{C_ERR}]{GLYPH_MERLIN} the spell hath broken: {reason}[/]\n"), live)
                break

            elif t == "task.routed":
                actor = escape(p.get("target_actor", "?"))
                _push(Text.from_markup(f"  [{C_DIM}]routed {GLYPH_ARROW} {actor}[/]"), live)

            elif t == "action.completed":
                pass

            # ── Sub-agent spawned ────────────────────────────────────
            elif t == "agent.spawned":
                worker_id = p.get("worker_id", "")
                personality = p.get("personality", "?")
                desc = p.get("description", "")[:80]
                _workers[worker_id] = {
                    "personality": personality,
                    "status": "running",
                    "description": desc,
                }
                p_glyph, p_color = _PERSONALITY_ICON.get(personality, ("?", C_DIM))
                _push(Text.from_markup(
                    f"  [{C_ACCENT}]{GLYPH_TOOL}[/] spawned "
                    f"[{p_color}]{p_glyph} {escape(personality)}[/] sub-agent"
                    f"  [{C_DIM}]{escape(desc)}[/]"
                ), live)
                if _workers_block_idx[0] is None:
                    blocks.append(_worker_status_panel(_workers))
                    _workers_block_idx[0] = len(blocks) - 1
                else:
                    blocks[_workers_block_idx[0]] = _worker_status_panel(_workers)
                _refresh(live)

            # ── Sub-agent completed ──────────────────────────────────
            elif t == "worker.completed":
                worker_id = p.get("worker_id", "")
                personality = p.get("personality", "?")
                status = p.get("status", "completed")
                if worker_id in _workers:
                    _workers[worker_id]["status"] = status
                    if _workers_block_idx[0] is not None:
                        blocks[_workers_block_idx[0]] = _worker_status_panel(_workers)
                s_glyph, s_color = _STATUS_ICON.get(status, ("?", C_DIM))
                _push(Text.from_markup(
                    f"  [{s_color}]{s_glyph}[/] sub-agent "
                    f"[{C_DIM}]{escape(personality)}[/] {status}"
                ), live)

            # ── System info (context fold, etc.) ─────────────────────
            elif t == "system.info":
                text = p.get("text", "")
                if text:
                    _push(Text.from_markup(
                        f"  [{C_DIM}]{GLYPH_DOT} {escape(text)}[/]"
                    ), live)


async def run_task(desc: str, s: Session) -> None:
    if not s.sse_connected:
        console.print(f"  [{C_DIM}]listening for whispers from the aether…[/]")
        for _ in range(10):
            if s.sse_connected:
                break
            await asyncio.sleep(0.5)
        if not s.sse_connected:
            console.print(f"  [{C_GOLD}]⚠ the voices are silent. is the tower still standing?[/]")

    try:
        data = await _post("/api/task", description=desc)
    except httpx.ConnectError:
        console.print(f"  [{C_ERR}]cannot reach backend.[/]")
        return

    if "error" in data:
        console.print(f"  [{C_ERR}]{data['error']}[/]")
        return

    task_id = data.get("task_id", "")
    if data.get("status") == "responded":
        console.print(f"  [{C_DIM}]response sent to {task_id}[/]\n")
    elif not task_id:
        console.print(f"  [{C_ERR}]no task_id returned: {data}[/]")
        return

    q: asyncio.Queue = asyncio.Queue()
    s.queues[task_id] = q
    for evt in s._buf:
        if evt.get("correlation_id") == task_id:
            await q.put(evt)
    try:
        await _render(task_id, q, s)
    finally:
        s.queues.pop(task_id, None)


# ── Commands ─────────────────────────────────────────────────────────────────

def _fmt_size(n: Optional[int]) -> str:
    if n is None:
        return ""
    if n < 1024:
        return f"{n}B"
    if n < 1048576:
        return f"{n / 1024:.1f}K"
    return f"{n / 1048576:.1f}M"


def _bar(value: float, width: int = 24, color: str = C_ACCENT) -> str:
    """Render a horizontal bar for telemetry."""
    filled = int(width * max(0, min(value, 100)) / 100)
    return f"[{color}]{'━' * filled}[/][{C_DIM}]{'╌' * (width - filled)}[/]"


COMMANDS: Dict[str, str] = {
    "/status":    "consult the channels (nats + oracle)",
    "/hw":        "survey the hardware",
    "/actors":    "census of the servants",
    "/tools":     "grimoire of incantations",
    "/caps":      "chronicle of gaps and mergers",
    "/logs [N]":  "scroll through the ether",
    "/files [p]": "browse the vault",
    "/read <p>":  "examine a tome",
    "/rollback":  "rewind the spell",
    "/think on|off": "show/hide the inner voice",
    "/raw on|off":   "toggle raw syllables",
    "/clear":     "wipe the slate",
    "/help":      "reveal the incantations",
}


async def cmd_hw(s: Session):
    try:
        llm = await _get("/api/llm/health", 8.0)
        if llm.get("tokens_per_sec"):
            s.telemetry["tokens_per_sec"] = llm["tokens_per_sec"]
        if llm.get("active_model"):
            s.telemetry["active_model"] = llm["active_model"]
    except Exception:
        pass
    t = s.telemetry
    vp = (t["vram"] / t["vram_total"] * 100) if t["vram_total"] > 0 else 0
    rp = t.get("ram", 0)
    cp = t.get("cpu", 0)
    body = (
        f"  vram  {_bar(vp, 24, C_GOLD)}  {t['vram']:.1f}/{t['vram_total']}G  ({vp:.0f}%)\n"
        f"  ram   {_bar(rp)}  {rp:.0f}%\n"
        f"  cpu   {_bar(cp)}  {cp:.1f}%\n"
        f"  speed [{C_OK}]{t.get('tokens_per_sec', 0):.1f}[/] t/s   "
        f"latency [{C_OK}]{t.get('latency_ms', 0):.0f}[/] ms"
    )
    m = t.get("active_model", "")
    if m:
        body += f"\n  model [bold {C_PRIMARY}]{escape(m)}[/]"
    console.print(Panel(
        body,
        title=f"[{C_GOLD}]hardware[/]",
        border_style=C_GOLD,
        padding=(0, 0),
    ))


async def cmd_actors():
    data = await _get("/api/debug/actors", 8.0)
    if "error" in data:
        console.print(f"  [{C_ERR}]{data['error']}[/]")
        return
    tbl = Table(border_style=C_DIM, padding=(0, 1), show_header=True, header_style=f"bold {C_ACCENT}")
    tbl.add_column("actor")
    tbl.add_column("nats", justify="center")
    tbl.add_column("handlers", style=C_DIM)
    tbl.add_column("mcp")
    for name, info in data.items():
        if not isinstance(info, dict):
            continue
        c = f"[{C_OK}]on[/]" if info.get("connected") else f"[{C_ERR}]off[/]"
        h = ", ".join(info.get("handlers", []))
        m = f"{len(info.get('mcp_tools', []))} tools"
        tbl.add_row(name, c, h or "-", m)
    console.print(tbl)
    console.print(f"  [{C_DIM}]ui listeners: {data.get('ui_listeners', 0)}[/]")


async def cmd_tools():
    data = await _get("/api/debug/actors", 8.0)
    if "error" in data:
        console.print(f"  [{C_ERR}]{data['error']}[/]")
        return
    base = ["shell", "read_file", "write_file", "search_memory",
            "write_memory", "fetch_url", "done", "request_human"]
    mcp_tools = []
    for info in data.values():
        if isinstance(info, dict):
            for t in info.get("mcp_tools", []):
                if t not in mcp_tools:
                    mcp_tools.append(t)
    tbl = Table(border_style=C_DIM, padding=(0, 1), show_header=True, header_style=f"bold {C_ACCENT}")
    tbl.add_column("tool")
    tbl.add_column("type")
    for t in base:
        tbl.add_row(t, f"[{C_ACCENT}]base[/]")
    for t in mcp_tools:
        tbl.add_row(t, f"[{C_GOLD}]mcp[/]")
    console.print(tbl)


async def cmd_caps(s: Session):
    if not s.cap_events:
        console.print(f"  [{C_DIM}]no capability events yet.[/]")
        return
    tbl = Table(border_style=C_DIM, padding=(0, 1), show_header=True, header_style=f"bold {C_ACCENT}")
    tbl.add_column("type")
    tbl.add_column("detail")
    tbl.add_column("task", style=C_DIM)
    colors = {"capability.gap": C_ERR, "improvement.queued": C_PRIMARY, "improvement.deployed": C_OK}
    for e in s.cap_events[:15]:
        et = e.get("type", "?")
        p = e.get("payload", {})
        detail = p.get("gap_description") or p.get("tool_name") or json.dumps(p)[:60]
        task = (p.get("triggering_task") or p.get("task_id", "-"))[:20]
        c = colors.get(et, "")
        tbl.add_row(f"[{c}]{et}[/]" if c else et, escape(detail), escape(task))
    console.print(tbl)


async def cmd_logs(n: int = 50):
    try:
        data = await _get(f"/api/logs?lines={min(n, 200)}", 5.0)
        lines = data.get("lines", [])
        total = data.get("total_buffered", 0)
        if not lines:
            console.print(f"  [{C_DIM}]no logs available.[/]")
            return
        console.print(f"  [{C_DIM}]{len(lines)} of {total} buffered[/]")
        for line in lines:
            console.print(f"  {escape(line)}")
    except httpx.ConnectError:
        console.print(f"  [{C_ERR}]backend unreachable.[/]")


async def cmd_files(path: str):
    try:
        data = await _get(f"/api/files?path={path}", 8.0)
    except httpx.ConnectError:
        console.print(f"  [{C_ERR}]backend unreachable.[/]")
        return
    if "error" in data:
        console.print(f"  [{C_ERR}]{data['error']}[/]")
        return
    entries = data.get("entries", [])
    dp = data.get("path", path or "/workspace")
    if not entries:
        console.print(f"  [{C_DIM}]{dp} is empty.[/]")
        return
    tbl = Table(
        title=f"[{C_ACCENT}]{escape(dp)}[/]",
        border_style=C_DIM,
        padding=(0, 1),
        show_header=True,
        header_style=f"bold {C_ACCENT}",
    )
    tbl.add_column("name")
    tbl.add_column("type", justify="center")
    tbl.add_column("size", justify="right", style=C_DIM)
    for e in entries:
        n = e.get("name", "?")
        is_dir = e.get("type") == "directory"
        tbl.add_row(
            f"[{C_GOLD}]{escape(n)}/[/]" if is_dir else escape(n),
            f"[{C_GOLD}]d[/]" if is_dir else f"[{C_ACCENT}]f[/]",
            _fmt_size(e.get("size")),
        )
    console.print(tbl)


async def cmd_read(path: str):
    if not path:
        console.print("  usage: /read <path>")
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{API_BASE}/api/files/read", params={"path": path})
            content = r.text if r.status_code == 200 else f"Error {r.status_code}: {r.text}"
    except httpx.ConnectError:
        content = "Error: backend unreachable"
    lines = content.splitlines()
    if len(lines) > 100:
        preview = "\n".join(lines[:100]) + f"\n[{C_DIM}]{GLYPH_DOT}{GLYPH_DOT}{GLYPH_DOT} {len(lines) - 100} more lines[/]"
        console.print(Panel(preview, title=f"[{C_ACCENT}]{escape(path)}[/]",
                            border_style=C_DIM, padding=(0, 1)))
    else:
        console.print(Panel(escape(content), title=f"[{C_ACCENT}]{escape(path)}[/]",
                            border_style=C_DIM, padding=(0, 1)))


# ── REPL ─────────────────────────────────────────────────────────────────────

BANNER = f"""[bold {C_PRIMARY}]
  {GLYPH_MERLIN} WzrdMerlin v3[/]  [{C_DIM}]a wizard most wise (and occasionally mischievous)[/]
  [{C_DIM}]speak thy will. /help to glimpse the grimoire.[/]
"""

HELP_TEXT = "\n".join(
    f"  [{C_ACCENT}]{k:<16}[/] [{C_DIM}]{v}[/]" for k, v in COMMANDS.items()
)


async def main_loop():
    global API_BASE
    console.clear()
    console.print(BANNER)
    console.print(f"  {await check_health()}\n")

    prompt = PromptSession()
    s = Session()
    s.sse_task = asyncio.create_task(_sse_loop(s))

    for _ in range(6):
        if s.sse_connected:
            console.print(f"  [{C_OK}]the channels sing. the tower is alive.[/]\n")
            break
        await asyncio.sleep(0.5)
    else:
        console.print(f"  [{C_GOLD}]listening for the whispers…[/]\n")

    try:
        while True:
            if s.waiting_for_human:
                indicator = HTML(
                    "<b><ansiyellow>✦ human? </ansiyellow><ansimagenta>›</ansimagenta></b> "
                )
            else:
                indicator = HTML(
                    "<b><ansimagenta>merlin›</ansimagenta></b> "
                )
            try:
                with patch_stdout():
                    text = await prompt.prompt_async(indicator)
            except (EOFError, KeyboardInterrupt):
                break

            text = text.strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit", ":q"):
                break

            cmd = text.lower()

            if cmd == "/help":
                console.print(HELP_TEXT)
            elif cmd == "/status":
                console.print(f"  {await check_health()}")
            elif cmd == "/clear":
                console.clear()
                console.print(BANNER)
            elif cmd == "/hw":
                await cmd_hw(s)
            elif cmd == "/actors":
                await cmd_actors()
            elif cmd == "/tools":
                await cmd_tools()
            elif cmd == "/caps":
                await cmd_caps(s)
            elif cmd == "/rollback":
                console.print(f"  [{C_GOLD}]triggering rollback…[/]")
                r = await _post("/api/rollback", timeout=15.0)
                console.print(f"  {escape(json.dumps(r))}")
            elif cmd.startswith("/logs"):
                n = 50
                parts = text.split()
                if len(parts) > 1 and parts[1].isdigit():
                    n = int(parts[1])
                await cmd_logs(n)
            elif cmd.startswith("/files"):
                await cmd_files(text[6:].strip())
            elif cmd.startswith("/read ") or cmd.startswith("/cat "):
                await cmd_read(text.split(" ", 1)[1].strip() if " " in text else "")
            elif cmd.startswith("/think "):
                val = text.split()[1].lower()
                if val in ("on", "off"):
                    s.show_thinking = val == "on"
                    console.print(f"  [{C_DIM}]thinking {'on' if s.show_thinking else 'off'}[/]")
                else:
                    console.print("  usage: /think on|off")
            elif cmd.startswith("/raw "):
                val = text.split()[1].lower()
                if val in ("on", "off"):
                    s.show_raw = val == "on"
                    console.print(f"  [{C_DIM}]raw tokens {'on' if s.show_raw else 'off'}[/]")
                else:
                    console.print("  usage: /raw on|off")
            elif cmd.startswith("/api "):
                val = text.split(" ", 1)[1].strip()
                if val.startswith("http"):
                    API_BASE = val.rstrip("/")
                    console.print(f"  [{C_DIM}]api {GLYPH_ARROW} {API_BASE}[/]")
                    if s.sse_task:
                        s.sse_task.cancel()
                    s.sse_connected = False
                    s.sse_task = asyncio.create_task(_sse_loop(s))
                else:
                    console.print("  usage: /api http://host:port")
            elif text.startswith("/"):
                console.print(f"  [{C_GOLD}]unknown command.[/] /help for list.")
            else:
                await run_task(text, s)

            console.print()

    finally:
        if s.sse_task:
            s.sse_task.cancel()
            try:
                await s.sse_task
            except asyncio.CancelledError:
                pass
        farewell_msgs = [
            "the tapestry is rewoven. farewell, seeker.",
            "the spell dissolves. until we meet again.",
            "wisdom whispers: return when the need stirs.",
            "the tower rests. go forth, and may your paths be clear.",
        ]
        import random
        msg = random.choice(farewell_msgs)
        console.print(f"[{C_DIM}]{GLYPH_MERLIN} {msg}[/]")


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
        console.print(f"\n[{C_DIM}]{GLYPH_MERLIN} farewell.[/]")


if __name__ == "__main__":
    main()
