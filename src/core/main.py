import logging
import logging.config

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"},
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    # Silence noisy third-party loggers
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "litellm": {"level": "WARNING"},
        "nats": {"level": "WARNING"},
    },
})

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
import os
import re
import time
import httpx
from src.core.events import Event, EventType
from src.core.router import DisCoRouter
from src.core.base_agent import BaseAgentActor
from src.core.state import StateStore
from src.core.self_improve import ImprovementManager

logger = logging.getLogger(__name__)

app = FastAPI(title="WzrdMerlin v2 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
router_actor = DisCoRouter()
base_agent = BaseAgentActor()
improvement_mgr = ImprovementManager()
state_store = StateStore()

# In-memory queues for UI SSE streaming
ui_queues = []

# In-memory active run registry for /api/run tasks
active_runs = {}

async def broadcast_to_ui(event: Event):
    for q in ui_queues:
        await q.put(event)

@app.on_event("startup")
async def startup_event():
    # Inject UI broadcast callback so the agent can push live narration directly
    # without per-token NATS round-trips
    base_agent._ui_broadcast = broadcast_to_ui

    # Connect NATS actors
    await router_actor.connect()
    await base_agent.connect()
    await improvement_mgr.connect()
    await state_store.connect()

    # Sniff all events via core NATS wildcard — no JetStream consumer needed.
    # Core NATS ">" matches every subject on the connection.
    async def _sniff_events(msg):
        try:
            data = json.loads(msg.data.decode())
            event = Event(**data)
            await broadcast_to_ui(event)
        except Exception:
            pass

    await router_actor.nc.subscribe("events.>", cb=_sniff_events)

@app.on_event("shutdown")
async def shutdown_event():
    await router_actor.close()
    await base_agent.close()
    await improvement_mgr.close()
    await state_store.close()

@app.post("/api/task")
async def create_task(description: str):
    task_id = f"task_{int(asyncio.get_event_loop().time())}"
    evt = Event(
        type=EventType.TASK_CREATED,
        source_actor="api",
        correlation_id=task_id,
        payload={"task_id": task_id, "description": description}
    )
    await router_actor.publish("events.task.created", evt)
    return {"status": "accepted", "task_id": task_id}

@app.get("/api/health")
async def health_check():
    nats_ok = router_actor.nc is not None and router_actor.nc.is_connected
    return {
        "status": "ok" if nats_ok else "degraded",
        "nats": "connected" if nats_ok else "disconnected",
    }


@app.get("/api/llm/health")
async def llm_health():
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    active_model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_base}/api/tags")
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            model_loaded = active_model in models
            return {
                "status": "ok" if model_loaded else "model_missing",
                "ollama": "connected",
                "active_model": active_model,
                "model_loaded": model_loaded,
                "available_models": models,
            }
    except httpx.ConnectError:
        return {"status": "error", "ollama": "unreachable", "active_model": active_model}
    except Exception as e:
        return {"status": "error", "ollama": str(e), "active_model": active_model}


@app.post("/api/run")
async def run_task(
    description: str,
    max_iterations: int | None = None,
    max_seconds: int | None = None,
):
    """
    Runs a task and streams narration back as newline-delimited JSON.
    Each line is a JSON object: {"type": "...", "text": "...", "tool": "..."}
    Types: thinking | narrative | token | tool_start | tool_end | done | error
    The TUI reads this directly — no SSE bus, no race conditions.
    """
    from src.core.llm import ModelInterface
    import uuid

    task_id = f"task_{uuid.uuid4().hex[:8]}"
    llm = ModelInterface()

    active_runs[task_id] = {
        "task_id": task_id,
        "description": description,
        "status": "running",
        "started_at": int(time.time()),
        "updated_at": int(time.time()),
        "iterations": 0,
    }

    env_max_iterations = int(os.getenv("RUN_MAX_ITERATIONS", "0"))
    env_max_seconds = int(os.getenv("RUN_MAX_SECONDS", "900"))
    stall_repeat_limit = int(os.getenv("RUN_STALL_REPEAT_LIMIT", "4"))

    effective_max_iterations = env_max_iterations if max_iterations is None else max_iterations
    effective_max_seconds = env_max_seconds if max_seconds is None else max_seconds
    if effective_max_iterations is None or effective_max_iterations < 0:
        effective_max_iterations = 0
    if effective_max_seconds is None or effective_max_seconds <= 0:
        effective_max_seconds = 900

    def classify_input_kind(user_text: str) -> str:
        text = (user_text or "").strip().lower()
        stripped = re.sub(r"[^a-z0-9\s]", "", text)
        compact = re.sub(r"\s+", " ", stripped).strip()

        small_talk = {
            "hi", "hello", "hey", "yo", "sup",
            "good morning", "good afternoon", "good evening",
            "thanks", "thank you", "bye", "goodbye",
        }
        if compact in small_talk:
            return "chat"
        if compact.startswith("how are you"):
            return "chat"
        if compact in {"help", "hello there", "hey there", "what can you do", "who are you"}:
            return "chat"

        chat_prefixes = (
            "what can you",
            "can you help",
            "could you help",
            "how do you work",
            "what are your capabilities",
            "what tools do you have",
            "tell me about yourself",
        )
        if any(compact.startswith(prefix) for prefix in chat_prefixes):
            return "chat"

        # Short conversational questions should default to chat unless they
        # clearly request concrete external actions.
        action_hints = (
            "write ", "create ", "build ", "run ", "execute ", "open ",
            "navigate ", "fetch ", "scrape ", "save ", "edit ", "install ",
        )
        if text.endswith("?") and len(compact.split()) <= 10 and not any(h in compact for h in action_hints):
            return "chat"

        return "task"

    input_kind = classify_input_kind(description)

    system_prompt = """You are WzrdMerlin, an autonomous agent. Select the right tool for each step.
Tools: shell(cmd), read_file(path), write_file(path,content), search_memory(query), write_memory(content,tags), fetch_url(url), request_human(question), done(summary).

RULES:
- Reply with ONLY a JSON object. No prose before or after.
- All strings in JSON must use double quotes and properly escape special chars.
- For shell commands with quotes, use single quotes inside the command or escape with backslash.
- Keep shell commands simple. Prefer python one-liners over complex grep/sed/awk.

Example: {"tool": "shell", "args": {"cmd": "ls -la"}}
When complete: {"tool": "done", "args": {"summary": "what was accomplished"}}
"""

    def tool_preview(tool_name: str, tool_args: dict) -> str:
        if tool_name == "read_file":
            return f"I'll inspect {tool_args.get('path', 'that file')} next."
        if tool_name == "write_file":
            return f"I'll write the update to {tool_args.get('path', 'the target file')}."
        if tool_name == "shell":
            return "I'll run a shell command to verify or change state."
        if tool_name == "fetch_url":
            return f"I'll fetch {tool_args.get('url', 'that URL')} for context."
        if tool_name == "search_memory":
            return "I'll scan memory for relevant prior context."
        if tool_name == "write_memory":
            return "I'll save this as a durable memory entry."
        if tool_name == "request_human":
            return "I need your input before I can proceed safely."
        if tool_name == "done":
            return "Everything needed appears complete — wrapping up now."
        return f"Next action: {tool_name}."

    def tool_observation(tool_name: str, result: str) -> str:
        if not result:
            return "I ran that step and got no output; moving to the next best action."
        if result.lower().startswith("error"):
            return "That step returned an error, so I'll adapt and try a different path."
        if tool_name == "read_file":
            return "I reviewed the file output and will use it to decide the next step."
        if tool_name == "write_file":
            return "The file update completed. I’ll validate the change next."
        if tool_name == "shell":
            return "Command finished. I’ll use this output to continue."
        if tool_name == "fetch_url":
            return "Fetch complete. I’ll extract the useful parts and continue."
        if tool_name == "search_memory":
            return "Memory search returned context I can use for the next action."
        if tool_name == "write_memory":
            return "Memory entry saved — this can help future tasks."
        if tool_name == "request_human":
            return "Paused at a human-input checkpoint."
        return "Step complete — proceeding with the next action."

    async def stream():
        history = []
        iteration = 0
        started_at = time.monotonic()
        last_fingerprint = None
        repeated_fingerprint_count = 0
        repeated_request_human = 0

        def line(obj: dict) -> bytes:
            return (json.dumps(obj) + "\n").encode()

        # Persist task creation in NATS KV for crash recovery
        try:
            await state_store.put(f"task.{task_id}", {
                "task_id": task_id,
                "description": description,
                "status": "running",
                "started_at": int(time.time()),
                "iterations": 0,
            })
            # Update task index
            task_index = await state_store.get("task_index") or {"ids": []}
            if task_id not in task_index["ids"]:
                task_index["ids"].append(task_id)
                # Keep last 200 tasks
                task_index["ids"] = task_index["ids"][-200:]
                await state_store.put("task_index", task_index)
        except Exception as e:
            logger.warning(f"State persistence failed for {task_id}: {e}")

        if input_kind == "chat":
            active_runs[task_id]["status"] = "completed"
            active_runs[task_id]["updated_at"] = int(time.time())
            yield line({"type": "narrative", "text": "Hey — I’m here and ready to help."})
            yield line({
                "type": "done",
                "text": "Share any concrete task and I’ll execute it step-by-step.",
            })
            active_runs.pop(task_id, None)
            return

        while True:
            iteration += 1
            if task_id in active_runs:
                active_runs[task_id]["iterations"] = iteration
                active_runs[task_id]["updated_at"] = int(time.time())

            elapsed = time.monotonic() - started_at
            if elapsed > effective_max_seconds:
                if task_id in active_runs:
                    active_runs[task_id]["status"] = "failed"
                    active_runs[task_id]["updated_at"] = int(time.time())
                yield line({
                    "type": "error",
                    "text": f"Execution timeout reached ({effective_max_seconds}s).",
                })
                active_runs.pop(task_id, None)
                return

            if effective_max_iterations > 0 and iteration > effective_max_iterations:
                if task_id in active_runs:
                    active_runs[task_id]["status"] = "failed"
                    active_runs[task_id]["updated_at"] = int(time.time())
                yield line({
                    "type": "error",
                    "text": f"Max iterations reached ({effective_max_iterations}).",
                })
                active_runs.pop(task_id, None)
                return

            yield line({"type": "thinking", "text": f"[step {iteration}] deciding…"})
            if iteration == 1:
                yield line({
                    "type": "narrative",
                    "text": "I’m on it. I’ll work step-by-step, explain what I’m doing, and keep you updated between tool calls.",
                })

            think_buf, content_buf = [], []
            async for chunk_type, text in llm.generate_action_streaming(system_prompt, history, description):
                if chunk_type == "think":
                    think_buf.append(text)
                    yield line({"type": "thinking", "text": text})
                else:
                    content_buf.append(text)
                    yield line({"type": "token", "text": text})

            full_response = "".join(content_buf)
            action = llm.parse_action(full_response)

            if not action:
                if task_id in active_runs:
                    active_runs[task_id]["status"] = "failed"
                    active_runs[task_id]["updated_at"] = int(time.time())
                yield line({"type": "error", "text": f"No valid JSON in response: {full_response[:200]}"})
                active_runs.pop(task_id, None)
                return

            tool_name = action.get("tool", "?")
            tool_args = action.get("args", {})
            key_arg = next(iter(tool_args.values()), "") if tool_args else ""
            yield line({"type": "narrative", "text": tool_preview(tool_name, tool_args)})
            yield line({"type": "tool_start", "tool": tool_name, "arg": str(key_arg)[:120]})

            result = await base_agent.execute_tool(tool_name, tool_args)
            yield line({"type": "tool_end", "tool": tool_name, "result": str(result)[:500]})
            yield line({"type": "narrative", "text": tool_observation(tool_name, str(result))})

            # Cap result size in history to avoid blowing the context window.
            # The full result was already streamed to the TUI; the LLM only needs
            # enough to decide its next action.
            capped_result = str(result)[:1500]
            if len(str(result)) > 1500:
                capped_result += f"\n... ({len(str(result))} chars total, truncated for context)"
            history.append({"action": action, "result": capped_result})

            if tool_name == "request_human":
                repeated_request_human += 1
            else:
                repeated_request_human = 0

            if repeated_request_human >= 2:
                yield line({
                    "type": "narrative",
                    "text": "I’m still missing required input, so I’m pausing here instead of looping.",
                })
                yield line({
                    "type": "done",
                    "text": str(key_arg) if key_arg else "Please provide the requested details so I can continue.",
                })
                if task_id in active_runs:
                    active_runs[task_id]["status"] = "completed"
                    active_runs[task_id]["updated_at"] = int(time.time())
                active_runs.pop(task_id, None)
                return

            fingerprint = json.dumps(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": str(result)[:200],
                },
                sort_keys=True,
                default=str,
            )
            if fingerprint == last_fingerprint:
                repeated_fingerprint_count += 1
            else:
                repeated_fingerprint_count = 0
                last_fingerprint = fingerprint

            if repeated_fingerprint_count >= stall_repeat_limit and tool_name != "done":
                if task_id in active_runs:
                    active_runs[task_id]["status"] = "failed"
                    active_runs[task_id]["updated_at"] = int(time.time())
                yield line({
                    "type": "error",
                    "text": (
                        "Detected repeated identical action/result loop. "
                        "Stopping to prevent runaway execution."
                    ),
                })
                active_runs.pop(task_id, None)
                return

            if tool_name == "done":
                if task_id in active_runs:
                    active_runs[task_id]["status"] = "completed"
                    active_runs[task_id]["updated_at"] = int(time.time())
                # Persist final state
                try:
                    await state_store.put(f"task.{task_id}", {
                        "task_id": task_id,
                        "description": description,
                        "status": "completed",
                        "iterations": iteration,
                        "result": str(key_arg)[:500],
                        "completed_at": int(time.time()),
                    })
                except Exception:
                    pass
                yield line({"type": "done", "text": str(key_arg)})
                active_runs.pop(task_id, None)
                return

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/tasks")
async def list_tasks():
    # Return tasks from state store
    task_index = await state_store.get("task_index") or {"ids": []}
    tasks = []
    for tid in task_index.get("ids", []):
        t = await state_store.get(f"task.{tid}")
        if t:
            tasks.append(t)
    return {"tasks": tasks}


@app.get("/api/tasks/active")
async def list_active_tasks():
    tasks = list(active_runs.values())
    tasks.sort(key=lambda t: t.get("started_at", 0), reverse=True)
    return {"tasks": tasks, "count": len(tasks)}


@app.post("/api/rollback")
async def rollback_last_improvement():
    """Roll back the most recent self-improvement merge."""
    result = await improvement_mgr.rollback_last()
    return {"result": result}


@app.get("/api/stream")
async def stream_events():
    q = asyncio.Queue()
    ui_queues.append(q)
    
    async def event_generator():
        try:
            while True:
                event = await q.get()
                yield {"event": event.type.value, "data": event.model_dump_json()}
        except asyncio.CancelledError:
            ui_queues.remove(q)
            
    return EventSourceResponse(event_generator())
