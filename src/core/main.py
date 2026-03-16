"""
WzrdMerlin v3 — FastAPI backend

v3 changes:
- inference_mgr (LlamaCppManager) removed entirely
- OllamaClient global shared across all agents
- lifespan: no llama-server startup, no ChromaDB reindex block
- /api/llm/health delegates to ollama_client.ensure_model_loaded()
- /api/models/switch: re-creates OllamaClient (no process management)
"""
import logging
import logging.config
import collections

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
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "nats": {"level": "WARNING"},
    },
})

# In-process log buffer — last 500 lines, readable via GET /api/logs
_log_buffer: collections.deque = collections.deque(maxlen=500)

class _DequeHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_buffer.append(self.format(record))
        except Exception:
            pass

_dh = _DequeHandler()
_dh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_dh)

# In-memory SSE event snapshot — last 200 events, readable via GET /api/events/recent
_event_buffer: collections.deque = collections.deque(maxlen=200)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse
from contextlib import asynccontextmanager
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from src.core.events import Event, EventType
from src.core.router import DisCoRouter
from src.core.base_agent import BaseAgentActor
from src.core.state import StateStore
from src.core.self_improve import ImprovementManager
from src.core.watchdog import WatchdogActor
from src.core.ollama_client import OllamaClient
from src.core.config import get_config, get_config_path, reload_config
from src.core.memory import get_memory
from src.core.gardener import GardenerActor, COOLDOWN_SECONDS

logger = logging.getLogger(__name__)

# Global instances
router_actor = DisCoRouter()
base_agent = BaseAgentActor()
auditor_agent = BaseAgentActor(role="auditor")
improvement_mgr = ImprovementManager()
state_store = StateStore()
watchdog = WatchdogActor()
gardener = GardenerActor()

# Global OllamaClient — shared across agents
_cfg = get_config()
_active = _cfg.get_active_model()
ollama_client = OllamaClient(
    base_url=os.getenv("OLLAMA_BASE_URL", _cfg.ollama.base_url),
    model=_active.model_name,
    context_window=_active.context_window,
    temperature=_active.temperature,
    think=_active.think,
    read_timeout=_cfg.ollama.read_timeout,
    keep_alive=_cfg.ollama.keep_alive,
)

# In-memory queues for UI SSE streaming
ui_queues = []

# Tracks which task is waiting for human input
_human_pending_task = None


async def broadcast_to_ui(event: Event):
    global _human_pending_task
    _event_buffer.append(event)
    logger.info(f"BROADCAST: Sending event {event.type.value} to {len(ui_queues)} listeners")
    if event.type == EventType.AGENT_TOOL_START and event.payload.get("tool") == "request_human":
        _human_pending_task = event.correlation_id
    elif event.type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED, EventType.ACTION_FAILED):
        if event.correlation_id == _human_pending_task:
            _human_pending_task = None
    for q in ui_queues:
        await q.put(event)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("MAIN: Starting up WzrdMerlin v3...")

    cfg = get_config()
    active = cfg.get_active_model()

    # Evict any stale models left in VRAM from a previous session
    # (e.g. worker 2B models loaded with keep_alive=-1 that survived a restart)
    await ollama_client.evict_all_models()

    # Check model exists in Ollama, then preload it into GPU
    loaded = await ollama_client.ensure_model_loaded()
    if not loaded:
        logger.warning(
            f"MAIN: Model '{active.model_name}' not loaded in Ollama — "
            "inference will fail until model is pulled"
        )
    else:
        logger.info(f"MAIN: Model '{active.model_name}' confirmed in Ollama — preloading into GPU")
        await ollama_client.preload_model()

    # Wire OllamaClient into agents
    base_agent.ollama = ollama_client
    auditor_agent.ollama = ollama_client
    watchdog._ollama_client = ollama_client

    # Inject UI broadcast callback
    base_agent._ui_broadcast = broadcast_to_ui
    auditor_agent._ui_broadcast = broadcast_to_ui

    # Connect NATS actors
    await router_actor.connect()
    await base_agent.connect()
    await auditor_agent.connect()
    await improvement_mgr.connect()
    await state_store.connect()
    await watchdog.connect()
    await gardener.connect()
    gardener.set_llm(ollama_client)

    # Sniff all events via core NATS wildcard
    async def _sniff_events(msg):
        try:
            data = json.loads(msg.data.decode())
            event = Event(**data)
            logger.info(f"SNIFF: Caught event {event.type.value} from {event.source_actor}")
            await broadcast_to_ui(event)
        except Exception as e:
            logger.error(f"SNIFF ERROR: {e}")

    await router_actor.nc.subscribe("events.>", cb=_sniff_events)
    logger.info("MAIN: NATS sniff subscription active on 'events.>'")

    # Episodic memory: migrate legacy JSON files → SQLite (one-time, idempotent)
    try:
        mem = get_memory()
        migrated = await mem.migrate_legacy()
        if migrated:
            logger.info(f"MAIN: Migrated {migrated} legacy memory entries")
        logger.info(
            f"MAIN: Memory ready "
            f"(episodic={mem.count()}, trajectories={mem.count('trajectories')})"
        )
    except Exception as e:
        logger.warning(f"MAIN: Memory init warning: {e}")

    yield

    # SHUTDOWN
    logger.info("MAIN: Shutting down actors...")
    await ollama_client.aclose()
    await router_actor.close()
    await base_agent.close()
    await auditor_agent.close()
    await improvement_mgr.close()
    await state_store.close()
    await watchdog.close()
    await gardener.close()


app = FastAPI(title="WzrdMerlin v3 Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
#  Task API
# ------------------------------------------------------------------

@app.post("/api/task")
async def create_task(description: str):
    global _human_pending_task
    if _human_pending_task:
        waiting_task_id = _human_pending_task
        _human_pending_task = None
        logger.info(f"API: Routing as human response to waiting task {waiting_task_id}")
        resumed = await base_agent.resume_with_human_response(waiting_task_id, description)
        if resumed:
            return {"status": "responded", "task_id": waiting_task_id}
        logger.warning(f"API: Failed to resume {waiting_task_id}, creating new task")

    logger.info(f"API: Received task request: {description}")
    task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    evt = Event(
        type=EventType.TASK_CREATED,
        source_actor="api",
        correlation_id=task_id,
        payload={"task_id": task_id, "description": description},
    )
    logger.info(f"API: Publishing events.task.created for {task_id}")
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
    """Ollama model health — delegates to OllamaClient.ensure_model_loaded()."""
    cfg = get_config()
    active = cfg.get_active_model()
    loaded = await ollama_client.ensure_model_loaded()
    return {
        "status": "ok" if loaded else "model_not_pulled",
        "model_loaded": loaded,
        "backend": "ollama",
        "active_model": active.model_name,
        "base_url": ollama_client.base_url,
        "circuit_breaker_open": ollama_client._breaker.is_open(),
    }


@app.get("/api/debug/actors")
async def debug_actors():
    return {
        "router": {
            "connected": router_actor.nc is not None and router_actor.nc.is_connected,
            "handlers": list(router_actor._handlers.keys()),
        },
        "agent": {
            "connected": base_agent.nc is not None and base_agent.nc.is_connected,
            "handlers": list(base_agent._handlers.keys()),
            "mcp_tools": list(base_agent.mcp_manager.tools.keys()),
        },
        "auditor": {
            "connected": auditor_agent.nc is not None and auditor_agent.nc.is_connected,
            "handlers": list(auditor_agent._handlers.keys()),
        },
        "ui_listeners": len(ui_queues),
    }


@app.get("/api/stream")
async def stream_events(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    ui_queues.append(q)
    logger.info(f"STREAM: New UI listener connected. Total: {len(ui_queues)}")

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=5.0)
                    yield {"data": event.model_dump_json()}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "keep-alive"}
        except Exception as e:
            logger.error(f"STREAM ERROR: {e}")
        finally:
            if q in ui_queues:
                ui_queues.remove(q)
            logger.info(f"STREAM: UI listener removed. Remaining: {len(ui_queues)}")

    return EventSourceResponse(event_generator())


# ------------------------------------------------------------------
#  Model / Config API
# ------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    cfg = get_config()
    active = cfg.get_active_model()
    env_override = os.getenv("MERLIN_MODEL")
    return {
        "active_model": env_override or cfg.active_model,
        "env_override": env_override,
        "config_file": get_config_path(),
        "active_profile": {
            "model_name": active.model_name,
            "context_window": active.context_window,
            "temperature": active.temperature,
            "think": active.think,
            "think_budget": active.think_budget,
        },
        "models": {
            name: {
                "model_name": p.model_name,
                "context_window": p.context_window,
                "think": p.think,
            }
            for name, p in cfg.models.items()
        },
    }


@app.post("/api/models/switch")
async def switch_model(model_name: str):
    """Hot-swap the active model. Re-creates OllamaClient with new profile settings."""
    global ollama_client
    cfg = get_config()
    try:
        cfg.switch_active_model(model_name)
    except ValueError as e:
        return {"error": str(e), "status": "failed"}

    active = cfg.get_active_model()
    old_client = ollama_client
    ollama_client = OllamaClient(
        base_url=os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url),
        model=active.model_name,
        context_window=active.context_window,
        temperature=active.temperature,
        think=active.think,
        read_timeout=cfg.ollama.read_timeout,
        keep_alive=cfg.ollama.keep_alive,
    )
    base_agent.ollama = ollama_client
    auditor_agent.ollama = ollama_client
    watchdog._ollama_client = ollama_client
    gardener.set_llm(ollama_client)
    await old_client.aclose()
    logger.info(f"MAIN: Switched active model to '{model_name}'")
    return {"status": "switched", "active_model": model_name}


@app.post("/api/config/reload")
async def config_reload():
    cfg = reload_config()
    return {
        "status": "reloaded",
        "config_file": get_config_path(),
        "active_model": cfg.active_model,
        "models": cfg.list_models(),
    }


# ------------------------------------------------------------------
#  Episodic Memory API
# ------------------------------------------------------------------

@app.get("/api/memory/stats")
async def memory_stats():
    mem = get_memory()
    return {
        "episodic_count": mem.count("episodic"),
        "trajectory_count": mem.count("trajectories"),
    }


@app.get("/api/gardener/status")
async def gardener_status():
    return {
        "running": gardener._running,
        "last_run": gardener._last_run,
        "idle_since": time.time() - gardener._last_active_task,
        "cooldown_remaining": max(0, COOLDOWN_SECONDS - (time.time() - gardener._last_run))
        if gardener._last_run else 0,
    }


@app.post("/api/gardener/run")
async def gardener_trigger():
    if gardener._running:
        return {"status": "already_running"}
    asyncio.create_task(gardener._run_consolidation())
    return {"status": "started"}


@app.get("/api/memory/search")
async def memory_search(query: str, top_k: int = 5, collection: str = "episodic"):
    mem = get_memory()
    results = await mem.search(query, top_k=top_k, collection=collection, min_score=0.1)
    return {"query": query, "results": results}


@app.post("/api/memory/prune")
async def memory_prune(max_age_days: int = 90):
    mem = get_memory()
    deleted = await mem.prune(max_age_days=max_age_days)
    return {"deleted": deleted}


@app.post("/api/memory/reindex")
async def memory_reindex(collection: str = "episodic"):
    mem = get_memory()
    count = await mem.reindex(collection=collection)
    return {"reindexed": count, "collection": collection}


@app.post("/api/memory/trajectory")
async def add_trajectory(request: Request):
    body = await request.json()
    content = body.get("content", "")
    task_description = body.get("task", "")
    outcome = body.get("outcome", "completed")
    if not content:
        return {"error": "content is required"}
    mem = get_memory()
    doc_id = await mem.add(
        content=content,
        tags=["trajectory", "teacher", f"outcome:{outcome}"],
        metadata={"task_id": "manual", "outcome": outcome, "task": task_description},
        collection="trajectories",
    )
    return {"id": doc_id, "collection": "trajectories"}


# ------------------------------------------------------------------
#  File Explorer API
# ------------------------------------------------------------------

WORKSPACE = os.getenv("MERLIN_WORKSPACE", "/workspace")


def _safe_resolve(requested: str) -> Path:
    base = Path(WORKSPACE).resolve()
    target = (base / requested).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError("Path traversal blocked")
    return target


@app.get("/api/files")
async def list_files(path: str = ""):
    try:
        target = _safe_resolve(path)
    except ValueError:
        return {"error": "Invalid path"}
    if not target.exists():
        return {"error": "Path not found"}
    if not target.is_dir():
        return {"error": "Not a directory"}
    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                stat = child.stat()
                entries.append({
                    "name": child.name,
                    "type": "directory" if child.is_dir() else "file",
                    "size": stat.st_size if child.is_file() else None,
                    "modified": int(stat.st_mtime),
                })
            except OSError:
                continue
    except PermissionError:
        return {"error": "Permission denied"}
    return {"path": str(target.relative_to(Path(WORKSPACE).resolve())), "entries": entries}


@app.get("/api/files/read")
async def read_file_content(path: str):
    try:
        target = _safe_resolve(path)
    except ValueError:
        return PlainTextResponse("Invalid path", status_code=400)
    if not target.exists():
        return PlainTextResponse("File not found", status_code=404)
    if not target.is_file():
        return PlainTextResponse("Not a file", status_code=400)
    if target.stat().st_size > 1_048_576:
        return PlainTextResponse("File too large (>1MB)", status_code=413)
    try:
        content = target.read_text(errors="replace")
        return PlainTextResponse(content)
    except Exception as e:
        return PlainTextResponse(f"Error reading file: {e}", status_code=500)


# ------------------------------------------------------------------
#  Task Status / Cancel / Debug APIs
# ------------------------------------------------------------------

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Read current task state from NATS KV."""
    try:
        state = await state_store.get(f"actor_state.{task_id}")
        if state is None:
            return {"error": "not found", "task_id": task_id}
        return {
            "task_id": task_id,
            "status": state.get("status", "unknown"),
            "iteration": state.get("iteration", 0),
            "result": str(state.get("result", ""))[:500],
            "instruction": str(state.get("instruction", ""))[:200],
        }
    except Exception as e:
        return {"error": str(e), "task_id": task_id}


@app.post("/api/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel an in-flight task by setting its status to 'cancelled' in NATS KV."""
    try:
        state = await state_store.get(f"actor_state.{task_id}")
        if state is None:
            return {"error": "not found", "task_id": task_id}
        state["status"] = "cancelled"
        await state_store.put(f"actor_state.{task_id}", state)
        return {"status": "cancelled", "task_id": task_id}
    except Exception as e:
        return {"error": str(e), "task_id": task_id}


@app.get("/api/logs")
async def get_logs(lines: int = 50):
    """Return last N lines from the in-process log buffer."""
    lines = min(lines, 500)
    recent = list(_log_buffer)[-lines:]
    return {"lines": recent, "total_buffered": len(_log_buffer)}


@app.get("/api/events/recent")
async def get_recent_events(task_id: str = "", limit: int = 20):
    """Return last N SSE events, optionally filtered by task_id (correlation_id)."""
    limit = min(limit, 200)
    events = list(_event_buffer)
    if task_id:
        events = [e for e in events if e.correlation_id == task_id]
    recent = events[-limit:]
    return {
        "events": [
            {
                "type": e.type.value,
                "correlation_id": e.correlation_id,
                "source_actor": e.source_actor,
                "payload_summary": str(e.payload)[:200],
            }
            for e in recent
        ],
        "count": len(recent),
        "task_id_filter": task_id or None,
    }


# ------------------------------------------------------------------
#  Rollback API
# ------------------------------------------------------------------

@app.post("/api/rollback")
async def rollback():
    """Revert the most recent self-improvement merge."""
    result = await improvement_mgr.rollback_last()
    return {"result": result}
