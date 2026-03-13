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
from fastapi.responses import StreamingResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse
from contextlib import asynccontextmanager
import asyncio
import json
import os
import re
import time
import uuid
import httpx
from pathlib import Path, PurePosixPath
from src.core.events import Event, EventType
from src.core.router import DisCoRouter
from src.core.base_agent import BaseAgentActor
from src.core.state import StateStore
from src.core.self_improve import ImprovementManager
from src.core.watchdog import WatchdogActor
from src.core.inference import LlamaCppManager
from src.core.config import get_config, get_config_path, reload_config

logger = logging.getLogger(__name__)

# Global instances
inference_mgr = LlamaCppManager()
router_actor = DisCoRouter()
base_agent = BaseAgentActor()
auditor_agent = BaseAgentActor(role="auditor")
improvement_mgr = ImprovementManager()
state_store = StateStore()
watchdog = WatchdogActor()

# In-memory queues for UI SSE streaming
ui_queues = []

# Tracks which task, if any, is waiting for human input
_human_pending_task = None

async def broadcast_to_ui(event: Event):
    global _human_pending_task
    logger.info(f"BROADCAST: Sending event {event.type.value} to {len(ui_queues)} listeners")
    # Track request_human suspensions for routing
    if event.type == EventType.AGENT_TOOL_START and event.payload.get("tool") == "request_human":
        _human_pending_task = event.correlation_id
    elif event.type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED, EventType.ACTION_FAILED):
        if event.correlation_id == _human_pending_task:
            _human_pending_task = None
    for q in ui_queues:
        await q.put(event)


async def _get_ollama_health() -> dict:
    """Report health for Ollama-backed inference."""
    cfg = get_config()
    active = cfg.get_active_model()
    base_url = os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code != 200:
                return {
                    "status": "unhealthy",
                    "model_loaded": False,
                    "backend": "ollama",
                    "active_model": active.model_name,
                    "base_url": base_url,
                    "detail": f"Ollama returned HTTP {resp.status_code}",
                }

            data = resp.json()
            models = data.get("models", [])
            installed = any(
                model.get("name") == active.model_name or model.get("model") == active.model_name
                for model in models
            )

            return {
                "status": "ok" if installed else "model_not_pulled",
                "model_loaded": installed,
                "backend": "ollama",
                "active_model": active.model_name,
                "base_url": base_url,
                "available_models": [m.get("name") or m.get("model") for m in models],
            }
    except Exception as e:
        return {
            "status": "unreachable",
            "model_loaded": False,
            "backend": "ollama",
            "active_model": active.model_name,
            "base_url": base_url,
            "detail": str(e),
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("MAIN: Starting up WzrdMerlin v2 Actors...")

    cfg = get_config()

    # Wire inference manager into agents so LLM calls route through llama-server
    base_agent.llm.inference_manager = inference_mgr
    auditor_agent.llm.inference_manager = inference_mgr

    # Pass inference manager to watchdog for telemetry
    watchdog.inference_mgr = inference_mgr

    # Auto-start local llama-server only when that backend is enabled.
    if cfg.inference.backend.lower() != "ollama" and inference_mgr.default_model_path:
        try:
            logger.info("MAIN: Starting inference engine...")
            await inference_mgr.start()
            logger.info("MAIN: Inference engine ready")
        except Exception as e:
            logger.warning(f"MAIN: Inference engine start failed (will retry on first call): {e}")
    elif cfg.inference.backend.lower() == "ollama":
        logger.info("MAIN: Inference backend is Ollama — skipping llama-server startup")
    else:
        logger.info("MAIN: No LLAMA_MODEL_PATH set — inference engine will start on demand")

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

    yield

    # SHUTDOWN
    logger.info("MAIN: Shutting down actors...")
    await inference_mgr.shutdown()
    await router_actor.close()
    await base_agent.close()
    await auditor_agent.close()
    await improvement_mgr.close()
    await state_store.close()
    await watchdog.close()

app = FastAPI(title="WzrdMerlin v2 Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/task")
async def create_task(description: str):
    global _human_pending_task
    # If a task is waiting for human input, route as a response
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
        payload={"task_id": task_id, "description": description}
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
    """Inference engine health + telemetry."""
    cfg = get_config()
    if cfg.inference.backend.lower() == "ollama":
        return await _get_ollama_health()

    health = await inference_mgr.get_health()
    tel = inference_mgr.last_telemetry
    if tel:
        health["tokens_per_sec"] = tel.tokens_per_sec
        health["tokens_predicted"] = tel.tokens_predicted
        health["kv_cache_used"] = tel.kv_cache_used_cells
        health["kv_cache_total"] = tel.kv_cache_total_cells
        health["requests_processing"] = tel.requests_processing
    health["slots"] = inference_mgr.get_all_slots_info()
    return health


@app.get("/api/debug/actors")
async def debug_actors():
    return {
        "router": {
            "connected": router_actor.nc is not None and router_actor.nc.is_connected,
            "handlers": list(router_actor._handlers.keys())
        },
        "agent": {
            "connected": base_agent.nc is not None and base_agent.nc.is_connected,
            "handlers": list(base_agent._handlers.keys()),
            "mcp_tools": list(base_agent.mcp_manager.tools.keys())
        },
        "auditor": {
            "connected": auditor_agent.nc is not None and auditor_agent.nc.is_connected,
            "handlers": list(auditor_agent._handlers.keys()),
            "mcp_tools": list(auditor_agent.mcp_manager.tools.keys())
        },
        "ui_listeners": len(ui_queues)
    }

@app.get("/api/stream")
async def stream_events(request: Request):
    q = asyncio.Queue()
    ui_queues.append(q)
    logger.info(f"STREAM: New UI listener connected. Total: {len(ui_queues)}")
    
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=5.0)
                    # Omit "event" key so it triggers eventSource.onmessage in the browser
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
    """List available model profiles and the currently active one."""
    cfg = get_config()
    active = cfg.get_active_model()
    env_override = os.getenv("MERLIN_MODEL")
    return {
        "active_model": env_override or cfg.active_model,
        "env_override": env_override,
        "config_file": get_config_path(),
        "active_profile": {
            "model_name": active.model_name,
            "model_path": active.model_path,
            "context_window": active.context_window,
            "gpu_layers": active.gpu_layers,
            "kv_cache_type_k": active.kv_cache_type_k,
            "kv_cache_type_v": active.kv_cache_type_v,
            "think": active.think,
            "think_budget": active.think_budget,
            "temperature": active.temperature,
        },
        "models": {
            name: {
                "model_name": p.model_name,
                "model_path": p.model_path,
                "context_window": p.context_window,
                "gpu_layers": p.gpu_layers,
                "think": p.think,
            }
            for name, p in cfg.models.items()
        },
    }


@app.post("/api/models/switch")
async def switch_model(model_name: str):
    """
    Hot-swap the active model. Stops the current llama-server and starts
    a new one with the specified profile's settings.
    """
    try:
        slot = await inference_mgr.switch_model(model_name)
        # Refresh all LLM instances so they pick up the new model settings
        for llm in (base_agent.llm, auditor_agent.llm):
            llm.refresh_from_config()
        return {
            "status": "switched",
            "active_model": model_name,
            "port": slot.port,
            "pid": slot.pid,
        }
    except ValueError as e:
        return {"error": str(e), "status": "failed"}
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        return {"error": str(e), "status": "failed"}


@app.post("/api/config/reload")
async def config_reload():
    """Reload merlin.config.yaml from disk without restarting the service."""
    cfg = reload_config()
    return {
        "status": "reloaded",
        "config_file": get_config_path(),
        "active_model": cfg.active_model,
        "models": cfg.list_models(),
    }


# ------------------------------------------------------------------
#  File Explorer API (Docker container workspace)
# ------------------------------------------------------------------

WORKSPACE = os.getenv("MERLIN_WORKSPACE", "/workspace")

def _safe_resolve(requested: str) -> Path:
    """Resolve a requested path and ensure it stays within WORKSPACE."""
    base = Path(WORKSPACE).resolve()
    target = (base / requested).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError("Path traversal blocked")
    return target


@app.get("/api/files")
async def list_files(path: str = ""):
    """List directory contents. Returns name, type, size, modified for each entry."""
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
    """Read file content. Returns plain text for text files, error for binary/large."""
    try:
        target = _safe_resolve(path)
    except ValueError:
        return PlainTextResponse("Invalid path", status_code=400)

    if not target.exists():
        return PlainTextResponse("File not found", status_code=404)
    if not target.is_file():
        return PlainTextResponse("Not a file", status_code=400)

    # Cap at 1MB for safety
    if target.stat().st_size > 1_048_576:
        return PlainTextResponse("File too large (>1MB)", status_code=413)

    try:
        content = target.read_text(errors="replace")
        return PlainTextResponse(content)
    except Exception as e:
        return PlainTextResponse(f"Error reading file: {e}", status_code=500)
