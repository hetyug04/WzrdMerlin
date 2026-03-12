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
import httpx
from pathlib import Path, PurePosixPath
from src.core.events import Event, EventType
from src.core.router import DisCoRouter
from src.core.base_agent import BaseAgentActor
from src.core.state import StateStore
from src.core.self_improve import ImprovementManager
from src.core.watchdog import WatchdogActor

logger = logging.getLogger(__name__)

# Global instances
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("MAIN: Starting up WzrdMerlin v2 Actors...")
    
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
    task_id = f"task_{int(time.time())}"
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
