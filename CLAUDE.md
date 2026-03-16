# WzrdMerlin v3 — Agent OS Build Blueprint

**What**: Redesigned local AI agent OS fixing three critical failures in v2:
1. GPU drops to 0% mid-task (300s httpx timeout, no keep_alive, no retry logic)
2. Agent stops silently (unhandled publish() RuntimeError, no KV read recovery, missing asyncio timeouts)
3. Bloated footprint (627-line inference.py, 760-line llm.py, plus dead code: litellm, chromadb, textual)

**How**: Single Ollama backend, SQLite FTS5 memory, reliable pub/sub with circuit breaker + reconnect.

**Hardware**: AMD RX 6750 XT (12GB, ROCm), i7-12700K, Qwen3.5-9B via Ollama on Windows host.

---

## Core Principles

- **Reliability first**: Every async operation has a timeout. Every publish has retry. Every KV read has recovery.
- **One inference backend**: Ollama only. Deleted: `inference.py`, `llm.py`, all llama-server logic.
- **One memory store**: SQLite FTS5 primary (no external deps), ChromaDB as optional upgrade.
- **No dead code**: Every import is used. Every file is committed.
- **Tight feedback loop**: When Claude makes a mistake, update this file so it doesn't repeat.

---

## Build Order (4 Phases, ~7 hours total)

Each phase is independently testable. Build in order.

### Phase 1: Reliability Fixes (2 hours)
**Files to modify**: `actor.py`, `base_agent.py`
**Goal**: Fix silent crashes and unhandled exceptions.

**actor.py changes:**
- Add reconnect callbacks: `max_reconnect_attempts=-1`, `reconnect_time_wait=2`
- Add `_on_nats_reconnect()`: re-subscribe all handlers after reconnect
- Rewrite `publish()` as no-raise with 3-attempt retry (2/4/8s backoff)
- **CRITICAL**: `publish()` must never raise. Log error and return. This prevents crash propagation.

**base_agent.py changes — Five surgical fixes:**

1. **Wrap entire `handle_step_requested` body in try/finally** (not just parts):
   ```python
   self._inflight_tasks.add(task_id)
   try:
       state = await self.state_store.get(...)
       # ALL step logic here
   except Exception as e:
       logger.error(f"BASE_AGENT: Unhandled error in step {task_id}: {e}", exc_info=True)
       await self._safe_publish("events.action.failed", Event(...))
   finally:
       self._inflight_tasks.discard(task_id)
   ```

2. **Add `_safe_publish()` helper** (wraps the no-raise publish):
   ```python
   async def _safe_publish(self, subject: str, event: Event) -> None:
       try:
           await self.publish(subject, event)
       except Exception as e:
           logger.error(f"_safe_publish {subject}: {e}")
   ```
   Replace all raw `await self.publish()` calls with `await self._safe_publish()`.

3. **KV read failure → re-queue step** (not silent return):
   ```python
   try:
       state = await self.state_store.get(f"actor_state.{task_id}")
   except Exception as e:
       logger.error(f"BASE_AGENT: KV read failed {task_id}: {e}")
       await asyncio.sleep(2)
       await self._safe_publish("events.step.requested", Event(
           type=EventType.STEP_REQUESTED,
           source_actor=self.name,
           correlation_id=task_id,
           payload={"task_id": task_id},
       ))
       return
   ```

4. **Per-tool timeout** in `tool_parallel_tools`:
   ```python
   TOOL_TIMEOUT = float(os.getenv("MERLIN_TOOL_TIMEOUT_SECONDS", "60"))
   
   async def _run_one_with_timeout(call):
       try:
           return await asyncio.wait_for(_run_one(call), timeout=TOOL_TIMEOUT)
       except asyncio.TimeoutError:
           name = call.get("tool", "unknown")
           return {
               "tool": name,
               "args": call.get("args", {}),
               "result": f"Error: tool '{name}' timed out after {TOOL_TIMEOUT:.0f}s"
           }
   ```

5. **History bounds check** in `resume_with_human_response`:
   ```python
   history = state.get("history", [])
   if not history:
       logger.warning(f"BASE_AGENT: Cannot resume {task_id} — empty history")
       return False
   history[-1]["result"] = f"Human responded: {response}"
   ```

**Verify**: `pytest tests/test_actor.py tests/test_agent_lifecycle.py -v`

---

### Phase 2: Inference (3 hours)
**Files to create/modify**: `ollama_client.py` (NEW), `config.py`, `main.py`, `watchdog.py`, `base_agent.py`, `self_improve.py`, `gardener.py`

**ollama_client.py** (NEW file, ~250 lines):
Core responsibilities:
- Single persistent `httpx.AsyncClient` (reuse TCP pool, never per-call)
- CircuitBreaker: CLOSED → OPEN after 5 consecutive failures, auto-recovery after 60s
- `ensure_model_loaded()`: GET /api/tags, return bool (no pre-pull)
- `stream_chat()`: streaming POST /api/chat with per-chunk timeout (120s), **keep_alive="-1" on every request**, heartbeat callback support
- `chat()`: non-streaming wrapper
- `parse_action()`: migrate **verbatim** from v2's `llm.py::ModelInterface.parse_action()` (6-strategy JSON parser, battle-tested)
- `build_messages()`: migrate **verbatim** from v2's `llm.py::ModelInterface._build_messages()`

**CRITICAL**: The JSON parser and `<think>` tag streaming logic are load-bearing. Do not simplify. Copy exactly from v2 llm.py.

**config.py rewrite**:
- Delete entire `InferenceConfig` class and all llama-server fields
- Keep only: `ModelProfile` (model_name, context_window, temperature, think, think_budget), `OllamaConfig` (base_url, keep_alive, read_timeout), `HardwareConfig` (HSA/RADV overrides)

**main.py changes**:
- Delete global `inference_mgr = LlamaCppManager()`
- Delete llama-server startup block (`if cfg.inference.backend == "llama.cpp"`)
- Delete ChromaDB reindex block (SQLite FTS needs no reindex)
- Add global `ollama_client = OllamaClient(...)`
- In lifespan startup: call `ollama_client.ensure_model_loaded()`, warn if not loaded
- Wire agents: `base_agent.ollama = ollama_client`, `watchdog._ollama_client = ollama_client`
- Update `/api/llm/health` to call `ollama_client.ensure_model_loaded()`
- Update `/api/models/switch` to re-create `OllamaClient` with new model name (no process start/stop)

**base_agent.py changes**:
- Replace `from src.core.llm import ModelInterface` with `from src.core.ollama_client import OllamaClient`
- In `__init__`, replace `self.llm = ModelInterface()` with:
  ```python
  self.ollama = OllamaClient(
      base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
      model=get_config().get_active_model().model_name,
      context_window=get_config().get_active_model().context_window,
      temperature=get_config().get_active_model().temperature,
      think=get_config().get_active_model().think,
  )
  ```
- In `_generate_single_action`: replace `self.llm.generate_action_streaming(...)` with `self.ollama.stream_chat(messages, heartbeat_cb=...)`
- In `_generate_single_action`: replace `self.llm.parse_action(...)` with `self.ollama.parse_action(...)`
- In `_self_consistency_cascade`: replace `self.llm.generate_action(...)` with `self.ollama.chat(...)`

**watchdog.py changes**:
- Delete entire inference_mgr telemetry block (~30 lines)
- Add `self._ollama_client` reference in `__init__`
- In `_hardware_loop`: replace llama-server telemetry with Ollama health check:
  ```python
  ollama_ok = False
  if self._ollama_client:
      ollama_ok = await self._ollama_client.ensure_model_loaded()
  
  heartbeat = Event(
      type=EventType.SYSTEM_HEARTBEAT,
      source_actor=self.name,
      correlation_id="system",
      payload={
          "ram_usage": vm.percent,
          "cpu_usage": cpu,
          "ollama_healthy": ollama_ok,
      },
  )
  await self._safe_publish("events.system.heartbeat", heartbeat)
  ```

**self_improve.py changes**:
- Replace `self.llm = ModelInterface()` with `self.ollama = OllamaClient(...)`
- Update `_apply_patch` to call `self.ollama.chat(...)` instead of `self.llm` method

**gardener.py changes**:
- `GardenerActor.set_llm(llm)` now receives `OllamaClient` instead of `ModelInterface`
- Update `_extract_facts` calls from `self._llm.generate_text(...)` to `await self._llm.chat(...messages...)`

**Verify**: `pytest tests/test_ollama_client.py -v` + manual `POST /api/task?description=hello`

---

### Phase 3: Memory (2 hours)
**Files to create/modify**: `memory.py` (rewrite), `memory_chroma.py` (NEW, copy), `requirements.txt`

**memory.py rewrite** (SQLiteMemoryStore):
- Use SQLite FTS5 (no external dependencies, no embedding calls)
- DB at `{MERLIN_WORKSPACE}/.merlin/memory.db`
- Schema: `memories(id TEXT PK, content TEXT, tags TEXT, metadata TEXT, timestamp INTEGER, collection TEXT)` + `memories_fts` FTS virtual table
- Two collections: `episodic` and `trajectories`
- Public API unchanged (callers need zero changes):
  - `async def add(content, tags=None, metadata=None, collection="episodic") -> str`
  - `async def search(query, top_k=5, collection="episodic", min_score=0.0, where=None) -> List[Dict]`
  - `async def recall(query, top_k=3) -> str` (formatted for prompt injection)
  - `async def prune(max_age_days=90) -> int`
  - `def count(collection="episodic") -> int`
  - `async def migrate_legacy() -> int` (import /workspace/memory/*.json, idempotent)
- FTS5 score normalization: `score = 1.0 / (1.0 + abs(bm25_rank))`

**memory_chroma.py** (NEW file):
- Copy v2's `memory.py` **verbatim** (full ChromaDB + Ollama embedding implementation)
- Only loaded if `chromadb` is installed

**requirements.txt**:
- Remove: `litellm`, `chromadb` (hard dependency), `textual`
- Keep all others unchanged
- Add comment: `# chromadb>=0.5.0   ← optional; install manually if needed`

**main.py**: Update singleton factory:
```python
def get_memory():
    global _memory_instance
    if _memory_instance is None:
        try:
            import chromadb  # noqa: F401
            from src.core.memory_chroma import EpisodicMemory
            _memory_instance = EpisodicMemory()
            logger.info("MEMORY: Using ChromaDB vector store")
        except ImportError:
            _memory_instance = SQLiteMemoryStore()
            logger.info("MEMORY: Using SQLite FTS5 store")
    return _memory_instance
```

**Verify**: `pytest tests/test_memory.py -v`

---

### Phase 4: Cleanup (1 hour)
**Files to modify**: `mcp/manager.py`, `docker-compose.yml`

**mcp/manager.py**:
- Find the busy polling loop:
  ```python
  # v2 (BAD)
  while name in self.sessions:
      await asyncio.sleep(1)
  ```
- Replace with:
  ```python
  # v3 (GOOD)
  _session_closed = asyncio.Event()
  # ... when session closes, set the event:
  await _session_closed.wait()
  ```

**docker-compose.yml**:
- Remove `sidecar` service entirely (Rust telemetry binary for llama-server, no longer needed)
- Add `extra_hosts` to `core` service: `extra_hosts: ["host.docker.internal:host-gateway"]`
- Keep `nats`, `core`, `dashboard` services unchanged

**Verify**: `docker compose up -d && curl http://localhost:8000/api/health`

---

## Key Files — Copy Verbatim from v2 (No Changes)

- `events.py` (event model, EventType enum)
- `router.py` (DisCoRouter logic)
- `state.py` (NATS KV wrapper)
- `src/tui/app.py` (TUI works)
- `self_improve.py` (kept as-is, only import updated)
- `mcp/forage.py` (ForagePipeline fine as on-demand tool)
- `mcp/codemode.py` (CodeModeSandbox)
- `Modelfile` (custom Ollama model definition)
- `entrypoint.sh` (Docker entrypoint)

---

## Config & Environment

**merlin.config.yaml** (new simplified version):
```yaml
active_model: merlin-brain

models:
  merlin-brain:
    model_name: merlin-model:latest
    context_window: 32768
    temperature: 0.3
    think: false
    think_budget: 0

  qwen3.5-14b:
    model_name: qwen3.5:14b
    context_window: 4096
    temperature: 0.3
    think: true
    think_budget: 2048

ollama:
  base_url: http://host.docker.internal:11434
  keep_alive: "-1"
  read_timeout: 120.0

hardware:
  hsa_override_gfx_version: "10.3.0"
  radv_perftest: nogttspill
```

**Environment vars for Docker**:
```bash
NATS_URL=nats://nats:4222
OLLAMA_BASE_URL=http://host.docker.internal:11434
MERLIN_WORKSPACE=/workspace
MERLIN_STEP_TIMEOUT_SECONDS=90
MERLIN_TOOL_TIMEOUT_SECONDS=60
MAX_ITERATIONS=20
```

---

## Ollama Client Implementation Rules (Non-Negotiable)

These are the specific mistakes that broke v2. Don't repeat them.

1. **`keep_alive: "-1"` on every POST body** — Ollama unloads models after 5 min idle. This prevents that.
2. **Per-chunk timeout via `httpx.Timeout`** — Set `read=120.0` so hung connections fail after 120s, not 300s.
3. **Single shared `httpx.AsyncClient`** — Never use `async with httpx.AsyncClient()` per-call. Reuse the same client instance.
4. **Circuit breaker** — CLOSED by default, OPEN after 5 failures, auto-recovery (HALF_OPEN) after 60s.
5. **Retry logic** — 3 attempts on connection error with 2/4/8s exponential backoff.
6. **JSON parser migration** — Copy the 6-strategy parser from v2's `llm.py` **verbatim**. Do not simplify.
7. **Heartbeat callback** — `stream_chat()` calls `heartbeat_cb` on every chunk so watchdog knows we're alive.

---

## Testing Checklist

After each phase, run the relevant tests. After all phases complete:

```bash
pytest tests/ -v                                                    # All unit tests
docker compose up -d                                               # Start services
curl http://localhost:8000/api/health                              # Health check
curl http://localhost:8000/api/llm/health                          # Model loaded?
curl -X POST "http://localhost:8000/api/task?description=hello"   # Submit task
# Watch SSE stream to observe events completing
docker compose restart nats                                         # Simulate NATS failure
# Agent should recover and continue within 2 seconds
```

---

## When Claude Goes Wrong

If Claude:
- Simplifies the JSON parser → revert, exact copy from v2 required
- Auto-commits without asking → remind me to update docker-compose.yml to `never-commit` setting
- Creates new files in deleted dirs → reference this CLAUDE.md: deleted files are `inference.py`, `llm.py` only
- Forgets the 5 base_agent.py fixes → tell Claude to append to this file so it remembers
- Adds chromadb as hard requirement → remove, it's optional only

---

## Code Size Target

| File | v2 lines | v3 target | Status |
|------|----------|-----------|--------|
| inference.py | 627 | **deleted** | ✓ |
| llm.py | 760 | **deleted** | ✓ |
| ollama_client.py | — | ~250 | NEW |
| config.py | 181 | ~120 | REWRITE |
| memory.py | 424 | ~200 | REWRITE |
| watchdog.py | 135 | ~105 | TRIM |
| actor.py | 108 | ~160 | EXTEND |
| base_agent.py | ~1350 | ~1200 | FIX |
| **Total** | **~5,500** | **~3,900** | **-29%** |

---

## NATS Subject Convention

All must start with `events.`:
- `events.task.created` — API → router
- `events.task.routed` — DisCoRouter → (UI sniff)
- `events.actor.{role}` — DisCoRouter → BaseAgentActor
- `events.step.requested` — BaseAgentActor → BaseAgentActor
- `events.action.completed` — BaseAgentActor → DisCoRouter
- `events.action.failed` — BaseAgentActor / WatchdogActor → DisCoRouter
- `events.system.improvement` — BaseAgentActor → ImprovementManager
- `events.system.heartbeat` — WatchdogActor → (UI sniff)

**KV Bucket**: `MERLIN_STATE`
**Key pattern**: `actor_state.{task_id}`