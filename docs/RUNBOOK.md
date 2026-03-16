# WzrdMerlin v3 — Operations Runbook

## Quick Start

```bash
cd C:/CS/sideProjects/WzrdMerlinV3
docker compose up -d
python -m src.tui          # launch TUI
```

## Health Checks

<!-- AUTO-GENERATED: from main.py endpoints -->
| Endpoint | What it checks | Healthy response |
|----------|----------------|-----------------|
| `GET /api/health` | NATS connectivity | `{"status":"ok","nats":"connected"}` |
| `GET /api/llm/health` | Ollama model loaded + circuit breaker | `{"status":"ok","model_loaded":true,...}` |
| `GET /api/debug/actors` | Per-actor NATS state + handler registry | all `"connected": true` |
<!-- END AUTO-GENERATED -->

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/llm/health
curl http://localhost:8000/api/debug/actors
```

---

## Common Issues & Fixes

### Circuit breaker is open

**Symptom**: `OllamaUnavailableError: circuit breaker is open for merlin-model:latest`

**Cause**: 5 consecutive failures talking to Ollama (bad response, wrong `keep_alive` format, model not loaded).

**Fix**:
1. Check Ollama is running on the host: `ollama list`
2. Check `/api/llm/health` — `model_loaded` should be `true`
3. The breaker auto-recovers after 60 seconds (HALF_OPEN → CLOSED on next success)
4. If it keeps re-opening, check Docker logs: `docker logs wzrdmerlinv3-core-1 --tail 50`

**Known gotcha**: `keep_alive: "-1"` in `merlin.config.yaml` is normalized to integer `-1` in `OllamaClient.__init__`. If you see `"time: missing unit in duration"` errors, verify `ollama_client.py` line 96–100 has the normalization logic.

---

### Port 4222 or 8000 already allocated

**Cause**: Previous v2 containers still running.

**Fix**:
```bash
docker stop wzrdmerlinv2-nats-1 wzrdmerlinv2-core-1 2>/dev/null || true
docker compose up -d
```

---

### Agent stops mid-task with no error

**Cause** (v2 bug, fixed in v3): Previously caused by unhandled `RuntimeError` from `publish()` on NATS disconnect, or KV read failure silently returning.

**v3 protections**:
- `publish()` in `actor.py` retries 3× with 2/4/8s backoff and **never re-raises**
- KV read failure in `handle_step_requested` re-queues the step instead of returning silently
- All of `handle_step_requested` is wrapped in `try/except/finally` that publishes `action.failed`

**If it still happens**: Check `docker logs wzrdmerlinv3-core-1` for `Unhandled error in step` lines.

---

### GPU drops to 0% / model unloads mid-task

**Cause** (v2 bug, fixed in v3): Ollama was unloading the model after 5 min idle, and the old 300s total timeout wasn't per-chunk.

**v3 protections**:
- `keep_alive: -1` (integer) sent on every `/api/chat` request — Ollama never unloads
- `httpx.Timeout(read=120.0)` is per-chunk (not total response)
- NATS reconnect with infinite retry (`max_reconnect_attempts=-1`)

---

### NATS disconnects during a task

**v3 behaviour**: `actor.py` reconnects automatically (infinite retry, 2s between attempts). On reconnect, all NATS subscriptions are re-registered. In-flight tasks will receive the re-queued `STEP_REQUESTED` event (with 2s delay from the KV read error path).

---

### TUI shows "Timed out waiting for events"

**Cause**: No `task.completed` or `task.failed` event arrived within 120 seconds.

**Likely causes**:
- Agent is waiting for human input (should show the `[human?]` prompt now — if not, it's the pre-fix version)
- Task failed silently (check backend logs)
- SSE stream disconnected

---

### `request_human` prompt never appears / TUI hangs

**Status**: Fixed in TUI v3. The `_render_task_events` loop now `break`s immediately when it sees `agent.tool_start{tool: request_human}`, returning control to the main REPL so the `[human?] >` prompt appears.

If you're on an older version of `app.py`, the symptom is: agent asks a question, TUI shows the yellow panel, but the prompt never returns — it times out after 120s.

---

## API Reference

<!-- AUTO-GENERATED: from main.py route decorators -->
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/task?description=` | Submit task or resume human-input-waiting task |
| GET | `/api/task/{task_id}` | Read task status + result from NATS KV |
| POST | `/api/task/{task_id}/cancel` | Cancel in-flight task (stops at next step boundary) |
| GET | `/api/health` | NATS connectivity |
| GET | `/api/llm/health` | Ollama model + circuit breaker status |
| GET | `/api/stream` | SSE event stream (infinite) |
| GET | `/api/events/recent?task_id=&limit=20` | Last N SSE events from in-memory snapshot |
| GET | `/api/logs?lines=50` | Last N lines from in-process log buffer |
| GET | `/api/models` | List model profiles |
| POST | `/api/models/switch?model_name=` | Hot-swap active model |
| POST | `/api/config/reload` | Reload `merlin.config.yaml` |
| GET | `/api/memory/stats` | Episodic + trajectory counts |
| GET | `/api/memory/search?query=` | FTS search memory |
| POST | `/api/memory/prune?max_age_days=` | Delete old entries |
| POST | `/api/memory/reindex` | Rebuild index (no-op for SQLite) |
| POST | `/api/memory/trajectory` | Add teacher trace |
| GET | `/api/gardener/status` | Consolidation status |
| POST | `/api/gardener/run` | Trigger memory consolidation |
| GET | `/api/files?path=` | List workspace directory |
| GET | `/api/files/read?path=` | Read workspace file (max 1MB) |
| GET | `/api/debug/actors` | Actor registry dump |
| POST | `/api/rollback` | Revert last self-improvement merge |
<!-- END AUTO-GENERATED -->

---

## MCP Server (Claude Code integration)

```bash
# Register once
claude mcp add merlin-agent -- python C:/CS/sideProjects/WzrdMerlinV3/tools/merlin_mcp.py

# Available tools in Claude Code after restart:
# get_health, get_llm_health
# submit_task, poll_task, cancel_task
# get_logs, get_task_events
# get_actors, list_models, switch_model, reload_config
# get_memory_stats, search_memory
# list_workspace_files, read_workspace_file
# get_gardener_status, trigger_rollback
```

**Workflow**: `submit_task` returns immediately with a `task_id`. Use `poll_task(task_id)` to check
completion (25s cap per call, safe for MCP client timeouts). Call `poll_task` repeatedly until
status is `completed` or `failed`. Use `cancel_task(task_id)` to abort a stuck task.

Use `get_logs()` to read backend logs without `docker exec`. Use `get_task_events(task_id)` to
see a snapshot of recent SSE events for a task without maintaining a live stream.

---

## Rollback Procedure

```bash
# Via API
curl -X POST http://localhost:8000/api/rollback

# Via TUI
/rollback

# Direct git (inside container)
docker exec -it wzrdmerlinv3-core-1 git log --oneline -5
docker exec -it wzrdmerlinv3-core-1 git revert HEAD
```

---

## Upgrading to ChromaDB Vector Memory

The default memory backend is SQLite FTS5 (no external deps). To upgrade to ChromaDB with Ollama embeddings:

```bash
# 1. Install inside the container
docker exec -it wzrdmerlinv3-core-1 pip install chromadb

# 2. Restart — get_memory() factory auto-detects chromadb import
docker compose restart core

# 3. Verify
curl http://localhost:8000/api/memory/stats
# Should log: "MEMORY: Using ChromaDB vector store"
```

The `nomic-embed-text` model must be pulled in Ollama for embeddings to work:
```bash
ollama pull nomic-embed-text
```

If the embedding model is unavailable, ChromaDB falls back to its built-in default embeddings automatically.

---

## Configuration Reference

`merlin.config.yaml`:

```yaml
active_model: merlin-brain     # which profile is active at startup

models:
  merlin-brain:
    model_name: merlin-model:latest   # Ollama model tag
    context_window: 32768             # tokens; used for context management
    temperature: 0.3
    think: false                      # enable <think> token streaming
    think_budget: 0                   # max think tokens (0 = unlimited)

ollama:
  base_url: http://host.docker.internal:11434
  keep_alive: "-1"       # normalized to int -1 in OllamaClient (never unload)
  read_timeout: 120.0    # per-chunk read timeout in seconds

hardware:
  hsa_override_gfx_version: "10.3.0"   # AMD ROCm GFX override (RX 6750 XT)
  radv_perftest: nogttspill             # prevent GTT spill on Navi 22
```

---

## ⚠️ Not Yet Implemented

These features are partially wired or stubbed but not complete:

### 1. Auditor Agent

**Status**: Wired in `main.py` as `auditor_agent = BaseAgentActor(role="auditor")` and connected to NATS, but the `DisCoRouter` only routes tasks to `agent-implementer`. The auditor never receives tasks.

**What's missing**:
- Routing policy in `router.py` to send some task types to the auditor
- Auditor-specific system prompt and response validation logic
- Audit result integration back into the task lifecycle

---

### 2. CodeModeSandbox JavaScript execution

**Status**: `src/core/mcp/codemode.py` has `execute_javascript()` which shells out to `node`. Node.js is **not installed** in the Docker container (`python:3.11-slim`).

**What's missing**:
- `node` installed in `Dockerfile` (add `nodejs npm` to apt-get)
- Or: swap to a Deno/Bun approach or remove JS execution entirely

---

### 3. Self-improvement patch validation

**Status**: `self_improve.py` generates patches via LLM and runs `pytest` for validation (`_validate_candidate`). This works, but the LLM-generated patches are hit-or-miss with the current Qwen3.5 model — the worktree is set up and torn down correctly.

**What's missing**:
- Feedback loop: failed improvement attempts should reduce re-trigger frequency (the `_is_recently_failed` cooldown exists but uses 24h fixed window)
- Sandbox the patch execution more tightly (currently runs tests in a git worktree, which shares the filesystem)

---

### 4. Forage MCP discovery over authenticated registries

**Status**: `forage.py` can search npm and PyPI and install discovered MCP servers. Works for public packages.

**What's missing**:
- Private registry support (npm auth tokens, pip index URLs)
- Verification step before install (currently installs blindly)
- Rollback if an installed MCP server crashes the agent

---

### 5. Hardware telemetry (VRAM/GPU temp)

**Status**: The TUI `/hw` command shows VRAM and GPU temperature fields, but the watchdog heartbeat only populates `ram_usage`, `cpu_usage`, and `ollama_healthy`. The VRAM/GPU fields are always zero.

**What's missing**:
- `psutil` has no GPU support — need `pynvml` (NVIDIA) or `pyamdgpu`/`rocm-smi` wrapper for AMD
- Or: poll Ollama's `/api/ps` endpoint which reports model memory usage as a proxy for VRAM usage

---

### 6. Context window token counting

**Status**: `OllamaClient._estimate_prompt_tokens()` uses `chars / 4` as a rough heuristic. The TUI context bar is based on this estimate.

**What's missing**:
- Actual tiktoken/tokenizer integration for accurate counts
- Or: use Ollama's `/api/chat` response field `eval_count` / `prompt_eval_count` to get real counts post-generation and update the estimate

---

### 7. Test coverage for self_improve, gardener, router

**Status**: No unit tests for `self_improve.py`, `gardener.py`, or `router.py`. These are complex actors with non-trivial logic.

**What's missing**:
- `tests/test_self_improve.py` — mock git worktree ops, test patch generation flow
- `tests/test_gardener.py` — mock NATS + memory, test consolidation trigger
- `tests/test_router.py` — test task routing decisions + capability gap forwarding

Current coverage is 49 tests covering actor reliability, OllamaClient, memory, and the 5 base_agent fixes only.
