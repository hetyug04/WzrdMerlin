# Contributing to WzrdMerlin v3

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Container uses 3.11-slim; host dev uses 3.13 |
| Docker Desktop | Latest | With WSL2 backend on Windows |
| Ollama | Latest | Running on Windows host at `localhost:11434` |
| NATS | via Docker | Managed by `docker compose` |

## Local Development Setup

```bash
# 1. Install Python deps on host (for running tests)
pip install -r requirements.txt

# 2. Start the stack
docker compose up -d

# 3. Verify everything is healthy
curl http://localhost:8000/api/health
curl http://localhost:8000/api/llm/health

# 4. Run the TUI
python -m src.tui
```

## Running Tests

All tests are pure unit tests — no Docker or NATS required.

```bash
python -m pytest tests/ -v
```

<!-- AUTO-GENERATED: test matrix -->
| Suite | File | Tests | What it covers |
|-------|------|-------|----------------|
| Actor reliability | `tests/test_actor.py` | 8 | publish no-raise, retry, reconnect re-subscribe |
| OllamaClient | `tests/test_ollama_client.py` | 20 | CircuitBreaker states, 6-strategy JSON parser |
| Memory | `tests/test_memory.py` | 12 | SQLite FTS5 add/search/prune, ChromaDB fallback |
| Agent lifecycle | `tests/test_agent_lifecycle.py` | 9 | 5 base_agent fixes: safe_publish, KV re-queue, tool timeout, resume guard |
<!-- END AUTO-GENERATED -->

**Coverage target**: 80%+. Run with `pytest --cov=src tests/`.

## Project Structure

```
WzrdMerlinV3/
├── src/
│   ├── core/
│   │   ├── actor.py          — BaseActor: NATS connect, publish-retry, reconnect callbacks
│   │   ├── base_agent.py     — BaseAgentActor: ReAct loop, tools, context folding
│   │   ├── ollama_client.py  — OllamaClient + CircuitBreaker (replaces v2 llm.py + inference.py)
│   │   ├── config.py         — MerlinConfig: model profiles, Ollama settings
│   │   ├── memory.py         — SQLiteMemoryStore (FTS5 primary)
│   │   ├── memory_chroma.py  — EpisodicMemory (ChromaDB optional upgrade)
│   │   ├── events.py         — Event model + EventType enum (stable, do not modify)
│   │   ├── router.py         — DisCoRouter: task routing + capability gap forwarding
│   │   ├── state.py          — NATS KV wrapper (MERLIN_STATE bucket)
│   │   ├── watchdog.py       — System heartbeat + max-iteration guard
│   │   ├── gardener.py       — Memory consolidation (idle-triggered)
│   │   ├── self_improve.py   — ImprovementManager: git-worktree patch generation
│   │   ├── main.py           — FastAPI app + lifespan wiring
│   │   └── mcp/
│   │       ├── manager.py    — MCPManager: stdio MCP server connections
│   │       ├── forage.py     — ForageManager: npm/PyPI MCP server discovery
│   │       └── codemode.py   — CodeModeSandbox: Python/JS subprocess execution
│   └── tui/
│       └── app.py            — Rich + prompt_toolkit REPL
├── tools/
│   └── merlin_mcp.py         — MCP server exposing Merlin REST API to Claude Code
├── tests/                    — pytest unit tests (no Docker required)
├── docs/
│   ├── CONTRIBUTING.md       — this file
│   └── RUNBOOK.md            — operations guide
├── merlin.config.yaml        — model profiles + Ollama settings
├── docker-compose.yml        — nats + core services
└── requirements.txt          — Python dependencies
```

## Environment Variables

<!-- AUTO-GENERATED: from CLAUDE.md env table + docker-compose.yml -->
| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `NATS_URL` | `nats://nats:4222` | Yes (in Docker) | NATS JetStream connection string |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Yes | Ollama API endpoint |
| `MERLIN_WORKSPACE` | `/workspace` | No | Persistent volume root (memory DB, rollback points) |
| `MERLIN_MODEL` | *(from config)* | No | Override active model profile at startup |
| `MERLIN_EMBED_MODEL` | `nomic-embed-text` | No | Embedding model (only used with ChromaDB) |
| `MERLIN_STEP_TIMEOUT_SECONDS` | `90` | No | Per-step LLM generation hard timeout |
| `MERLIN_TOOL_TIMEOUT_SECONDS` | `60` | No | Per-tool execution timeout (parallel tools) |
| `MAX_ITERATIONS` | `20` | No | Watchdog max ReAct iterations per task |
| `MERLIN_API` | `http://localhost:8000` | No | MCP server target URL (tools/merlin_mcp.py only) |
| `MERLIN_TASK_TIMEOUT` | `300` | No | Max seconds to wait for task in MCP server |
<!-- END AUTO-GENERATED -->

## Adding a New Tool to the Agent

1. Add a handler method to `BaseAgentActor` in `src/core/base_agent.py`:
   ```python
   async def tool_my_tool(self, args: Dict[str, Any]) -> str:
       ...
       return result_string
   ```
2. Register it in `__init__` under `self.tools`:
   ```python
   "my_tool": self.tool_my_tool,
   ```
3. Add the tool signature to the system prompt in `_generate_single_action`.
4. Write a test in `tests/test_agent_lifecycle.py`.

## Adding a New Model Profile

Edit `merlin.config.yaml`:
```yaml
models:
  my-model:
    model_name: my-model:latest
    context_window: 16384
    temperature: 0.3
    think: false
    think_budget: 0
```
Then hot-swap: `curl -X POST "http://localhost:8000/api/models/switch?model_name=my-model"`

## Code Style

- Python 3.11+ features only (matching the container)
- No type: ignore comments — fix the types
- All async I/O — no blocking calls in async contexts
- `logger.error` for unexpected failures, `logger.warning` for expected degraded states
- Do not catch bare `except Exception` except at actor event-handler boundaries

## PR Checklist

- [ ] `pytest tests/ -v` passes (49 tests, 0 failures)
- [ ] `python -m py_compile src/core/*.py src/tui/app.py tools/merlin_mcp.py` clean
- [ ] No new hard dependencies added to `requirements.txt` without discussion
- [ ] New tools documented in system prompt inside `base_agent.py`
- [ ] RUNBOOK updated if operational behavior changed
