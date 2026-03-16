# WzrdMerlin v3

A local-first autonomous AI agent OS built on NATS pub/sub, Ollama, and FastAPI.

Merlin runs multi-step ReAct agent loops with tool use, episodic memory, self-improvement via git worktrees, and a terminal UI — all on your own hardware with no cloud dependencies.

## Architecture

```
┌─────────────┐    ┌───────────┐    ┌──────────────┐
│  TUI / API  │───▶│  FastAPI   │───▶│    NATS      │
│ (Rich REPL) │    │  :8000     │    │  JetStream   │
└─────────────┘    └───────────┘    └──────┬───────┘
                                           │ pub/sub
                   ┌───────────────────────┼───────────────────┐
                   │                       │                   │
            ┌──────▼──────┐  ┌─────────────▼──┐  ┌────────────▼───┐
            │ DisCoRouter │  │ BaseAgentActor  │  │  WatchdogActor │
            │ (task route)│  │ (ReAct loop)    │  │  (heartbeat)   │
            └─────────────┘  └────────┬────────┘  └────────────────┘
                                      │
                          ┌───────────┼───────────┐
                          │           │           │
                    ┌─────▼───┐ ┌─────▼───┐ ┌────▼──────┐
                    │ Ollama  │ │ SQLite  │ │   Tools   │
                    │ Client  │ │ FTS5    │ │ (MCP,code)│
                    └─────────┘ └─────────┘ └───────────┘
```

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Container uses 3.11-slim |
| Docker Desktop | Latest | WSL2 backend on Windows |
| Ollama | Latest | Running on host at `localhost:11434` |

**Hardware tested on**: AMD RX 6750 XT (12GB, ROCm), i7-12700K, running Qwen3.5-9B.

## Quick Start

```bash
# 1. Clone and start the stack
git clone git@github.com:hetyug04/WzrdMerlin.git
cd WzrdMerlin
docker compose up -d

# 2. Verify
curl http://localhost:8000/api/health
curl http://localhost:8000/api/llm/health

# 3. Launch the TUI
pip install -r requirements.txt
python -m src.tui

# 4. Submit a task
curl -X POST "http://localhost:8000/api/task?description=hello+world"
```

## Key Features

- **Reliable pub/sub**: NATS JetStream with auto-reconnect, retry-on-publish, and circuit breaker on Ollama
- **ReAct agent loop**: Multi-step reasoning with parallel tool execution and per-tool timeouts
- **Episodic memory**: SQLite FTS5 (zero dependencies), optional ChromaDB upgrade for vector search
- **Self-improvement**: LLM generates code patches, validates with pytest in git worktrees, auto-merges on pass
- **Memory gardener**: Background consolidation of episodic memories into reusable knowledge
- **MCP integration**: Discover and connect to MCP tool servers (npm/PyPI), plus a built-in MCP server for Claude Code
- **Terminal UI**: Rich + prompt_toolkit REPL with live SSE streaming

## Project Structure

```
src/
├── core/
│   ├── actor.py           # NATS actor base: connect, publish-retry, reconnect
│   ├── base_agent.py      # ReAct loop, tools, context folding
│   ├── ollama_client.py   # Ollama HTTP client + circuit breaker
│   ├── config.py          # Model profiles, Ollama settings
│   ├── memory.py          # SQLite FTS5 memory store
│   ├── events.py          # Event model + EventType enum
│   ├── router.py          # DisCoRouter: task routing
│   ├── state.py           # NATS KV wrapper
│   ├── watchdog.py        # System heartbeat + iteration guard
│   ├── gardener.py        # Memory consolidation
│   ├── self_improve.py    # Git worktree patch generation
│   ├── main.py            # FastAPI app + lifespan
│   └── mcp/               # MCP server management + tools
├── tui/
│   └── app.py             # Terminal UI
tools/
└── merlin_mcp.py          # MCP server for Claude Code integration
tests/                     # pytest unit tests (no Docker required)
```

## Configuration

Edit `merlin.config.yaml` to configure model profiles and Ollama settings:

```yaml
active_model: merlin-brain

models:
  merlin-brain:
    model_name: merlin-model:latest
    context_window: 32768
    temperature: 0.3
    think: false
    think_budget: 0

ollama:
  base_url: http://host.docker.internal:11434
  keep_alive: "-1"
  read_timeout: 120.0
```

Hot-swap models at runtime:
```bash
curl -X POST "http://localhost:8000/api/models/switch?model_name=qwen3.5-14b"
```

## Claude Code Integration

Register the MCP server to use Merlin as a tool from Claude Code:

```bash
claude mcp add merlin-agent -- python tools/merlin_mcp.py
```

## Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Documentation

- [Contributing Guide](docs/CONTRIBUTING.md) — setup, code style, adding tools/models
- [Operations Runbook](docs/RUNBOOK.md) — health checks, troubleshooting, API reference

## License

[MIT](LICENSE)
