WzrdMerlin v1 — Complete Architecture Dump

  System at a Glance

  A 4-service Docker stack (core, sandbox, postgres+chromadb, dashboard) implementing an autonomous agent OS. LangGraph     
  drives a state machine pipeline, LiteLLM abstracts all model providers, and a React 19 dashboard provides real-time       
  observability. The codebase is ~5,000+ lines of Python (core+sandbox) and ~3,500+ lines of TypeScript/CSS (dashboard).    

  ---
  1. THE ROUTER MECHANISM (the interesting one)

  How It Actually Works

  The router is a stateless dispatch function — all mutable state lives in RateLimitStore (in-memory + Postgres-backed).    
  Before every single model call, the orchestrator asks the router: "I need the implementer role — which model?"

  Router.select_model(role, task) (merlin-core/router/router.py:~60):
  1. Calls _detect_mode() — checks host_idle_minutes from HardwareMonitor
  2. Gets primary model + fallback model from router_config.yaml role mapping
  3. Checks RateLimitStore.is_available(primary, mode) — if quota allows, return primary
  4. Checks fallback — if available, return it (logs downgrade warning)
  5. If both exhausted: returns whichever resets sooner (never blocks)

  Dual-mode logic (_detect_mode()):
  - CONSERVE: host idle < 15 min. Never exceed 80% of any model's hourly quota (configurable conserve_headroom). Protects   
  quota for user's own work.
  - UTILIZE: host idle >= 15 min AND any model has >50% quota remaining. Drains quota fully — unused requests at reset =    
  waste.

  RateLimitStore (merlin-core/router/rate_state.py, 222 lines):
  - Per-model tracking: requests_used, tokens_used, resets_at, mode
  - Auto-resets counters when now >= resets_at (transparent on every get())
  - Rolling token event deque for computing tokens_per_second
  - Tracks last_latency_ms, last_prompt_tokens, last_completion_tokens, context_buffer_pct
  - Persists to Postgres on every write (survives restarts)
  - is_available(model_id, mode): In CONSERVE, caps at requests_per_minute * (1 - conserve_headroom). In UTILIZE, caps at   
  full requests_per_minute.

  HardwareMonitor (merlin-core/monitor/hardware.py, 315 lines) — the mode trigger:
  - Polls every 30s: CPU load, input device activity (platform-specific: macOS ioreg, Linux who)
  - CPU > 20% → host is active
  - Writes host_idle_minutes to shared state
  - Also collects GPU stats (nvidia-smi) and Ollama stats (/api/ps)
  - Has set_overdrive(force) — manual UTILIZE override from dashboard
  - Only component allowed to trigger mode transitions

  Configuration (router_config.yaml):
  roles:
    architect:
      model: "anthropic/claude-sonnet-4-20250514"
      fallback: "ollama/qwen3:32b"
      max_tokens: 16384
      temperature: 1.0
      think: true
      think_budget_tokens: 10000
      think_min_complexity: "medium"
    implementer:
      model: "anthropic/claude-sonnet-4-20250514"
      fallback: "ollama/qwen3:32b"
      # ...
    auditor:
      model: "ollama/qwen3:32b"
      fallback: "ollama/qwen3:32b"
      # ...
    compressor:
      model: "ollama/qwen3:32b"
      # ...
    watchdog:
      model: "ollama/qwen3:32b"
      # ...

  Rate limits are per-model, not per-role. Multiple roles sharing a model share the same quota pool.

  What's working well:
  - Never blocks — always returns something
  - Graceful degradation with fallbacks
  - Live quota telemetry to the dashboard (progress bars, tokens/sec, latency)
  - Mode switching is hardware-driven, not timer-based
  - Entire fleet is swappable via YAML — no code changes

  What's limited in v1:
  - Mode detection is binary (CONSERVE/UTILIZE) — no gradient
  - No predictive scheduling — doesn't pre-assign tasks across upcoming reset windows (spec mentions this but not
  implemented)
  - No cost-aware routing — doesn't consider $/token, only availability
  - Fallback is a single model per role — no ranked fallback chain
  - No cross-role quota awareness — doesn't know if draining Implementer quota will starve Auditor when they share a model  

  ---
  2. SELF-IMPROVEMENT CAPABILITY

  This is the SELF_MODIFY task type — Merlin can modify its own codebase.

  Pipeline Flow for Self-Improvement

  The pipeline has special nodes for this (merlin-core/orchestrator/pipeline.py):

  classify → self_improve_prepare → architect → compress → implement → audit
                                                                        ↓
                                                                sandbox_exec (with interrupt gate)
                                                                        ↓
                                                            self_improve_validate
                                                                        ↓
                                                                    complete

  node_self_improve_prepare (~line 200 in pipeline.py):
  - Creates a git worktree at /workspace/.merlin/candidates/{task_id}
  - Copies dirty/untracked files from the source repo into the worktree
  - Generates a baseline manifest (SHA256 hashes of all tracked files)
  - The Implementer works in this isolated copy, not the live repo

  node_self_improve_validate (~line 700 in pipeline.py):
  - Compares candidate worktree files against the baseline manifest
  - Identifies which files actually changed (by hash comparison)
  - Runs pytest + compileall in the candidate worktree
  - If tests pass: promotes changed files back to the source repo
  - If tests fail: task fails, candidate is discarded
  - Cleans up the worktree either way

  Safety mechanisms:
  - All self-modification is interrupt-gated (requires human approval before sandbox execution)
  - SELF_MODIFY always routes through Architect (even at MEDIUM complexity via needs_architect())
  - Changes are isolated in a git worktree — live code is never touched until validation passes
  - Baseline manifest ensures only intentional changes are promoted
  - compileall catches syntax errors before promotion
  - pytest catches regressions

  Routing for after_classify: If task_type == SELF_MODIFY → routes to self_improve_prepare first. After sandbox → routes to 
  self_improve_validate instead of watchdog.

  What's implemented:
  - Worktree isolation and baseline comparison
  - Automated test + compile validation
  - File promotion on success
  - Interrupt gate for human approval

  What's NOT implemented (or thin):
  - No rollback mechanism if promoted changes break something discovered later
  - No multi-step self-improvement (can't chain modifications)
  - No self-generated improvement tasks — Merlin doesn't identify its own weaknesses
  - No A/B comparison of candidate vs current behavior
  - No confidence scoring on the modifications
  - The validation is basic (pytest + compileall) — no behavioral testing against the agent's own capabilities

  ---
  3. EXECUTION PIPELINE (ReAct Loop)

  The Implementer runs in a ReAct loop (node_implement in pipeline.py, ~line 400):

  1. Call Implementer with task + context + past tool results
  2. Parse JSON action array from response (fuzzy: handles code fences, prose, duplicates)
  3. Deduplicate actions (JSON stringification comparison)
  4. Detect stalling (only read/list actions for 2+ iterations → inject nudge)
  5. Execute each action via sandbox HTTP call
  6. Feed results back as conversation history
  7. Repeat up to react_max_iterations (default 10)
  8. Stop on done action type or max iterations

  Action types: shell, write_file, read_file, patch_file, list_dir, python, done

  Exception for interrupt-gated tasks (high complexity, self_modify): single-shot generation instead of ReAct loop. The full   action list is generated once, then reviewed by Auditor before execution.

  ---
  4. MEMORY SYSTEM

  Episodic Memory (merlin-core/memory/episodic.py, ChromaDB):
  - Every Auditor rejection → embedded and stored with task context, critique, and eventual fix
  - Before each Implementer call → query for similar past failures (cosine similarity > 0.6)
  - Relevant failures prepended to prompt context
  - Auto-prune: records older than 90 days or not referenced in 30 days
  - This is the core self-improvement feedback loop — prevents repeating mistakes

  Thread State (merlin-core/memory/thread_state.py, Postgres):
  - Full task CRUD with JSONB for context/result fields
  - LangGraph checkpoints every pipeline step transition
  - Crash recovery: reads latest checkpoint and continues from last confirmed step
  - Task queries: pending (by priority), suspended, all active, recent terminal

  ---
  5. INTERRUPT SYSTEM

  InterruptHandler (merlin-core/interrupt/handler.py, 234 lines):
  - 5 types: SEND_MESSAGE, FINANCIAL, EXTERNAL_COMMIT, DESTRUCTIVE, CREDENTIAL
  - Non-blocking: task suspends → checkpoints to Postgres → next task runs immediately
  - Timeout loop (every 30s): expires stale interrupts based on config
  - FINANCIAL and DESTRUCTIVE never auto-expire — always require explicit user response
  - On approval: task moves from SUSPENDED → PENDING, resumes from checkpoint
  - On rejection: task → CANCELLED

  ---
  6. DASHBOARD (React 19 + Vite)

  Replaced the old Streamlit app with a modern SPA:

  - 3-column layout: Sidebar (hardware/quotas/interrupts) | Chat | Tasks
  - Streaming chat: SSE to /chat/stream, intent detection (chat vs task)
  - Pipeline visualization: PipelineStrip component with 9 stages, color-coded status
  - Terminal component: Smart event merging, collapsible output blocks, thinking blocks, audit results
  - Config page: Full system configuration editor — model fleet, rate limits, timeouts, sandbox settings
  - Polling: 2-5s intervals for hardware, quotas, tasks, interrupts
  - Container controls: Rebuild/restart services from the UI

  ---
  7. KEY DESIGN DECISIONS & TRADE-OFFS

  ┌─────────────────────────┬───────────────────────────────────────┬───────────────────────────────────────────────────┐   
  │        Decision         │                  Why                  │                    Limitation                     │   
  ├─────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────┤   
  │ LiteLLM for all model   │ Single interface for cloud + local    │ Adds a dependency layer, thinking param handling  │   
  │ calls                   │                                       │ is model-specific                                 │   
  ├─────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────┤   
  │ LangGraph state machine │ Checkpoint/resume for free, interrupt │ Complex pipeline definition (~1000 lines),        │   
  │                         │  handling built-in                    │ debugging state transitions is hard               │   
  ├─────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────┤   
  │ Classifier runs on      │ No cloud cost for classification      │ Hard dependency on Ollama — if Qwen is down,      │   
  │ local Qwen              │                                       │ nothing works                                     │   
  ├─────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────┤   
  │ Single fallback per     │ Simple config                         │ No ranked fallback chain, can't express "try 3    │   
  │ role                    │                                       │ models before giving up"                          │   
  ├─────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────┤   
  │ ReAct loop in           │ Flexible tool use                     │ Action parsing is fragile (regex + JSON           │   
  │ pipeline.py             │                                       │ extraction), stall detection is heuristic         │   
  ├─────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────┤   
  │ Git worktree for        │ Safe isolation                        │ No multi-step modifications, no rollback after    │   
  │ self-modify             │                                       │ promotion                                         │   
  ├─────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────┤   
  │ Polling-based dashboard │ Simple, works everywhere              │ Not truly real-time, 2-5s latency on updates      │   
  ├─────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────┤   
  │ In-memory event buffer  │ Fast, simple                          │ Events lost on restart, no persistence            │   
  │ (500 max)               │                                       │                                                   │   
  └─────────────────────────┴───────────────────────────────────────┴───────────────────────────────────────────────────┘   

  ---
  8. FILE MAP (key files only)

  merlin-core/
    main.py              (444 lines) — FastAPI app, lifespan, all HTTP endpoints
    events.py            (53 lines)  — In-memory event buffer
    orchestrator/
      orchestrator.py    (500 lines) — Central loop, task execution, control API
      pipeline.py        (1013 lines)— LangGraph graph, all nodes, ReAct loop, self-improve
      state.py           (47 lines)  — PipelineState TypedDict
    router/
      router.py          (179 lines) — select_model(), _detect_mode(), quota_summary()
      rate_state.py      (222 lines) — RateLimitStore, quota tracking, persistence
      config.py          (115 lines) — RouterConfig Pydantic model, YAML loader
    classifier/
      classifier.py      (264 lines) — Task classification via local Qwen
    roles/
      base.py            (179 lines) — BaseRole with streaming LiteLLM calls
      implementer.py     (133 lines) — ReAct action format, tool results handling
      architect.py       (73 lines)  — Blueprint generation and reframing
      auditor.py         (112 lines) — Adversarial review, PASS/FAIL verdict
      compressor.py      (103 lines) — Token counting, context reduction
      watchdog.py        (167 lines) — Output monitoring, syntax/test running
    models/task.py       (135 lines) — Task, Interrupt, enums, ModelRateState
    memory/
      episodic.py        (189 lines) — ChromaDB failure memory
      thread_state.py    (201 lines) — Postgres task persistence
    interrupt/handler.py (234 lines) — Interrupt lifecycle, timeout loop
    monitor/hardware.py  (315 lines) — CPU/GPU/input polling, idle detection
    scheduler/scheduler.py (147 lines)— Cron-based task firing

  merlin-sandbox/
    executor.py          (331 lines) — Action dispatch, path safety, payload parsing
    schemas.py           (113 lines) — Pydantic action/request/response models
    main.py              (79 lines)  — FastAPI bridge

  merlin-dashboard/src/
    App.tsx, main.tsx     — React Router setup
    types.ts             — All TypeScript interfaces
    api.ts               — Fetch helpers + SSE streaming
    hooks.ts             — usePolling, useTaskEvents
    pages/Dashboard.tsx  — 3-column layout
    pages/Config.tsx     — Full config editor
    components/          — Sidebar, Chat, TasksPanel, ActivityLog, Terminal, PipelineStrip

  Root:
    router_config.yaml   — The single source of truth for fleet config
    docker-compose.yml   — 5-service stack
    tasks.yaml           — 4 scheduled recurring tasks

  ---
  9. FOR V2 — What Matters Most

  Router mechanism strengths to preserve:
  - Role-to-model indirection (swap anything via YAML)
  - Never-block dispatch with fallback
  - Hardware-driven mode switching
  - Live quota telemetry

  Router gaps to address in v2:
  - Predictive scheduling across reset windows (spec'd but not built)
  - Cost-aware routing ($/token consideration)
  - Ranked fallback chains (not just one fallback)
  - Cross-role quota awareness when models are shared
  - Gradient modes instead of binary CONSERVE/UTILIZE

  Self-improvement strengths to preserve:
  - Episodic memory loop (failures → context → prevention)
  - Git worktree isolation for self-modification
  - Automated validation (test + compile) before promotion

  Self-improvement gaps to address in v2:
  - No self-generated improvement tasks (Merlin doesn't identify its own weaknesses)
  - No behavioral regression testing (only pytest/compileall)
  - No rollback after promotion
  - No confidence scoring on modifications
  - No A/B evaluation of candidate changes
  - No chained multi-step self-improvement
  - Episodic memory only stores failures, not successes or patterns

  That's everything you've built. The bones are solid — the pipeline, interrupt system, and role abstraction are
  well-designed. The two areas you're targeting (router + self-improvement) are exactly where the most room for growth      
  Exists.


This is my project spec right now. I want to make a complete remake of version 2. The main features I want to focus on are the router, the self-improving agent, and efficiency of local source models. Here is what I learned from the last version
One of my main goals is to be able to use local models, like Qwen 3.5:9b as I want to avoid any API costs. This means that this agent and its structure must be very efficient, in terms of speed, token overhead, context engineering/injection, etc. I basically want to make this agent focused on the average person who doesnt have $600 to drop on a mac mini when they already have a good system
Another is that I want the self-improvement capability to be very emergent. What I mean by that is I want to give the most base agent (think a newly hatched openclaw) the minimum skills it needs to accomplish any goal that the user can think of, within hardware and ethical guidelines. For example,  I should be able to tell the base agent to modify itself so it can use the playwright mcp, it should iterate continuously, on a smart scheduling based system, until it does the job
Previously, I had alot of issues with the dashboard and getting it to work with the features I want. I believe it was because of my framework, or maybe because I hosted it on local host instead of making an application, thats up to you to figure out. I think I will use figma to design the next UI
Another is that I believe the Langgraph approach is too restricted. I basically want the agent to iterate exactly how an enterprise agent would do. Step by step, tool call by tool call, narrating everything. Think of like how the github copilot vscode extension interacts with the user, I want that type of computation. I think the stateful architecture is a good design choice though
The agents themselves, orchestrator, watchdog, etc, dont seem right. Look into that
Here of some things that just popped up but seem related. Context engineering I feel is lacking, same with context distillation and automatic context clearing and handoff.  Based on all of this, help me refine for my version 2

SYSTEM CONTEXT: WzrdMerlin v2 Autonomous Agent OS
1. Project Overview & Architecture Philosophy
WzrdMerlin v2 is a highly efficient, event-driven multi-agent operating system utilizing a Distributed Cognition (DisCo) model. It abandons sequential, blocking graphs (like LangGraph) in favor of asynchronous, non-blocking actor networks. The entire architecture is strictly bound and optimized for consumer-grade hardware limits.
2. Target Hardware & Environmental Constraints
GPU: AMD Radeon RX 6750 XT (12GB VRAM limit).
CPU / System RAM: Intel i7-12700K / 48GB.
Driver Spoofing: The RX 6750 XT uses the Navi 22 / gfx1031 architecture, which lacks official pre-compiled ROCm support. The system must export HSA_OVERRIDE_GFX_VERSION=10.3.0 to force the ROCm runtime to utilize gfx1030 binaries.
3. Inference Engine (llama.cpp)
Foundation: Utilize llama.cpp over high-throughput engines like vLLM to maintain granular control over memory allocation and avoid PagedAttention VRAM bloat.
Dual-API Disaggregation Strategy:
Prefill Phase (Compute-Bound): Route to the ROCm 7.x backend to process massive context histories utilizing rocWMMA/rocBLAS.
Decode Phase (Memory-Bound): Route to the Vulkan backend to maximize token-per-second generation speeds via memory bandwidth optimization.
Memory Management & Spilling:
Model Weights: Load Qwen 3.5 9B using Q5_K_M GGUF quantization (~6.5GB VRAM).
KV Cache: Quantize context memory to Q8_0 or Q4_K to reduce footprint by 50%.
GTT Spilling: Leverage Linux AMDGPU Graphics Translation Table (GTT) spilling mechanics to purposefully overflow the secondary KV cache into the abundant 48GB System RAM over PCIe when the 12GB VRAM limit is hit (avoid spilling model weights).
Predictive Resource Scheduler:
Enforce a "Max Active Backends" limit with Least Recently Used (LRU) eviction.
Idle Watchdog: Unload idle models after 15 minutes.
Busy Watchdog: Terminate hung processes exceeding 10 minutes.
4. DisCo Orchestration & Event Mesh
Event Mesh: Use NATS JetStream (or Redis Streams) as the high-velocity data backbone. Requires durable message storage and exactly-once delivery semantics to survive transient agent crashes.
Actor Model (XState v5): Every agent is an encapsulated virtual state machine.
Agents must utilize durable execution. During long tool I/O waits, capture the state via getPersistedSnapshot(), serialize it to disk, and sleep the agent process to free system threads. Hydrate and resume upon event arrival.
Narrated Computation Loop & Firewall: Agent output MUST adhere to a 4-block structure:
Thinking Block: Wrapped in <think> and </think> tags. CRITICAL: The OS parser must be strictly firewalled to ignore any string inside these tags to prevent phantom tool calls and recursive execution loops.
Narrative Block: User-facing Copilot-style explanation.
Action Block: Validated JSON tool execution.
Observation Block: Environment return data.
5. Extensibility: Model Context Protocol (MCP) & Forage
Universal Interface: All tool, data, and API interactions must route through MCP servers.
Self-Improvement Pipeline (isaac-levine/forage):
When an agent encounters a capability gap, it invokes forage_search to query the Official MCP Registry, Smithery, or npm.
Uses forage_install to spawn the newly discovered MCP server as a local proxy subprocess (requires user authorization via confirm: true).
Uses forage_learn to write tool schemas and usage rules to persistent .merlin/rules/ files, permanently expanding the context capability without restarting the OS.
Code Mode Sandbox: To prevent token exhaustion on massive data (e.g., 10,000-row CSVs), agents must generate Python/TypeScript scripts to process the data locally inside an MCP sandbox, returning only the distilled summary.
6. Context Engineering Topologies
Observation Masking (JetBrains Strategy): Maintain a strict rolling window of the last $M=10$ interaction turns. For turns falling outside this window, strip the verbose environment observation and replace it with a static placeholder (e.g., [Output Omitted for brevity]). Full reasoning and action histories must be retained.
Context Folding (Phase Handoff): Triggered when transitioning task phases (e.g., Data Gathering $\rightarrow$ Code Generation).
Permanently evict all transient tool logs.
Swap phase-specific MCP tools.
Compress findings into dense semantic "Artifacts" (~2KB) to carry forward.
Write a chronological journal entry to a durable audit log hidden from the active prompt.
The Gardener (Background Consolidation): An asynchronous agent that runs during host idle time. It reads raw chat logs from a temporary buffer, extracts "Atomic Facts", builds bi-directional links in a local vector graph, and generates a highly compressed Summary Layer for future semantic retrieval.
In-Context Distillation: For highly complex tasks, dynamically retrieve top-$k$ successful reasoning traces generated by a higher-tier "Teacher" model (e.g., Claude 3.5 Sonnet) from a local vector database and inject them as few-shot examples into the 9B model's context.
7. UI/UX: Native Agent Workspace
Framework: Tauri v2 + React 19.
VRAM Preservation: Absolutely no bundled Chromium engines (Electron). Must utilize the OS native webview (WKWebView / Edge WebView2 / WebKitGTK) to reduce UI memory footprint to ~50MB-125MB, freeing critical VRAM for the inference engine.
Rust Sidecars: Orchestration layer, hardware sensor polling (temperature, VRAM saturation), and zombie-process cleanup are handled by a low-latency Rust backend.
Hybrid Streaming:
WebSockets: Bidirectional real-time chat.
Server-Sent Events (SSE): Unidirectional high-throughput log and telemetry streaming.
8. File & Directory Structure
merlin-v2/
├── src-tauri/ # Rust sidecar: hardware telemetry, native OS hooks, cleanup
├── merlin-core/ # DisCo Logic (Python/Rust)
│ ├── actors/ # XState v5 durable state machines
│ ├── bus/ # NATS JetStream integration
│ ├── inference/ # llama.cpp Vulkan/ROCm dual-API wrapper & GTT configs
│ └── mcp/ # Dynamic MCP Client, Forage integration, Code Mode Sandbox
├── merlin-dashboard/ # React 19 Frontend (Figma-exported components)
│ ├── components/ # Narrative terminal, VRAM/GTT monitors, Trace Inspector
│ └── hooks/ # SSE & WebSocket listeners
└──.merlin/ # Persistent state, teacher vector DB, Forage tool rules

