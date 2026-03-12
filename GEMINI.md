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

Architectural Specification and Viability Analysis for WzrdMerlin v2 Autonomous Agent OSIntroduction to the Distributed Cognition Operating SystemThe transition from monolithic, sequential artificial intelligence pipelines to asynchronous, multi-agent operating systems represents a critical evolution in the field of local machine learning inference architecture. The specification provided for WzrdMerlin v2 outlines an "Autonomous Agent OS" meticulously engineered for consumer-grade hardware. Specifically, the architecture is designed around a strictly constrained hardware topology: a 12GB VRAM Graphics Processing Unit (the AMD Radeon RX 6750 XT), coupled with an expansive 48GB of System RAM, and driven by an Intel i7-12700K processor. This precise hardware split dictates every systemic design choice within the v2 architecture. By shifting from rigid, blocking frameworks to an event-driven Distributed Cognition (DisCo) model, the system is engineered to maximize utilization across heterogeneous compute resources while aggressively preserving volatile memory.This comprehensive architectural report evaluates the proposed WzrdMerlin v2 specification, validating its technical viability against current 2026 software stacks, GPU drivers, and advanced machine learning paradigms. The analysis explores the intricacies of dual-API AMD inference, the implementation of durable actor models, dynamic tool discovery protocols, state-of-the-art context engineering, and memory-efficient native webview interfaces. The synthesis of this evidence indicates that the v2 specification represents a highly optimized, production-ready framework for local autonomous agents, provided that specific environmental overrides, driver configurations, and memory management heuristics are strictly enforced.Hardware-Aware Inference Engine: RDNA2 and Dual-API ComputeAt the foundation of the WzrdMerlin v2 specification is a highly customized inference engine designed to bypass the traditional limitations of running modern large language models on mid-range AMD hardware. The strategic choice to utilize llama.cpp over higher-level, throughput-oriented serving engines like vLLM is structurally sound for this specific single-user, memory-constrained environment. Throughput engines are designed to maximize concurrent request batching in data center environments, which inevitably incurs massive VRAM overhead due to mechanisms like PagedAttention. Conversely, llama.cpp is optimized for low-latency, single-batch inference and provides granular control over memory allocation, making it the superior foundation for a local agentic operating system.Overcoming RDNA2 Limitations via Environmental SpoofingThe AMD Radeon RX 6750 XT utilizes the Navi 22 GPU architecture, which is identified within the LLVM compiler infrastructure by the target designation gfx1031. Officially, the modern ROCm 7.x software stack deprecates or omits direct, pre-compiled support for gfx1031, focusing its enterprise support instead on the gfx1030 target (which covers the RX 6800 and 6900 series), as well as newer RDNA3 and CDNA datacenter architectures. This lack of official support traditionally forces developers into complex, error-prone manual recompilation of PyTorch and ROCm libraries.However, because the fundamental Instruction Set Architectures (ISAs) for the gfx1030 and gfx1031 hardware are virtually identical at the silicon level, the v2 specification correctly deploys the HSA_OVERRIDE_GFX_VERSION=10.3.0 environment variable. This highly specific spoofing mechanism forces the ROCm runtime to treat the RX 6750 XT as if it were an RX 6800. By doing so, the system seamlessly loads the pre-built gfx1030 code objects. This allows the RX 6750 XT to execute heavily optimized, hardware-accelerated machine learning workloads without modifying the underlying ROCm binaries, ensuring compatibility with the latest llama.cpp builds and maintaining system stability.The Dual-API Strategy: Disaggregated Prefill and DecodePerhaps the most sophisticated and highly specialized aspect of the WzrdMerlin v2 inference layer is the implementation of a dual-API strategy. The specification proposes utilizing the ROCm 7.x backend for prompt processing (the prefill phase) and the Vulkan backend for token generation (the decode phase). This hybrid approach directly addresses the architectural performance characteristics of the RDNA2 compute units.Large language model inference is fundamentally divided into two distinct computational phases, each stressing the hardware in entirely different ways:The Prefill Phase (Compute-Bound): When a user submits a prompt, or an agent retrieves its context history, the system must compute the key-value (KV) cache for the entire input sequence simultaneously. This phase requires massive parallel matrix multiplications and is intensely compute-bound. The ROCm backend, utilizing AMD's highly optimized rocWMMA and rocBLAS libraries, excels at this task, processing large batches of prompt tokens significantly faster than alternative APIs.The Decode Phase (Memory-Bandwidth Bound): Once the prompt is ingested, the model begins generating subsequent tokens one by one in an autoregressive manner. For every single token generated, the entire multi-gigabyte weight matrix of the model must be loaded from the physical VRAM into the compute units. This phase is entirely constrained by memory bandwidth rather than raw computational teraflops.Recent 2026 benchmarking data confirms that the Vulkan backend—specifically when utilizing the open-source RADV driver within the Linux Mesa stack—has achieved remarkable optimizations for memory-bound machine learning workloads. Empirical testing demonstrates that Vulkan can outperform ROCm by up to 50% in token generation speeds on RDNA2 and RDNA3 hardware. Conversely, Vulkan suffers severe performance degradation during the compute-heavy prefill phase, often taking significantly longer than ROCm to process the initial prompt.By adopting a disaggregated prefill and decode architecture, the WzrdMerlin v2 operating system optimally routes the workflow. The compute-bound prefill phase is executed via ROCm to ensure rapid ingestion of the agent's massive context history, while the memory-bound decode phase is seamlessly handed over to Vulkan to achieve maximum token-per-second generation speeds. This dual-API execution split represents a cutting-edge adaptation of the disaggregated architectures currently being deployed in multi-node enterprise serving engines, successfully scaled down for a heterogeneous single-machine environment.GTT Memory Spilling and KV Cache QuantizationThe hard physical limit of 12GB of VRAM on the target hardware necessitates an aggressive, multi-layered approach to memory management. The specification mandates the use of a 9-billion parameter reasoning model, specifically Alibaba Cloud's Qwen 3.5 9B (released in early 2026). To fit this foundational model into the available hardware envelope, the system employs the Q5_K_M GGUF quantization format. At this precision level, the model weights consume approximately 6.5GB of VRAM, preserving the intricate reasoning capabilities and multilingual proficiencies of the base model while liberating crucial memory space.This leaves roughly 5.5GB of VRAM available to host the operating system's display server, the UI framework, and the critical KV cache. To support the massive context windows required for autonomous agent operations (which can scale up to Qwen 3.5's native 262,144 token limit), the architecture relies on "GTT (Graphics Translation Table) spilling".Model ComponentPrecision FormatEstimated VRAM FootprintDescriptionQwen 3.5 9B WeightsQ5_K_M~6.5 GBBalanced integer quantization preserving core reasoning fidelity.Operating System UIDynamic~0.5 GBHeadless or native webview rendering overhead.Primary KV CacheQ8_0 / Q4_K~5.0 GBQuantized context memory pinned to the primary GPU.Spilled KV CacheQ8_0 / Q4_KUp to 48.0 GBOverflow context memory spilled to the i7-12700K System RAM.Table 1: Memory allocation hierarchy for the WzrdMerlin v2 12GB VRAM target architecture.In Linux environments utilizing the AMDGPU driver, when physical VRAM is exhausted, the memory management unit can implicitly spill memory allocations into the host system RAM over the PCIe bus. If model weights are spilled, token generation speeds collapse to unusable levels due to PCIe bottlenecking. However, this spilling mechanism is highly effective for storing the secondary KV cache in the abundant 48GB of system RAM.A critical optimization parameter noted in modern graphics driver stacks is the RADV_PERFTEST=nogttspill environment variable, introduced in Mesa 25.2.0. This flag gives the system explicit control over preventing catastrophic, uncontrolled driver-level spilling during Vulkan compute operations. By utilizing llama.cpp, which has explicit host-memory caching parameters, the OS can gracefully overflow the KV cache into system memory without crashing the graphics driver, an operation that strict VRAM allocators like vLLM generally fail to handle on consumer hardware.Furthermore, the specification mandates quantizing the KV cache itself to 8-bit (Q8_0) or 4-bit (Q4_K) precision. This action slashes the context memory footprint by 50% to 75%. While aggressive KV quantization can theoretically result in a minor degradation of perplexity, empirical benchmarking data suggests that for agentic workflows involving structured code generation and JSON tool selection, the quality loss is statistically negligible. The operational capability gained by functionally doubling the effective context length within the same physical memory constraints far outweighs the marginal precision loss.Predictive Resource Scheduler and Watchdog ConstraintsTo effectively manage these tightly packed and highly contested memory allocations, the WzrdMerlin v2 architecture introduces a Predictive Resource Scheduler driven by strict eviction heuristics. Advanced AI agents frequently require specialized auxiliary models to complete complex tasks, such as a localized embedding model for vector search or a lightweight vision encoder for parsing user interface elements.The scheduler establishes a "Max Active Backends" threshold. When this threshold is breached by a request to load a new model, the system enforces a Least Recently Used (LRU) eviction strategy, purging the oldest idle model from VRAM to make room for the new requirement.Furthermore, the implementation of dual watchdog mechanisms guarantees system resilience. The Idle Watchdog acts as a garbage collector, automatically unloading any model from VRAM after 15 minutes of inactivity, returning those resources to the host operating system. The Busy Watchdog serves as a fail-safe, aggressively terminating any hung inference processes that exceed a 10-minute compute timeout. This ensures that the system does not silently leak memory or permanently lock up the AMD GPU during unattended autonomous background operations, maintaining the rigorous stability required for a continuous operating system environment.Distributed Cognition (DisCo) OrchestrationThe WzrdMerlin v2 architecture completely abandons the rigid, sequential, graph-based execution pipelines that characterized earlier frameworks like LangChain and LangGraph. Instead, the OS transitions to an Event-Driven Actor Model. This paradigm shift to Distributed Cognition (DisCo) is fundamentally necessary to achieve non-blocking, multi-agent collaboration on localized, constrained hardware.The Event Mesh: NATS JetStream and Redis StreamsThe high-velocity data backbone of the operating system operates on a decentralized event mesh, utilizing robust message brokering technologies such as Redis Streams or NATS. While Redis is highly performant for in-memory publish/subscribe architectures, NATS JetStream is uniquely suited for the distributed agentic architecture proposed in v2.NATS JetStream introduces durable message storage, exactly-once delivery semantics, and built-in consumer groups to the lightweight NATS core. In an autonomous operating system, failure is an expected state. If a primary agent crashes due to an unhandled exception, or if the underlying inference model restarts due to an Out of Memory (OOM) error, NATS JetStream ensures that pending tool execution events, webhooks, or inter-agent communications are not lost in the ether. These events are durably persisted to disk or system memory and are systematically replayed the moment the specific agent reconnects to the mesh. This robust messaging infrastructure allows for massive parallelism, enabling independent "Worker" agents to execute long-running tasks—such as web scraping via a Playwright instance—entirely in the background, while the "Manager" agent continues to narrate progress or interact with the user in real-time without interface blocking.Durable Execution via XState ActorsWithin the DisCo framework, the behavior, lifecycle, and local memory of every individual agent are modeled as virtual state machines using XState. The principles of the actor model dictate that each agent maintains its own encapsulated, private state, which cannot be modified by any external entity. Agents communicate strictly through asynchronous message passing over the event mesh, pulling one event at a time from their internal mailboxes.The critical architectural advantage of utilizing XState v5 within this specification is its native support for durable execution. Autonomous workflows frequently involve long-horizon tasks. An agent waiting for an external API to process a request, or a worker agent traversing a deeply nested Document Object Model (DOM) structure, may take several minutes to complete its action. In a traditional system, the agent process would remain active in system RAM, occupying a thread and wasting computational resources while awaiting a callback.The WzrdMerlin v2 system resolves this inefficiency through state persistence. When an agent enters a waiting state, the OS captures the actor's exact internal state machine configuration via the getPersistedSnapshot() function. This state is serialized and durably stored in a local database or key-value store. The agent is then effectively terminated; it "sleeps," consuming zero CPU cycles and freeing up the local event loop. Upon the arrival of the necessary webhook or event payload via the NATS mesh, the OS retrieves the snapshot, hydrates the actor from persistent storage, and seamlessly resumes execution precisely where it left off. This capability to pause and resume complex workflows across transient hardware failures is vital for mitigating agentic chaos and ensuring reliable, long-running operations.The Narrated Computation Loop and the Thinking FirewallTo properly harness the capabilities of modern reasoning models (which often utilize a hidden chain-of-thought process before generating a final output), the v2 specification enforces a strict, multi-tiered structural output pattern for every inference turn. The output must parse into four distinct phases: a Thinking Block, a Narrative Block, an Action Block, and an Observation Block.The implementation of the "Thinking Firewall" represents a crucial parsing innovation for local execution. Advanced models that output raw chain-of-thought reasoning often hallucinate syntactical structures that closely mimic the JSON or XML formats of actual tool calls while "deliberating" on their options. If an un-firewalled, naive parser detects these phantom calls within the reasoning stream, it can trigger recursive, infinite execution loops, commanding the OS to execute broken or dangerous tools and rapidly draining system resources.By rigidly enforcing <think> and </think> tags around the chain-of-thought generation, and firewalling the OS parser to completely ignore all strings generated within these boundaries, the system guarantees safety. Only the deliberate, formatted, and validated commands localized within the designated Action Block are passed to the execution environment. The Narrative Block subsequently translates the internal, firewalled logic into a user-facing explanation, maintaining system transparency and providing a Copilot-style explanation of the agent's planned trajectory.Capability Extension: Model Context Protocol and ForageA defining feature of a truly Autonomous Agent OS is its capacity to adapt to novel, unseen tasks without requiring human developer intervention. WzrdMerlin v2 achieves this dynamic extensibility through the pervasive adoption of the Model Context Protocol (MCP) and the integration of a dedicated self-improvement pipeline.The Model Context Protocol (MCP) StandardizationIntroduced in late 2024 and widely adopted by 2026, the Model Context Protocol acts as a universal, open standard client-server architecture for connecting AI models to disparate data sources and execution environments. Historically, agent developers had to write brittle, fragmented, custom Python scripts for every single API or tool they wished to integrate. MCP replaces this fragmentation.The protocol defines exactly how an AI model can query standardized schemas, fetch local files, interact with enterprise databases, or execute browser automation through highly isolated, purpose-built servers. By enforcing MCP for all interactions, the v2 OS treats tools not as hardcoded functions, but as modular, composable, and easily interchangeable system components that can be mounted and unmounted at will.The "Forage" Self-Improvement ProtocolTo push the boundaries of system autonomy further, the architecture integrates isaac-levine/forage, an advanced MCP server explicitly engineered for self-improving tool discovery. Traditional AI agents are strictly bounded by the static list of tools injected into their context prompt during initialization. If a user asks a traditional agent to query a specialized remote database, but the necessary database tool was not pre-configured by the developer, the agent simply fails and outputs an error.The Forage protocol eliminates this rigid boundary. When the WzrdMerlin v2 agent encounters a capability gap or recognizes it lacks the necessary tool for a user request, it autonomously invokes the forage_search command. This tool queries massive, decentralized community registries, such as the Official MCP Registry, the Smithery ecosystem, or the global npm database, searching for servers that match the semantic requirements of the task.Once an appropriate MCP server is identified, the agent evaluates its parameters and executes forage_install. Crucially, Forage operates as a gateway or proxy server. It securely installs the new MCP toolset as a localized child process and establishes a connection via Standard I/O (stdio) transports. It then dynamically re-registers the child's tools under a specific namespace (e.g., foraged__smithery__postgres_query) and emits a list_changed event notification directly to the agent.This mechanism allows the OS to hot-swap new capabilities directly into the agent's active context window in real-time. It requires no system restarts, no application reloads, and zero manual developer configuration. To complete the self-improvement cycle, the agent utilizes the forage_learn command to serialize the specific usage instructions, schemas, and quirks of the newly acquired tool into persistent .merlin/rules/ files. This ensures that the knowledge is retained, permanently expanding the OS's capability surface for all future sessions.To mitigate security risks associated with autonomous code execution, the implementation inherently requires user authorization for the actual installation phase (confirm: true), ensuring that the supply chain of newly discovered tools is subject to a human-in-the-loop security checkpoint before being granted host access.Code Mode for Token EfficiencyWhile MCP provides the universal connection protocol, passing massive amounts of raw data back through the protocol and into the LLM's context window is computationally disastrous. Streaming a 10,000-row CSV file or a deeply nested DOM tree from a web scraper into the model rapidly exhausts the finite token budget, dilutes the attention mechanism, and degrades the model's reasoning capabilities.To mitigate this, the OS utilizes a localized "Code Mode" sandbox. Instead of engaging in iterative, token-heavy ReAct (Reasoning and Acting) loops—where the model requests a row of data, reads the row, reasons about it, and requests the next row—the agent writes a single, comprehensive Python or TypeScript script. This script is transmitted to the Code Mode sandbox, where it natively executes the data processing, interacts with the relevant MCP servers locally, and returns only the final calculated summary, the specific analytical extraction requested, or a visualization artifact. This architectural shift from "reading the data" to "doing operations on the data" drastically reduces the token transmission overhead by up to 98.7%. It preserves the 9B model's limited attention span for high-level orchestration, planning, and synthesis, rather than wasting cycles on rote data parsing.Advanced Context Engineering and Memory TopologiesThe operational success of a 9-billion parameter model operating over long-horizon tasks is directly proportional to the hygiene, curation, and structural integrity of its context window. As an agent loops through interactions, its context history rapidly balloons with stale diffs, repetitive tool logs, verbose compiler error messages, and iterative data dumps. Without intervention, this leads to severe "lost in the middle" cognitive degradation, where the model forgets its primary objective. WzrdMerlin v2 combats this entropy through a sophisticated, multi-tiered context engineering strategy.Observation Masking (The JetBrains Strategy)To manage the exponential accumulation of environment observations, the OS implements Observation Masking, a highly efficient memory management technique validated by JetBrains Research in late 2025. Traditional context management architectures often rely on LLM summarization, where an auxiliary model periodically reads the entire history and compresses it into a narrative summary. However, rigorous empirical studies demonstrate that summarization is computationally expensive, adds significant API latency, and often harms downstream task performance. Summarization creates a "trajectory elongation" effect; it tends to mask the harsh reality of repeated failure signals, preventing the agent from realizing its current approach is fundamentally flawed.Observation Masking takes a brutally simple but highly effective approach. It maintains a strict rolling window of the most recent agent turns (typically optimized at $M=10$). For any conversational turn that ages out of this window, the raw, verbose tool observation (e.g., a 4,000-token git diff or a massive JSON payload) is stripped entirely from the context and replaced with a static placeholder string, such as [Output Omitted for brevity].Crucially, the agent's reasoning process and the specific action/tool call from that older turn are preserved in full. This allows the model to continuously remember what it previously thought and what specific commands it executed, without being cognitively drowned in the historical noise of what the environment returned. Extensive benchmarks conducted on the SWE-agent scaffold prove that Observation Masking reduces operational token costs by over 50% compared to unmanaged raw agents, while simultaneously improving complex task solve rates by 2.6% over complex LLM summarization techniques.Phase Handoff and Context FoldingFor comprehensive workflows that naturally segment into distinct logical stages (e.g., initial Data Gathering, followed by Data Analysis, leading into Code Generation), the OS employs the Phase Handoff, or Context Folding, methodology developed by Blueshift. When an agent determines that a specific operational phase is complete, it triggers a native phase_handoff event.This event initiates a highly structured garbage collection and state transition routine:Clearing: All transient tool outputs, raw environmental data, and massive JSON payloads accumulated during the previous phase are permanently evicted from the active context window.Tool Swapping: The specific MCP tools utilized for the previous phase are unloaded from the registry, and the specialized tools required for the subsequent phase are injected.Artifact Preservation: Before clearing the memory, the agent analyzes the data and extracts the semantic core of its findings (the "Artifacts"). These are compressed into a highly dense semantic payload (typically ~2KB) and carried forward into the new phase.Journaling: A brief chronological journal entry detailing the phase transition is written to a durable audit log, which is hidden from the active prompt to prevent distraction.This mechanism forces the overall context size to oscillate naturally (e.g., expanding to 150K tokens during research, folding back down to 60K tokens at handoff) rather than growing monotonically toward infinity. By ensuring the model only carries forward highly distilled semantic artifacts, the Phase Handoff system drastically improves reasoning coherence and task completion rates on complex, multi-step operations.The Gardener: Asynchronous Background Memory ConsolidationTo manage cross-session continuity and long-term knowledge retention, WzrdMerlin v2 relies on an autonomous background service known as "The Gardener". Operating entirely asynchronously during the host machine's CPU and GPU idle cycles, The Gardener acts as a dedicated memory consolidation agent.During live user interactions, the system writes raw conversational dialogue and tool execution logs directly to a fast, temporary buffer (the "Hot Path"). Once the system watchdog detects an idle state, The Gardener wakes up and processes this raw buffer. It meticulously prunes redundant information, extracts core entities and relationships, and condenses the historical data into discrete "Atomic Facts". These Atomic Facts are then integrated into a durable local knowledge graph or vector database, utilizing bi-directional linking to map relationships based on connection density.By separating the high-latency, token-heavy task of memory indexing from the real-time chat interface, the OS ensures that the user experiences zero processing lag during interaction. Furthermore, when the primary agent needs to retrieve historical information, it searches the highly compressed "Summary Layer" of Atomic Facts rather than scanning raw, verbose chat logs. The agent then reconstructs the memory using upward reconstruction only when necessary, keeping the active context focused and minimizing token overhead during retrieval operations.In-Context Distillation and Self-Consistency CascadesTo enable the 9B parameter reasoning model to punch significantly above its weight class, the system utilizes an advanced technique known as In-Context Distillation. An incredibly large, massive-capacity "Teacher" model (such as a 70B parameter network run offline, or an external frontier API like Claude 3.5 Sonnet) is occasionally utilized to successfully solve highly complex logic puzzles, difficult architectural designs, or intricate coding tasks. These successful, high-quality reasoning traces are captured and stored in a local, durable demonstration memory bank.When the local 9B "Student" model encounters a structurally similar problem during normal operations, the OS dynamically retrieves the top-$k$ most relevant teacher demonstrations from the vector database. It injects these pristine reasoning traces directly into the system prompt as few-shot examples. This process allows the smaller, highly efficient local model to mimic the advanced reasoning pathways and formatting structures of the teacher on-the-fly, achieving near-teacher accuracy without the catastrophic hardware requirement of actively loading a massive 70B model into the constrained 12GB VRAM environment.Furthermore, if the 9B model's internal confidence metrics remain low during a task, it employs Self-Consistency Cascades. The model generates multiple varied responses and evaluates them for consensus. If the responses diverge significantly, indicating confusion or hallucination, the OS automatically halts the execution loop and "cascades" the prompt. It either flags the operation for human intervention or briefly utilizes the GTT spilling mechanics to load a higher-tier local model to resolve the specific bottleneck before returning control to the 9B base model.UI/UX: The Native Agent Workspace and VRAM PreservationThe graphical user interface (GUI) of an Autonomous Agent OS must adhere to a strict philosophy: the UI must never compete with the inference engine for systemic resources. Historically, modern desktop applications built on popular web technologies (specifically frameworks like Electron) inherently consume massive amounts of System RAM and VRAM. This is due to the inherent overhead of bundling a full, independent Chromium browser engine and a complete Node.js runtime environment for every single application. In a system that is meticulously balancing a 12GB VRAM and 48GB SysRAM split to maximize machine learning context windows, wasting half a gigabyte of memory on UI rendering is an unacceptable architectural compromise.Tauri v2 and Native Webview IntegrationWzrdMerlin v2 elegantly solves this critical bottleneck by utilizing the Tauri v2 framework coupled with React 19. Tauri fundamentally alters the desktop application paradigm by entirely discarding the embedded Chromium engine. Instead, it leverages the host operating system's pre-existing, native webview components to render the application interface (e.g., WKWebView on macOS, WebKitGTK on Linux, and Edge WebView2 on Windows).Framework ArchitectureRendering EngineAverage Base Memory FootprintCompiled Binary SizeElectron ApplicationsBundled Chromium + Node.js~300MB - 580MB80MB - 120MB+Tauri v2 ApplicationsOS Native Webview + Rust~50MB - 150MB5MB - 15MBTable 2: Memory footprint and binary size comparisons between desktop application frameworks, highlighting the efficiency of Tauri v2.By utilizing Tauri, the base memory footprint of the entire application interface drops precipitously to approximately 50MB–125MB. This architectural choice liberates hundreds of megabytes of system RAM and VRAM. In the specific context of LLM inference, 300MB of reclaimed memory translates directly to the ability to store tens of thousands of additional tokens in a quantized KV cache, tangibly increasing the agent's cognitive horizon and analytical depth.Rust Sidecars and Zero-Latency TelemetryTauri's backend architecture is written entirely in Rust, a systems programming language that provides highly efficient, safe, and concurrent memory management without the overhead of a garbage collector. WzrdMerlin v2 utilizes native Rust "sidecars" to manage the entire multi-agent orchestration layer and the hardware monitoring services. The Rust sidecar interfaces directly with the host system's hardware sensors, polling the i7-12700K CPU and RX 6750 XT GPU utilization, temperatures, and VRAM saturation at near-zero latency.Because the Rust backend natively handles all heavy computational workloads (such as file system parsing, local database queries, vector similarity search, and process group cleanup), the JavaScript/React frontend is entirely decoupled from intensive system operations. The React 19 interface functions solely as a thin, highly responsive rendering layer that visually maps the Figma-designed UI components to the underlying Rust commands.Hybrid Streaming Data FlowTo seamlessly handle the massive, asynchronous data throughput generated by multiple independent agents operating concurrently, the OS employs a hybrid streaming strategy over the Inter-Process Communication (IPC) bridge.WebSockets are utilized strictly for bidirectional, real-time chat interactions between the user and the primary manager agent, ensuring immediate responsiveness and low latency for human inputs.Server-Sent Events (SSE) are deployed for unidirectional data streams. This includes piping thousands of lines of terminal execution logs, high-frequency hardware telemetry metrics, and rapid agent state transitions to the UI trace inspector.This hybrid approach ensures that the high-frequency background noise of worker agent execution does not block, overload, or latency-spike the user's primary communication channel with the manager agent.Implementation Architecture OverviewThe physical file structure of the WzrdMerlin v2 specification reflects the clear separation of concerns mandated by the DisCo orchestration and hardware-aware inference engines:src-tauri/: Houses the highly efficient Rust sidecar. This layer manages the hardware sensor polling, low-level OS interactions, and process group cleanup ensuring no zombie MCP servers remain active.merlin-core/: The brain of the operation, containing the Python and Rust hybrid logic.actors/: Manages the XState state machines, handling the durable state transitions, sleeping, and waking of agents.bus/: The integration point for the Redis Stream or NATS JetStream event mesh.inference/: The highly customized llama.cpp wrapper executing the dual-API (Vulkan/ROCm) strategy and GTT spilling mechanics.mcp/: Manages the dynamic MCP Client, the Forage protocol integration, and the Code Mode secure execution sandbox.merlin-dashboard/: The thin, VRAM-preserving React 19 frontend rendering the native webview, providing the narrative terminal, GTT monitors, and trace inspectors..merlin/: The persistent storage directory containing the durable memory stores, the teacher demonstration vector database, and the dynamically generated tool rules from the Forage server.ConclusionThe WzrdMerlin v2 specification presents a masterful alignment of advanced software architecture with constrained hardware realities. By explicitly designing around the specific computational limitations and strengths of the RDNA2 architecture and the rigid 12GB VRAM boundary, the system successfully avoids the standard performance pitfalls of generalized, web-based LLM wrappers.The implementation of a dual-API inference engine correctly routes compute-heavy prefill tasks to the ROCm API while intelligently leveraging Vulkan's superior memory bandwidth utilization for the token generation phase. The total paradigm shift to a Distributed Cognition framework via XState and the NATS JetStream event mesh ensures durable, fault-tolerant agent execution capable of spanning long-horizon tasks. Furthermore, the integration of dynamic, self-improving MCP tool discovery via the Forage protocol, combined with aggressive, mathematically proven context engineering techniques like JetBrains Observation Masking and Blueshift Context Folding, guarantees that the 9B reasoning model operates perpetually within its optimal cognitive parameters. Supported by a remarkably lightweight Tauri v2 interface that prioritizes VRAM availability over bundled web engines, the WzrdMerlin v2 architecture successfully establishes a robust, highly efficient, and extensible local operating system for autonomous artificial intelligence agents.

