# WzrdMerlin v2 Implementation Roadmap

This document tracks the progress of the v2 "Distributed Cognition" (DisCo) Autonomous Agent OS, cross-referenced against the research specification in `GEMINI.md`.

---

## **Phase 1: The Resource Foundation (Survival & Space)**
*Goal: Reclaim the VRAM and System RAM needed to actually run 9B reasoning models without crashing.*

- [x] **Tauri v2 Wrapper**: Move the dashboard into a native webview to save ~500MB+ VRAM vs standard browsers.
- [x] **Rust Sidecar**: Native hardware telemetry (VRAM/GTT saturation) and cleanup hooks.
- [x] **Dual-API Disaggregation**: 
    - [x] ROCm backend for Prefill (compute-bound).
    - [x] Vulkan backend for Decode (memory-bound).
- [x] **GTT Spilling Heuristics**: Dynamic KV cache management to spill into 48GB System RAM when VRAM saturation is detected.
- [x] **Local Model Interface**: Direct httpx-based Ollama/llama-server integration (removing LiteLLM overhead).
- [ ] **Predictive Resource Scheduler**: LRU-based model eviction and idle watchdog (currently a placeholder in `inference.py`).

---

## **Phase 2: Universal Connectivity (The MCP Shift)**
*Goal: Stop hardcoding tools and enable the "emergent" capability discovery you requested.*

- [x] **Core Toolset**: Basic `shell`, `read_file`, `write_file`, and `memory` tools.
- [x] **MCP-Native Architecture**: Integrated `mcp` SDK and `MCPManager` for dynamic tool loading.
- [x] **Forage Pipeline**:
    - [x] `forage_search`: Query MCP Registry/Smithery for new tools (simulated registry).
    - [x] `forage_install`: Spawn new MCP servers as local proxies and update `mcp_config.json`.
    - [x] `forage_learn`: Persist tool schemas to `.merlin/rules/`.
- [x] **Code Mode Sandbox**: Agent-generated scripts for heavy data processing (CSV/JSON) via `python_sandbox`.


---

## **Phase 3: Cognitive Reliability (Durable Execution)**
*Goal: Enable the agent to work on tasks that take hours or days without getting "lost" or stuck in loops.*

- [x] **Event Mesh Foundation**: NATS JetStream integration for agent communication.
- [x] **Basic Actor Pattern**: `BaseActor` class with NATS `publish`/`listen` semantics.
- [x] **Durable Execution (XState v5 Pattern)**: Encapsulated virtual state machines persisted to NATS KV after every step.
- [x] **Asynchronous "Sleep/Resume"**: Agents serialize state and terminate between steps, hydrating via `STEP_REQUESTED` events.
- [x] **Multi-Agent Hand-off**: Specialized routing in `DisCoRouter` for task lifecycle management.
- [x] **Observation Masking**: Rolling window logic to replace verbose tool logs with `[Output Omitted]` placeholders.
- [x] **Context Folding**: Automatic compression of history into masked artifacts during long-running tasks.


---

## **Phase 4: Autonomous Evolution (The "Gardener")**
*Goal: Merlin begins to manage its own growth.*

- [x] **Isolated Worktrees**: Git worktree isolation for candidate code generation.
- [x] **Validation Pipeline**: `compileall` + `pytest` checks before promotion.
- [x] **Rollback Mechanism**: Persistent `rollback.json` to revert failed improvements.
- [ ] **The Gardener**: Background agent to consolidate logs into "Atomic Facts" and a local vector graph.
- [ ] **Self-Generated Tasks**: Enabling Merlin to identify its own capability gaps without user prompting.
- [ ] **Behavioral Regression Testing**: Testing the agent's actual tool-use ability after code changes, not just unit tests.
- [ ] **In-Context Distillation**: Dynamic retrieval of "Teacher" (Claude 3.5) reasoning traces for few-shot injection.
- [ ] **Figma UI Implementation**: Update the dashboard to match the new high-fidelity design.

---

## **Strategy Rationale**

### **Why this order?**
If you build the **Self-Improvement** (Phase 4) first, the agent will frequently fail because it doesn't have enough **VRAM** (Phase 1) or the right **Tools** (Phase 2) to fix itself. By building the "Hardware -> Tools -> State -> Autonomy" stack, you ensure the agent has the physical and logical resources it needs at every step of its evolution.

1. **Phase 1 (Hardware)**: Recovers ~800MB of VRAM (via Tauri) to allow 32k+ context on a 12GB card.
2. **Phase 2 (Connectivity)**: Replaces hardcoded tools with universal MCP servers, allowing the agent to "forage" for new skills (e.g., Playwright).
3. **Phase 3 (State)**: Implements durable execution so long-running tasks don't crash the system if a process restarts.
4. **Phase 4 (Autonomy)**: The final layer where the agent autonomously cleans its own memory and generates its own improvement tasks.
