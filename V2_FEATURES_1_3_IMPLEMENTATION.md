# V2 Features 1-3 Implementation Reference

This document defines and tracks implementation status for:

1. Reliability Core
2. Actor Contract Normalization
3. Dynamic Router Orchestration

## Feature 1: Reliability Core

### Goals
- Prevent infinite or duplicate step execution.
- Guarantee fast failure on hung model inference.
- Ensure failed tasks emit failed events on the correct channel.

### Implemented
- Hard per-step LLM timeout in `BaseAgentActor.handle_step_requested`.
- Idempotency guard for duplicate `step.requested` event IDs.
- In-flight task guard to prevent re-entrant concurrent execution for the same task.
- Forced loop termination when identical action repeats beyond threshold.
- `_fail_task()` now publishes `ACTION_FAILED` on `events.action.failed`.

### Acceptance Criteria
- Duplicate `step.requested` events for the same event ID are ignored.
- Re-entrant `step.requested` for same task while running is ignored.
- LLM timeout marks task failed and emits `events.action.failed`.

## Feature 2: Actor Contract Normalization

### Goals
- Enforce strict schema for critical event payloads.
- Fail fast on malformed payloads.
- Keep event contracts centralized and reusable.

### Implemented
- Added payload models in `src/core/events.py`:
  - `TaskCreatedPayload`
  - `StepRequestedPayload`
  - `ActorActionRequestPayload`
- Added `validate_event_payload(event_type, payload)` utility.
- Router and BaseAgent now validate incoming payloads for critical paths.

### Acceptance Criteria
- Invalid `task.created` payload causes `task.failed`.
- Invalid `action.requested` payload causes `action.failed`.
- Invalid `step.requested` payload is rejected without crash.

## Feature 3: Dynamic Router Orchestration

### Goals
- Replace hardcoded implementer-only routing.
- Route by policy based on task semantics and complexity.
- Ensure target actors exist and are connected.

### Implemented
- Router policy method `_select_target_actor(description)`:
  - Review/analysis/audit-like tasks -> `agent-auditor`
  - Long high-complexity requests -> `agent-auditor`
  - Default -> `agent-implementer`
- Emits `task.routed` events with rationale.
- Added `auditor_agent` runtime actor in `main.py` and wired lifecycle.

### Acceptance Criteria
- Audit/review prompts route to `agent-auditor`.
- Standard execution prompts route to `agent-implementer`.
- `task.routed` event is published with target and rationale.

## Test Plan

### Unit Tests
- Reliability:
  - duplicate step event dedupe
  - fail-task subject/type correctness
- Contracts:
  - invalid payload rejection in router/base agent
- Routing:
  - keyword-based auditor routing
  - default implementer routing

### Integration Smoke
- Start stack with `docker compose up --build -d`.
- Submit two tasks:
  - "audit/review" phrasing (should route auditor)
  - normal implementation phrasing (should route implementer)
- Verify in `api/debug/actors` and logs.

## Notes
- This document is the reference baseline for future Features 4+.
