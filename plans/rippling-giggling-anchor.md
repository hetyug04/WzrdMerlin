# Plan: Agent Model Unload on Task End

## Context

When `spawn_agents` is called, the orchestrator explicitly unloads the 9B model from GPU VRAM so workers have headroom for their 2B models. When all workers finish, `_wake_orchestrator()` reloads the 9B model. This part already works.

The bug: **worker 2B models are never explicitly unloaded**. Because they were loaded with `keep_alive="-1"`, Ollama keeps them in VRAM indefinitely. After a task with sub-agents completes:
- Orchestrator 9B: reloaded ✓
- Worker 2B model: still occupying GPU VRAM ✗

Additionally, when a task ends in **failure or cancellation** (especially after `spawn_agents` was called), the orchestrator's 9B model is not guaranteed to be reloaded — the failure paths skip `_wake_orchestrator()` entirely.

## Changes

### 1. `src/core/worker.py` — Unload worker model in `_cleanup_worker()`

Inside `_patch_agent_completion()`, the `_cleanup_worker()` inner function (lines 244–254) handles NATS cleanup when a worker reaches a terminal state. Add an explicit `unload_model()` call **before** NATS cleanup so the 2B model is evicted from Ollama VRAM immediately.

```python
async def _cleanup_worker():
    # Evict worker model from GPU VRAM before cleanup
    try:
        if hasattr(agent, "ollama") and agent.ollama is not None:
            await agent.ollama.unload_model()
            logger.info(
                f"WORKER {worker_id} ({personality_name}): model unloaded from GPU"
            )
    except Exception as e:
        logger.debug(f"WORKER {worker_id}: model unload error: {e}")
    # Existing NATS cleanup follows unchanged
    try:
        for sub in list(agent._subs):
            ...
```

### 2. `src/core/base_agent.py` — Add `_schedule_model_preload()` helper

Add a new method to the class that safely fire-and-forgets a model preload for the orchestrator only. Workers are identified by `self.role.startswith("worker-")` (set in `worker.py` line 72: `role=f"worker-{personality_name}"`).

```python
def _schedule_model_preload(self) -> None:
    """Fire-and-forget: reload the orchestrator 9B model after a task ends.
    Workers skip this — _cleanup_worker() handles their model lifecycle."""
    if not self.role.startswith("worker-") and hasattr(self, "ollama") and self.ollama:
        asyncio.create_task(self.ollama.preload_model())
```

### 3. `src/core/base_agent.py` — Call `_schedule_model_preload()` at all terminal paths

Add `self._schedule_model_preload()` immediately before `return` in these five places:

| Location | Description | Approx. line |
|---|---|---|
| `done()` tool path | After `logger.info("Task completed successfully")` | ~655 |
| Cancelled path | After inline `_safe_publish(action.failed)` in `handle_step_requested` | ~387 |
| Stall force-exit | After `_safe_publish(action.failed)` in the stall block | ~506 |
| `_fail_task()` helper | After `_safe_publish(action.failed)` | ~1164 |
| Unhandled exception catch | At end of `except Exception` block | ~716 |

The `done()` path also runs for workers. The guard in `_schedule_model_preload()` prevents workers from triggering a 9B preload there.

### Not changed

- `_wake_orchestrator()`: already calls `self.ollama.preload_model()` when workers finish normally — no change needed.
- `tool_spawn_agents()`: already calls `self.ollama.unload_model()` when spawning — no change needed.
- `ollama_client.py`: `unload_model()` and `preload_model()` already exist and work correctly.

## Edge Cases

**Workers still running when orchestrator fails**: The `_schedule_model_preload()` will fire the 9B preload while workers may still be active. Ollama will load the 9B alongside the 2B model; if VRAM is insufficient it evicts the LRU model. Workers may need to reload their 2B model for any remaining inference — acceptable (slower, not broken). Workers will still call `_cleanup_worker()` when they finish, unloading their 2B model.

**No spawn_agents used**: `_schedule_model_preload()` is a no-op in practice — 9B was never unloaded, so Ollama just refreshes its `keep_alive` timer.

## Verification

1. Run a task that uses `spawn_agents` (e.g. a multi-step research task)
2. After task completes, check `GET /api/llm/health` → `model_loaded: true` for the 9B
3. Check Ollama directly: `curl http://host.docker.internal:11434/api/tags` — worker model (`qwen3.5:2b`) should NOT appear in the running models list
4. Run a task that fails mid-way (e.g. trigger a stall) — verify 9B is still loaded after failure
5. Check logs for `WORKER ... model unloaded from GPU` entries

## Files Modified

- `src/core/worker.py` — `_cleanup_worker()` inner function in `_patch_agent_completion()`
- `src/core/base_agent.py` — new `_schedule_model_preload()` method + 5 call sites
