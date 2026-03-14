"""
WzrdMerlin v2 — The Gardener

Asynchronous background agent for memory consolidation.
Wakes during idle periods (detected via watchdog heartbeats) and:

  1. Scans completed task traces from NATS KV state
  2. Uses the LLM to extract "Atomic Facts" from verbose histories
  3. Deduplicates against existing memories via vector similarity
  4. Stores extracted facts in ChromaDB episodic memory
  5. Archives full task trajectories for In-Context Distillation

The Gardener never runs during active user tasks—it only
processes during CPU/GPU idle windows to avoid resource contention.
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

from src.core.actor import BaseActor
from src.core.events import Event, EventType
from src.core.memory import get_memory
from src.core.llm import ModelInterface

logger = logging.getLogger(__name__)

# How long the system must be idle (no active tasks) before the Gardener runs
IDLE_THRESHOLD_SECONDS = 120          # 2 minutes
# Minimum time between Gardener runs
COOLDOWN_SECONDS = 600                # 10 minutes
# Maximum tasks to process per run (avoid hogging resources)
MAX_BATCH = 10
# Similarity threshold above which an extracted fact is considered duplicate
DEDUP_THRESHOLD = 0.85

CONSOLIDATION_PROMPT = """\
You are a memory consolidation engine. Given the following task execution trace,
extract a concise list of **Atomic Facts** — discrete, self-contained knowledge items.

Focus on:
- Patterns that worked (successful approaches, useful tool sequences)
- Patterns that failed (errors encountered, dead ends)
- API endpoints, file paths, or configuration details used
- Domain-specific knowledge discovered during the task

Rules:
- Each fact should be 1-2 sentences maximum
- Output ONLY a JSON array of strings, one per fact
- Skip trivial facts (e.g. "the task was completed")
- Aim for 2-6 facts per trace; output [] if nothing worth remembering

TASK TRACE:
{trace}

JSON ARRAY:"""


class GardenerActor(BaseActor):
    """Background memory consolidation agent."""

    def __init__(self, nats_url: str = None):
        super().__init__(name="gardener", nats_url=nats_url)
        self._last_run: float = 0.0
        self._last_active_task: float = time.time()
        self._running = False
        self._llm: Optional[ModelInterface] = None

    async def connect(self):
        await super().connect()
        # Listen for heartbeats to detect idle windows
        self.on("events.system.heartbeat", self._on_heartbeat)
        await self.listen("events.system.heartbeat")
        # Listen for task creation/completion to track activity
        self.on("events.action.completed", self._on_task_activity)
        await self.listen("events.action.completed")
        self.on("events.action.requested", self._on_task_activity)
        await self.listen("events.action.requested")
        # Start the idle check loop
        asyncio.create_task(self._idle_loop())
        logger.info("GardenerActor active — waiting for idle windows")

    def set_llm(self, llm: ModelInterface):
        self._llm = llm

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_heartbeat(self, event: Event):
        """Track heartbeats but don't trigger work — the idle loop handles that."""
        pass

    async def _on_task_activity(self, event: Event):
        """Reset the idle timer on any task activity."""
        self._last_active_task = time.time()

    # ------------------------------------------------------------------
    # Idle detection loop
    # ------------------------------------------------------------------

    async def _idle_loop(self):
        """Periodically check if the system is idle enough to run consolidation."""
        while True:
            await asyncio.sleep(30)
            if self._running:
                continue
            now = time.time()
            idle_for = now - self._last_active_task
            since_last_run = now - self._last_run
            if idle_for >= IDLE_THRESHOLD_SECONDS and since_last_run >= COOLDOWN_SECONDS:
                asyncio.create_task(self._run_consolidation())

    # ------------------------------------------------------------------
    # Core consolidation logic
    # ------------------------------------------------------------------

    async def _run_consolidation(self):
        """Main consolidation run — process completed task traces."""
        if self._running:
            return
        self._running = True
        self._last_run = time.time()
        logger.info("GARDENER: Starting memory consolidation run")

        try:
            # 1. Find completed tasks that haven't been consolidated
            traces = await self._collect_unprocessed_traces()
            if not traces:
                logger.info("GARDENER: No unprocessed traces found")
                return

            logger.info(f"GARDENER: Processing {len(traces)} task traces")

            mem = get_memory()
            facts_added = 0
            trajectories_stored = 0

            for task_id, trace_data in traces[:MAX_BATCH]:
                # Check if still idle — abort if user starts a new task
                if time.time() - self._last_active_task < 30:
                    logger.info("GARDENER: User activity detected, pausing consolidation")
                    break

                # 2. Extract atomic facts via LLM
                facts = await self._extract_facts(trace_data)

                # 3. Deduplicate against existing memories
                for fact in facts:
                    is_dup = await self._is_duplicate(fact, mem)
                    if not is_dup:
                        await mem.add(
                            content=fact,
                            tags=["atomic_fact", "gardener", f"task:{task_id}"],
                        )
                        facts_added += 1

                # 4. Store full trajectory for distillation
                trajectory_content = self._format_trajectory(task_id, trace_data)
                if trajectory_content:
                    await mem.add(
                        content=trajectory_content,
                        tags=["trajectory", f"task:{task_id}"],
                        metadata={
                            "task_id": task_id,
                            "outcome": trace_data.get("status", "unknown"),
                        },
                        collection="trajectories",
                    )
                    trajectories_stored += 1

                # 5. Mark as consolidated
                await self._mark_consolidated(task_id)

            logger.info(
                f"GARDENER: Consolidation complete — "
                f"{facts_added} new facts, {trajectories_stored} trajectories stored"
            )

            # Emit stats event for dashboard
            await self.publish("events.system.info", Event(
                type=EventType.SYSTEM_INFO,
                source_actor=self.name,
                correlation_id="gardener",
                payload={
                    "event": "consolidation_complete",
                    "facts_added": facts_added,
                    "trajectories_stored": trajectories_stored,
                    "traces_processed": min(len(traces), MAX_BATCH),
                },
            ))

        except Exception as e:
            logger.error(f"GARDENER: Consolidation error: {e}", exc_info=True)
        finally:
            self._running = False

    # ------------------------------------------------------------------
    # Trace collection from KV state
    # ------------------------------------------------------------------

    async def _collect_unprocessed_traces(self) -> List[tuple]:
        """Scan KV state for completed tasks not yet consolidated."""
        traces = []
        try:
            # List all actor_state keys
            kv = self.state_store.kv
            if not kv:
                return traces
            try:
                keys = await kv.keys()
            except Exception:
                # KV bucket empty — keys() raises "no keys found"
                return traces
            for key in keys:
                if not key.startswith("actor_state."):
                    continue
                try:
                    entry = await kv.get(key)
                    state = json.loads(entry.value.decode())
                except Exception:
                    continue

                # Only process completed/failed tasks
                status = state.get("status", "")
                if status not in ("completed", "failed"):
                    continue
                # Skip if already consolidated
                if state.get("gardener_consolidated"):
                    continue
                task_id = key.replace("actor_state.", "")
                traces.append((task_id, state))
        except Exception as e:
            logger.warning(f"GARDENER: Error collecting traces: {e}")
        return traces

    # ------------------------------------------------------------------
    # LLM-based fact extraction
    # ------------------------------------------------------------------

    async def _extract_facts(self, trace_data: Dict) -> List[str]:
        """Use the LLM to extract atomic facts from a task trace."""
        if not self._llm:
            logger.warning("GARDENER: No LLM available for fact extraction")
            return []

        # Build a condensed trace string
        instruction = trace_data.get("instruction", "")[:500]
        result = trace_data.get("result", "")[:500]
        history = trace_data.get("history", [])

        trace_parts = [f"Instruction: {instruction}"]
        if result:
            trace_parts.append(f"Final result: {result}")

        # Include key tool calls from history (condensed)
        for step in history[-10:]:  # Last 10 steps max
            action = step.get("action", "")
            tool_result = step.get("result", "")
            if action:
                trace_parts.append(f"Action: {action[:200]}")
            if tool_result:
                trace_parts.append(f"Observation: {tool_result[:200]}")

        trace_text = "\n".join(trace_parts)
        if len(trace_text) < 50:
            return []  # Too short to extract anything meaningful

        prompt = CONSOLIDATION_PROMPT.format(trace=trace_text)

        try:
            response = await self._llm.generate_text(prompt, max_tokens=500)
            # Parse JSON array from response
            response = response.strip()
            # Handle markdown code blocks
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            facts = json.loads(response)
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, str) and len(f) > 10]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"GARDENER: Failed to parse facts from LLM: {e}")
        return []

    # ------------------------------------------------------------------
    # Deduplication via vector similarity
    # ------------------------------------------------------------------

    async def _is_duplicate(self, fact: str, mem) -> bool:
        """Check if a fact already exists in memory (above similarity threshold)."""
        try:
            results = await mem.search(fact, top_k=1, min_score=DEDUP_THRESHOLD)
            return len(results) > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Trajectory formatting
    # ------------------------------------------------------------------

    def _format_trajectory(self, task_id: str, trace_data: Dict) -> Optional[str]:
        """Format a task trace for the trajectories collection (distillation)."""
        instruction = trace_data.get("instruction", "")
        result = trace_data.get("result", "")
        history = trace_data.get("history", [])
        status = trace_data.get("status", "unknown")

        if not instruction or not history:
            return None

        parts = [
            f"Task: {instruction[:500]}",
            f"Outcome: {status}",
        ]

        for step in history:
            action = step.get("action", "")
            obs = step.get("result", "")
            if action:
                parts.append(f"Action: {action[:300]}")
            if obs:
                parts.append(f"Observation: {obs[:300]}")

        if result:
            parts.append(f"Final: {result[:500]}")

        trajectory = "\n".join(parts)
        # Only store trajectories with enough substance
        if len(trajectory) < 100:
            return None
        # Cap at reasonable size for embedding
        return trajectory[:3000]

    # ------------------------------------------------------------------
    # State bookkeeping
    # ------------------------------------------------------------------

    async def _mark_consolidated(self, task_id: str):
        """Mark a task as consolidated in KV state."""
        try:
            state = await self.state_store.get(f"actor_state.{task_id}")
            if state:
                state["gardener_consolidated"] = True
                state["gardener_timestamp"] = int(time.time())
                await self.state_store.put(f"actor_state.{task_id}", state)
        except Exception as e:
            logger.warning(f"GARDENER: Failed to mark {task_id} consolidated: {e}")
