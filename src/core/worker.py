"""
WzrdMerlin v3 — WorkerAgentActor

A lightweight specialist agent spawned by the orchestrator (BaseAgentActor) to
execute a focused sub-task. Key differences from the orchestrator:

- Uses qwen3.5:2b (or the personality's model) instead of the 9B model
- Limited to a caller-specified tool allowlist
- Bounded max_iterations (15 by default)
- On completion/failure publishes WORKER_COMPLETED so the orchestrator wakes up
- Supports recursive spawn up to workers.max_depth (depth guard prevents runaway spawning)
- Ephemeral: unsubscribes from NATS after completing and does not persist state across restarts
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from src.core.actor import BaseActor
from src.core.config import get_config
from src.core.events import Event, EventType
from src.core.handoff import get_personality

logger = logging.getLogger(__name__)


class WorkerAgentActor:
    """
    Ephemeral specialist agent.  Wraps a BaseAgentActor with:
      • personality-scoped system prompt
      • tool allowlist filtering
      • depth-capped recursive spawning
      • WORKER_COMPLETED publication when done or failed
    """

    def __init__(
        self,
        task_description: str,
        parent_task_id: str,
        worker_id: str,
        personality_name: str,
        allowed_tools: List[str],
        context_summary: str = "",
        spawn_depth: int = 0,
        nats_url: Optional[str] = None,
        parent_http_client=None,
    ):
        self.task_description = task_description
        self.parent_task_id = parent_task_id
        self.worker_id = worker_id
        self.personality_name = personality_name
        self.allowed_tools = allowed_tools
        self.context_summary = context_summary
        self.spawn_depth = spawn_depth
        self._nats_url = nats_url
        self._parent_http_client = parent_http_client

        # Resolve personality
        self.personality = get_personality(personality_name)
        if not self.personality:
            logger.warning(
                f"WORKER {worker_id}: Unknown personality '{personality_name}' — "
                "falling back to defaults"
            )

        # Build the underlying agent (import here to avoid circular imports)
        from src.core.base_agent import BaseAgentActor  # noqa: PLC0415
        self._agent = BaseAgentActor(
            nats_url=nats_url,
            role=f"worker-{personality_name}",
        )
        self._agent.name = f"worker-{worker_id}"

        # Apply personality model override
        cfg = get_config()
        model_key = (
            self.personality.model_profile_key
            if self.personality
            else cfg.workers.default_model
        )
        profile = cfg.models.get(model_key) or cfg.models.get(cfg.workers.default_model)
        if profile:
            from src.core.ollama_client import OllamaClient  # noqa: PLC0415
            self._agent.ollama = OllamaClient(
                base_url=os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url),
                model=profile.model_name,
                context_window=profile.context_window,
                temperature=profile.temperature,
                think=profile.think,
                read_timeout=cfg.ollama.read_timeout,
                keep_alive=cfg.ollama.keep_alive,
                shared_http_client=self._parent_http_client,
            )

        # Inject spawn depth + personality info so the agent can pass them downstream
        self._agent._spawn_depth = spawn_depth
        if self.personality:
            self._agent._active_personality = personality_name

        # Filter tool allowlist (empty list = all tools allowed).
        # Workers NEVER get spawn_agents — recursive spawning exhausts VRAM
        # and the 2B models aren't reliable enough to orchestrate sub-agents.
        _WORKER_BANNED_TOOLS = {"spawn_agents", "handoff_personality"}
        if allowed_tools:
            self._agent.tools = {
                k: v for k, v in self._agent.tools.items()
                if k in allowed_tools and k not in _WORKER_BANNED_TOOLS
            }
        else:
            for banned in _WORKER_BANNED_TOOLS:
                self._agent.tools.pop(banned, None)

        # Cap max iterations
        self._max_iterations = cfg.workers.max_iterations

        # Patch agent's system prompt generation to use personality override
        if self.personality and self.personality.system_prompt_override:
            self._apply_personality_prompt()

        # Wire completion hook into the agent
        self._patch_agent_completion()

    # ------------------------------------------------------------------
    # Personality prompt injection
    # ------------------------------------------------------------------

    def _apply_personality_prompt(self):
        """Monkey-patch _generate_single_action to use the personality's system prompt."""
        personality = self.personality
        context_summary = self.context_summary
        original_generate = self._agent._generate_single_action

        async def _patched_generate(task_id, instruction, history, iteration):
            # Build augmented instruction with orchestrator context
            augmented = instruction
            if context_summary:
                augmented = (
                    f"[ORCHESTRATOR CONTEXT]\n{context_summary}\n\n"
                    f"[YOUR SUB-TASK]\n{instruction}"
                )

            # Swap ollama's system prompt temporarily via the existing generate path
            # We override by calling stream_chat with our prompt directly
            from src.core.events import EventType as ET  # noqa: PLC0415
            ollama = self._agent.ollama
            system_prompt = personality.system_prompt_override

            # Append tool list to the personality prompt if tools are restricted
            if self._agent.tools:
                tool_names = ", ".join(self._agent.tools.keys())
                system_prompt += f"\n\nAVAILABLE TOOLS: {tool_names}"

            await self._agent._ui_emit(ET.AGENT_THINKING, task_id, {
                "iteration": iteration,
                "status": "generating",
                "text": f"[worker/{self._agent._active_personality}] iteration {iteration}…",
            })

            messages = ollama.build_messages(system_prompt, history, augmented)
            content_buf = []

            async for chunk_type, text in ollama.stream_chat(messages, heartbeat_cb=None):
                if chunk_type == "think":
                    self._agent._ui_emit_nowait(ET.AGENT_THINKING, task_id, {"text": text})
                else:
                    content_buf.append(text)
                    self._agent._ui_emit_nowait(ET.AGENT_STREAMING, task_id, {"text": text})

            full_response = "".join(content_buf)
            action_payload = ollama.parse_action(full_response)
            action_payload = self._agent._normalize_action_payload(
                action_payload, full_response, augmented
            )

            if not action_payload:
                cascade = await self._agent._self_consistency_cascade(
                    system_prompt, history, augmented, task_id, iteration
                )
                if cascade:
                    return cascade, ""

            return action_payload, ""

        self._agent._generate_single_action = _patched_generate

    # ------------------------------------------------------------------
    # Completion hook
    # ------------------------------------------------------------------

    def _patch_agent_completion(self):
        """Intercept handle_step_requested to enforce max_iterations and publish
        WORKER_COMPLETED when the agent finishes (done/failed)."""
        parent_task_id = self.parent_task_id
        worker_id = self.worker_id
        personality_name = self.personality_name
        max_iter = self._max_iterations
        agent = self._agent

        original_handle_step = agent.handle_step_requested

        async def _patched_handle_step(event):
            # Hard cap on iterations
            task_id = event.payload.get("task_id", "")
            try:
                state = await agent.state_store.get(f"actor_state.{task_id}")
                if state and state.get("iteration", 0) >= max_iter:
                    reason = f"Worker hit max_iterations={max_iter}"
                    logger.warning(f"WORKER {worker_id}: {reason}")
                    state["status"] = "failed"
                    state["result"] = reason
                    await agent.state_store.put(f"actor_state.{task_id}", state)
            except Exception:
                pass

            # Run the actual step
            await original_handle_step(event)

            # After step: check terminal state and publish WORKER_COMPLETED
            try:
                state = await agent.state_store.get(f"actor_state.{task_id}")
                if state and state.get("status") in ("completed", "failed"):
                    result_summary = state.get("result", "")[:2000]
                    status = state["status"]
                    await agent._safe_publish(
                        "events.worker.completed",
                        Event(
                            type=EventType.WORKER_COMPLETED,
                            source_actor=agent.name,
                            target_actor=None,
                            correlation_id=task_id,
                            payload={
                                "parent_task_id": parent_task_id,
                                "worker_id": worker_id,
                                "personality": personality_name,
                                "status": status,
                                "result_summary": result_summary,
                            },
                        ),
                    )
                    logger.info(
                        f"WORKER {worker_id} ({personality_name}): "
                        f"published WORKER_COMPLETED status={status}"
                    )
                    # Unsubscribe to avoid memory leaks from ephemeral workers
                    await _cleanup_worker()
            except Exception as e:
                logger.warning(f"WORKER {worker_id}: completion hook error: {e}")

        async def _cleanup_worker():
            # Evict worker model from GPU VRAM before NATS cleanup
            try:
                if hasattr(agent, "ollama") and agent.ollama is not None:
                    await agent.ollama.unload_model()
                    logger.info(
                        f"WORKER {worker_id} ({personality_name}): model unloaded from GPU"
                    )
            except Exception as e:
                logger.debug(f"WORKER {worker_id}: model unload error: {e}")
            # NATS cleanup
            try:
                for sub in list(agent._subs):
                    try:
                        await sub.unsubscribe()
                    except Exception:
                        pass
                agent._subs.clear()
                agent._handlers.clear()
            except Exception as e:
                logger.debug(f"WORKER {worker_id}: cleanup error: {e}")

        agent.handle_step_requested = _patched_handle_step

    # ------------------------------------------------------------------
    # Public API: connect + run
    # ------------------------------------------------------------------

    async def connect(self):
        """Connect the underlying agent to NATS (registers its subscriptions)."""
        await self._agent.connect()

    async def run(self, task_id: str, instruction: str):
        """Initialize state in KV and fire the first STEP_REQUESTED."""
        from src.core.events import Event, EventType  # noqa: PLC0415

        # Build initial state directly (bypass handle_action_requested to avoid
        # memory recall overhead on short-lived workers)
        augmented_instruction = instruction
        if self.context_summary:
            augmented_instruction = (
                f"[ORCHESTRATOR CONTEXT]\n{self.context_summary}\n\n"
                f"[YOUR SUB-TASK]\n{instruction}"
            )

        state = {
            "task_id": task_id,
            "assigned_actor": self._agent.name,
            "instruction": augmented_instruction,
            "history": [],
            "iteration": 0,
            "status": "running",
            "created_at": int(time.time()),
            "worker_id": self.worker_id,
            "parent_task_id": self.parent_task_id,
            "personality": self.personality_name,
        }
        await self._agent.state_store.put(f"actor_state.{task_id}", state)

        step_evt = Event(
            type=EventType.STEP_REQUESTED,
            source_actor=self._agent.name,
            correlation_id=task_id,
            payload={"task_id": task_id},
        )
        await self._agent._safe_publish("events.step.requested", step_evt)
