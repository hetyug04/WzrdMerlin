"""
WzrdMerlin v3 — WatchdogActor

v3 changes: stripped inference_mgr telemetry (~30 lines removed).
Replaced with Ollama health poll via OllamaClient.ensure_model_loaded().
"""
import logging
import asyncio
import time
import os
from typing import Optional, TYPE_CHECKING

import psutil

from src.core.actor import BaseActor
from src.core.events import Event, EventType

if TYPE_CHECKING:
    from src.core.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class WatchdogActor(BaseActor):
    """
    System Watchdog Actor.
    - Limits iteration count to prevent infinite loops.
    - Monitors hardware (RAM/CPU) and emits alerts.
    - Polls Ollama health (replaces llama-server telemetry from v2).
    """

    def __init__(self, nats_url: str = None):
        super().__init__(name="system-watchdog", nats_url=nats_url)
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "20"))
        self.memory_threshold = 90.0
        self._ollama_client: Optional["OllamaClient"] = None

    async def connect(self):
        await super().connect()
        self.on("events.step.requested", self.monitor_step)
        await self.listen("events.step.requested")
        asyncio.create_task(self._hardware_loop())
        logger.info("WatchdogActor active and monitoring steps/hardware...")

    async def monitor_step(self, event: Event):
        task_id = event.payload.get("task_id")
        if not task_id:
            return
        state = await self.state_store.get(f"actor_state.{task_id}")
        if not state:
            return
        iteration = state.get("iteration", 0)
        if iteration >= self.max_iterations:
            logger.warning(
                f"WATCHDOG: Task {task_id} reached max iterations ({iteration}). Forcing termination."
            )
            state["status"] = "failed"
            state["reason"] = (
                f"Iteration limit ({self.max_iterations}) exceeded. Task potentially stuck."
            )
            await self.state_store.put(f"actor_state.{task_id}", state)
            await self.publish("events.action.failed", Event(
                type=EventType.ACTION_FAILED,
                source_actor=self.name,
                correlation_id=task_id,
                payload={"task_id": task_id, "status": "failed", "reason": state["reason"]},
            ))

    async def _hardware_loop(self):
        """Poll hardware and Ollama health every 10s; emit SYSTEM_HEARTBEAT."""
        while True:
            await asyncio.sleep(10)
            vm = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)

            if vm.percent > self.memory_threshold:
                logger.error(f"WATCHDOG: High System RAM usage: {vm.percent}%")
                await self.publish("events.system.hardware", Event(
                    type=EventType.SYSTEM_ERROR,
                    source_actor=self.name,
                    correlation_id="system",
                    payload={"metric": "ram", "value": vm.percent, "status": "critical"},
                ))

            # Ollama health (replaces llama-server telemetry from v2)
            ollama_ok = False
            if self._ollama_client:
                try:
                    ollama_ok = await self._ollama_client.ensure_model_loaded()
                except Exception:
                    pass

            heartbeat = Event(
                type=EventType.SYSTEM_HEARTBEAT,
                source_actor=self.name,
                correlation_id="system",
                payload={
                    "ram_usage": vm.percent,
                    "cpu_usage": cpu,
                    "ollama_healthy": ollama_ok,
                    "timestamp": time.time(),
                    "status": "healthy" if vm.percent < self.memory_threshold else "degraded",
                },
            )
            await self.publish("events.system.heartbeat", heartbeat)
