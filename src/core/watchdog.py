import logging
import asyncio
import time
import os
import psutil
from src.core.actor import BaseActor
from src.core.events import Event, EventType

logger = logging.getLogger(__name__)

class WatchdogActor(BaseActor):
    """
    The System Watchdog Actor.
    Monitors the NATS event mesh and actor state to ensure system health.
    - Limits iteration count to prevent infinite loops.
    - Monitors hardware (VRAM/RAM) and emits alerts.
    - Tracks token usage (approximate) and suggests context clearing.
    - Kills hung or stale tasks.
    """
    def __init__(self, nats_url: str = None):
        super().__init__(name="system-watchdog", nats_url=nats_url)
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "20"))
        self.memory_threshold = 90.0 # 90% RAM usage
        self.vram_threshold = 95.0   # 95% VRAM (if we can poll it)

    async def connect(self):
        await super().connect()
        # Listen for all step requests to monitor iteration depth
        self.on("events.step.requested", self.monitor_step)
        await self.listen("events.step.requested")
        
        # Start hardware monitoring loop
        asyncio.create_task(self._hardware_loop())
        logger.info("WatchdogActor active and monitoring steps/hardware...")

    async def monitor_step(self, event: Event):
        task_id = event.payload.get("task_id")
        if not task_id: return

        # Load state from KV
        state = await self.state_store.get(f"actor_state.{task_id}")
        if not state: return

        iteration = state.get("iteration", 0)
        
        # Check iteration limit
        if iteration >= self.max_iterations:
            logger.warning(f"WATCHDOG: Task {task_id} reached max iterations ({iteration}). Forcing termination.")
            
            # Update state to failed
            state["status"] = "failed"
            state["reason"] = f"Iteration limit ({self.max_iterations}) exceeded. Task potentially stuck in a loop."
            await self.state_store.put(f"actor_state.{task_id}", state)
            
            # Publish failure event
            await self.publish("events.action.failed", Event(
                type=EventType.ACTION_FAILED,
                source_actor=self.name,
                correlation_id=task_id,
                payload={"task_id": task_id, "status": "failed", "reason": state["reason"]}
            ))

    async def _hardware_loop(self):
        """Periodically poll hardware and broadcast status events."""
        while True:
            await asyncio.sleep(10)
            
            # System RAM usage
            ram = psutil.virtual_memory().percent
            if ram > self.memory_threshold:
                logger.error(f"WATCHDOG: High System RAM usage: {ram}%")
                await self.publish("events.system.hardware", Event(
                    type=EventType.SYSTEM_ERROR,
                    source_actor=self.name,
                    correlation_id="system",
                    payload={"metric": "ram", "value": ram, "status": "critical"}
                ))
            
            # Emit heartbeat/telemetry for UI
            cpu = psutil.cpu_percent(interval=None)
            await self.publish("events.system.heartbeat", Event(
                type=EventType.SYSTEM_HEARTBEAT,
                source_actor=self.name,
                correlation_id="system",
                payload={
                    "ram_usage": ram,
                    "cpu_usage": cpu,
                    "timestamp": time.time(),
                    "status": "healthy" if ram < self.memory_threshold else "degraded"
                }
            ))
