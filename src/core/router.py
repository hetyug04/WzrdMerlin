import logging
import os
from typing import Dict, Any, List
from src.core.actor import BaseActor
from src.core.events import Event, EventType

logger = logging.getLogger(__name__)

class DisCoRouter(BaseActor):
    """
    Asynchronous event-driven router replacing the LangGraph state machine.
    Listens to task events and dispatches action requests to the right actor queues.
    """
    def __init__(self, nats_url: str = None):
        super().__init__(name="disco-router", nats_url=nats_url)
        default_model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
        self.role_models = {
            "implementer": default_model,
            "auditor": default_model,
        }
    
    async def connect(self):
        await super().connect()
        # Register handlers keyed by the actual NATS subject strings
        self.on("events.task.created", self.handle_task_created)
        self.on("events.action.completed", self.handle_action_completed)
        self.on("events.capability.gap", self.handle_capability_gap)

        # Subscribe to subjects (must match the events.> stream)
        await self.listen("events.task.created")
        await self.listen("events.action.completed")
        await self.listen("events.capability.gap")
        logger.info("DisCoRouter listening for tasks and actions...")

    def distill_context(self, history: List[Dict[str, Any]]) -> str:
        """
        Compresses previous conversation/action loops into a dense summary.
        Placeholder for actual LLM-based summary distillation logic.
        """
        if not history:
            return ""
        return f"[Summarized Action State: {len(history)} previous events]"

    async def handle_task_created(self, event: Event):
        task_id = event.payload.get("task_id")
        description = event.payload.get("description")
        
        logger.info(f"Router received new task {task_id}: {description}")
        
        # Determine routing. For a new task, we usually start with the 'implementer'
        # In an emergent DisCo architecture, the Implementer handles planning natively.
        target_role = "implementer"
        
        # Dispatch action request
        action_req = Event(
            type=EventType.ACTION_REQUESTED,
            source_actor=self.name,
            target_actor=target_role,
            correlation_id=event.correlation_id or task_id,
            payload={
                "task_id": task_id,
                "model": self.role_models.get(target_role),
                "instruction": description,
                "history": [] # Start fresh
            }
        )
        
        await self.publish(f"events.actor.{target_role}", action_req)
        logger.info(f"Routed task {task_id} to {target_role}")

    async def handle_action_completed(self, event: Event):
        # Once an action completes, the router updates StateStore and decides 
        # what to do next (e.g., hand-off to auditor, or complete the task).
        pass
        
    async def handle_capability_gap(self, event: Event):
        gap = event.payload.get("gap_description")
        logger.warning(f"Router received CAPABILITY_GAP: {gap}. Queuing for self-improvement.")
        
        improvement_req = Event(
            type=EventType.IMPROVEMENT_QUEUED,
            source_actor=self.name,
            correlation_id=event.correlation_id,
            payload=event.payload
        )
        await self.publish("events.system.improvement", improvement_req)
