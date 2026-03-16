import logging
import os
from typing import Dict, Any, List
from src.core.actor import BaseActor
from src.core.events import Event, EventType, validate_event_payload, ValidationError

logger = logging.getLogger(__name__)

class DisCoRouter(BaseActor):
    """
    Asynchronous event-driven Orchestrator.
    Implements Phase 3 Specialized Routing:
    - Performs initial classification and planning.
    - Manages task lifecycle transitions.
    - Routes to specialized agents (implementer, auditor).
    """
    def __init__(self, nats_url: str = None):
        super().__init__(name="disco-router", nats_url=nats_url)
        default_model = os.getenv("OLLAMA_MODEL", "merlin-model:latest")
        self.role_models = {
            "orchestrator": default_model,
            "implementer": default_model,
            "auditor": default_model,
        }

    async def connect(self):
        await super().connect()
        self.on("events.task.created", self.handle_task_created)
        self.on("events.action.completed", self.handle_action_completed)
        self.on("events.action.failed", self.handle_action_failed)
        self.on("events.capability.gap", self.handle_capability_gap)

        await self.listen("events.task.created")
        await self.listen("events.action.completed")
        await self.listen("events.action.failed")
        await self.listen("events.capability.gap")
        logger.info("Orchestrator listening for tasks and actions...")

    async def handle_task_created(self, event: Event):
        try:
            payload = validate_event_payload(EventType.TASK_CREATED, event.payload)
        except ValidationError as e:
            logger.error(f"ORCHESTRATOR: Invalid task.created payload: {e}")
            await self.publish("events.task.failed", Event(
                type=EventType.TASK_FAILED,
                source_actor=self.name,
                correlation_id=event.correlation_id,
                payload={
                    "task_id": event.payload.get("task_id", event.correlation_id),
                    "status": "failed",
                    "reason": "Invalid task payload",
                },
            ))
            return

        task_id = payload["task_id"]
        description = payload["description"]

        target_role, rationale = self._select_target_actor(description)
        logger.info(
            f"ORCHESTRATOR: Received task {task_id}, routed to {target_role} (reason: {rationale})"
        )

        await self.publish("events.task.routed", Event(
            type=EventType.TASK_ROUTED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={
                "task_id": task_id,
                "target_actor": target_role,
                "rationale": rationale,
            },
        ))

        action_req = Event(
            type=EventType.ACTION_REQUESTED,
            source_actor=self.name,
            target_actor=target_role,
            correlation_id=task_id,
            payload={
                "task_id": task_id,
                "model": self.role_models.get("implementer"),
                "instruction": description
            }
        )

        await self.publish(f"events.actor.{target_role}", action_req)

    def _select_target_actor(self, description: str) -> tuple[str, str]:
        """Policy router for initial actor selection."""
        text = (description or "").lower()

        auditor_terms = [
            "audit", "review", "analyze", "analysis", "compare", "evaluate",
            "inspect", "compliance", "security review", "architecture review",
            "root cause", "postmortem",
        ]
        if any(term in text for term in auditor_terms):
            return "agent-auditor", "keyword policy: analytical/review task"

        if len(text.split()) > 90:
            return "agent-auditor", "length policy: high-complexity request"

        return "agent-implementer", "default policy: execution task"

    async def handle_action_completed(self, event: Event):
        task_id = event.payload.get("task_id")
        result = event.payload.get("result")

        logger.info(f"Router: Action completed for task {task_id}.")

        completed_evt = Event(
            type=EventType.TASK_COMPLETED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={
                "task_id": task_id,
                "status": "completed",
                "result": result
            }
        )
        await self.publish("events.task.completed", completed_evt)

    async def handle_action_failed(self, event: Event):
        task_id = event.payload.get("task_id")
        reason = event.payload.get("reason")

        logger.error(f"Router: Action FAILED for task {task_id}: {reason}")

        failed_evt = Event(
            type=EventType.TASK_FAILED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={
                "task_id": task_id,
                "status": "failed",
                "reason": reason
            }
        )
        await self.publish("events.task.failed", failed_evt)

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
