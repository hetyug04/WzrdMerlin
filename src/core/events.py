from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict, Optional, List
from enum import Enum
from datetime import datetime
import uuid

class EventType(str, Enum):
    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
    STEP_REQUESTED = "step.requested" # Trigger next iteration in a durable loop
    ACTION_REQUESTED = "action.requested"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"
    
    CAPABILITY_GAP = "capability.gap"
    IMPROVEMENT_QUEUED = "improvement.queued"
    IMPROVEMENT_DEPLOYED = "improvement.deployed"

    # Live agent narration — streamed directly to UI, not persisted in NATS
    AGENT_THINKING = "agent.thinking"   # <think> block content
    AGENT_STREAMING = "agent.streaming" # regular token stream
    AGENT_TOOL_START = "agent.tool_start"
    AGENT_TOOL_END = "agent.tool_end"

    SYSTEM_INFO = "system.info"
    SYSTEM_ERROR = "system.error"
    SYSTEM_HEARTBEAT = "system.heartbeat"
    TASK_ROUTED = "task.routed"

class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_actor: str
    target_actor: Optional[str] = None
    correlation_id: str
    payload: Dict[str, Any]

class TaskEventPayload(BaseModel):
    task_id: str
    description: str
    status: str
    result: Optional[str] = None

class ActionRequestPayload(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]


class TaskCreatedPayload(BaseModel):
    task_id: str
    description: str


class StepRequestedPayload(BaseModel):
    task_id: str


class ActorActionRequestPayload(BaseModel):
    task_id: str
    instruction: str
    model: Optional[str] = None
    
class CapabilityGapPayload(BaseModel):
    gap_description: str
    triggering_task: str
    priority: int


def validate_event_payload(event_type: EventType, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize payloads for critical event contracts.

    Raises ValidationError for invalid payloads.
    """
    if event_type == EventType.TASK_CREATED:
        return TaskCreatedPayload(**payload).model_dump()
    if event_type == EventType.STEP_REQUESTED:
        return StepRequestedPayload(**payload).model_dump()
    if event_type == EventType.ACTION_REQUESTED:
        return ActorActionRequestPayload(**payload).model_dump()
    return payload


__all__ = [
    "Event",
    "EventType",
    "TaskEventPayload",
    "ActionRequestPayload",
    "TaskCreatedPayload",
    "StepRequestedPayload",
    "ActorActionRequestPayload",
    "CapabilityGapPayload",
    "validate_event_payload",
    "ValidationError",
]
