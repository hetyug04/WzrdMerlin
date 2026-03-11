from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
from enum import Enum
from datetime import datetime
import uuid

class EventType(str, Enum):
    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
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
    
class CapabilityGapPayload(BaseModel):
    gap_description: str
    triggering_task: str
    priority: int
