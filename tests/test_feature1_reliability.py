import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.core.base_agent import BaseAgentActor
from src.core.events import Event, EventType


@pytest.mark.asyncio
async def test_fail_task_publishes_action_failed_subject_and_type():
    agent = BaseAgentActor()
    agent.publish = AsyncMock()

    await agent._fail_task("task-1", "boom")

    agent.publish.assert_awaited_once()
    subject, evt = agent.publish.await_args_list[0].args
    assert subject == "events.action.failed"
    assert evt.type == EventType.ACTION_FAILED
    assert evt.payload["task_id"] == "task-1"
    assert evt.payload["reason"] == "boom"


@pytest.mark.asyncio
async def test_step_requested_duplicate_event_id_is_deduped():
    agent = BaseAgentActor()
    agent.state_store = MagicMock()
    agent.state_store.get = AsyncMock(
        return_value={
            "task_id": "task-dup",
            "instruction": "do x",
            "history": [],
            "iteration": 0,
            "status": "running",
        }
    )
    agent.state_store.put = AsyncMock()
    agent._generate_single_action = AsyncMock(return_value=({"tool": "done", "args": {"summary": "ok"}}, ""))
    agent.execute_tool = AsyncMock(return_value="ok")
    agent.publish = AsyncMock()
    agent._ui_emit = AsyncMock()
    agent._save_memory = AsyncMock()

    evt = Event(
        type=EventType.STEP_REQUESTED,
        source_actor="test",
        correlation_id="task-dup",
        payload={"task_id": "task-dup"},
    )

    await agent.handle_step_requested(evt)
    await agent.handle_step_requested(evt)

    # Same event object => same event.id, should only execute once.
    assert agent._generate_single_action.await_count == 1


@pytest.mark.asyncio
async def test_step_requested_inflight_task_is_deduped():
    agent = BaseAgentActor()
    agent.state_store = MagicMock()
    agent.state_store.get = AsyncMock(
        return_value={
            "task_id": "task-inflight",
            "instruction": "do x",
            "history": [],
            "iteration": 0,
            "status": "running",
        }
    )
    agent.state_store.put = AsyncMock()

    gate = AsyncMock()

    async def slow_generate(*_args, **_kwargs):
        await gate()
        return {"tool": "done", "args": {"summary": "ok"}}, ""

    agent._generate_single_action = slow_generate
    agent.execute_tool = AsyncMock(return_value="ok")
    agent.publish = AsyncMock()
    agent._ui_emit = AsyncMock()
    agent._save_memory = AsyncMock()

    evt1 = Event(
        type=EventType.STEP_REQUESTED,
        source_actor="test",
        correlation_id="task-inflight",
        payload={"task_id": "task-inflight"},
    )
    evt2 = Event(
        type=EventType.STEP_REQUESTED,
        source_actor="test",
        correlation_id="task-inflight",
        payload={"task_id": "task-inflight"},
    )

    # Trigger first step and immediately send second one.
    t1 = asyncio.create_task(agent.handle_step_requested(evt1))
    await asyncio.sleep(0)
    await agent.handle_step_requested(evt2)
    await t1

    # Second should be skipped while first is inflight.
    assert gate.await_count == 1
