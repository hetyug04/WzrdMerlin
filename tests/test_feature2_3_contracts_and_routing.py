import pytest
from unittest.mock import AsyncMock

from src.core.events import Event, EventType, validate_event_payload, ValidationError
from src.core.router import DisCoRouter


def test_validate_event_payload_task_created_contract():
    good = validate_event_payload(EventType.TASK_CREATED, {"task_id": "t1", "description": "do stuff"})
    assert good["task_id"] == "t1"

    with pytest.raises(ValidationError):
        validate_event_payload(EventType.TASK_CREATED, {"task_id": "t1"})


def test_router_policy_selects_auditor_for_review_tasks():
    router = DisCoRouter()
    actor, rationale = router._select_target_actor("Please audit and review this architecture for risks")
    assert actor == "agent-auditor"
    assert "policy" in rationale


def test_router_policy_selects_implementer_for_default_tasks():
    router = DisCoRouter()
    actor, rationale = router._select_target_actor("Create a small python script to read a file")
    assert actor == "agent-implementer"
    assert "default" in rationale


@pytest.mark.asyncio
async def test_handle_task_created_invalid_payload_emits_task_failed():
    router = DisCoRouter()
    router.publish = AsyncMock()

    bad_evt = Event(
        type=EventType.TASK_CREATED,
        source_actor="api",
        correlation_id="bad-task",
        payload={"task_id": "bad-task"},  # missing description
    )

    await router.handle_task_created(bad_evt)

    router.publish.assert_awaited_once()
    subject, evt = router.publish.await_args_list[0].args
    assert subject == "events.task.failed"
    assert evt.type == EventType.TASK_FAILED
    assert evt.payload["status"] == "failed"


@pytest.mark.asyncio
async def test_handle_task_created_emits_task_routed_and_action_requested():
    router = DisCoRouter()
    router.publish = AsyncMock()

    evt = Event(
        type=EventType.TASK_CREATED,
        source_actor="api",
        correlation_id="task-2",
        payload={"task_id": "task-2", "description": "Please review and analyze this design"},
    )

    await router.handle_task_created(evt)

    # Should publish task.routed and actor dispatch.
    assert router.publish.await_count == 2
    first_subject, first_event = router.publish.await_args_list[0].args
    second_subject, second_event = router.publish.await_args_list[1].args

    assert first_subject == "events.task.routed"
    assert first_event.type == EventType.TASK_ROUTED
    assert first_event.payload["target_actor"] == "agent-auditor"

    assert second_subject == "events.actor.agent-auditor"
    assert second_event.type == EventType.ACTION_REQUESTED
    assert second_event.payload["task_id"] == "task-2"
