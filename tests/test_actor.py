"""
Tests for BaseActor v3 reliability fixes:
- publish() never re-raises
- publish() retries with 2/4/8s backoff
- _on_nats_reconnect() re-subscribes all handlers
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.core.events import Event, EventType
from src.core.actor import BaseActor


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_actor() -> BaseActor:
    actor = BaseActor.__new__(BaseActor)
    actor.name = "test-actor"
    actor.nats_url = "nats://localhost:4222"
    actor.nc = MagicMock()
    actor.nc.is_connected = True
    actor.js = None
    actor._handlers = {}
    actor._subs = []
    from src.core.state import StateStore
    actor.state_store = MagicMock(spec=StateStore)
    return actor


def _make_event() -> Event:
    return Event(
        type=EventType.TASK_CREATED,
        source_actor="test",
        correlation_id="c1",
        payload={"task_id": "t1", "description": "hello"},
    )


# ── publish() ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_publish_succeeds_first_attempt():
    """publish() calls nc.publish once on success and returns normally."""
    actor = _make_actor()
    actor.nc.publish = AsyncMock()
    event = _make_event()

    await actor.publish("events.task.created", event)

    actor.nc.publish.assert_called_once()


@pytest.mark.asyncio
async def test_publish_does_not_reraise_after_exhausting_retries():
    """publish() MUST NOT raise even after all retries are exhausted."""
    actor = _make_actor()
    actor.nc.publish = AsyncMock(side_effect=Exception("NATS down"))

    event = _make_event()
    # Should complete without raising — this is the critical fix from v2
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await actor.publish("events.task.created", event, retries=3)


@pytest.mark.asyncio
async def test_publish_retries_on_transient_failure():
    """publish() retries up to `retries` times before giving up."""
    actor = _make_actor()
    call_count = 0

    async def _fail_twice(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("transient error")

    actor.nc.publish = AsyncMock(side_effect=_fail_twice)
    event = _make_event()

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await actor.publish("events.task.created", event, retries=3)

    assert call_count == 3


@pytest.mark.asyncio
async def test_publish_disconnected_nats_does_not_raise():
    """publish() with disconnected NATS does not raise RuntimeError."""
    actor = _make_actor()
    actor.nc.is_connected = False

    event = _make_event()
    with patch("asyncio.sleep", new_callable=AsyncMock):
        # Must NOT raise
        await actor.publish("events.task.created", event, retries=1)


@pytest.mark.asyncio
async def test_publish_none_nc_does_not_raise():
    """publish() when nc is None does not raise."""
    actor = _make_actor()
    actor.nc = None

    event = _make_event()
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await actor.publish("events.task.created", event, retries=1)


# ── reconnect ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reconnect_resubscribes_registered_handlers():
    """_on_nats_reconnect() subscribes nc for each registered handler."""
    actor = _make_actor()

    mock_sub = MagicMock()
    actor.nc.subscribe = AsyncMock(return_value=mock_sub)

    handler_a = AsyncMock()
    handler_b = AsyncMock()
    actor._handlers = {
        "events.task.created": handler_a,
        "events.step.requested": handler_b,
    }

    await actor._on_nats_reconnect()

    assert actor.nc.subscribe.call_count == 2
    subjects = {c.args[0] for c in actor.nc.subscribe.call_args_list}
    assert "events.task.created" in subjects
    assert "events.step.requested" in subjects


@pytest.mark.asyncio
async def test_reconnect_clears_old_subs_before_resubscribing():
    """_on_nats_reconnect() clears _subs list first to avoid duplicates."""
    actor = _make_actor()

    old_sub = MagicMock()
    actor._subs = [old_sub]  # stale sub from before disconnect

    mock_sub = MagicMock()
    actor.nc.subscribe = AsyncMock(return_value=mock_sub)
    actor._handlers = {"events.task.created": AsyncMock()}

    await actor._on_nats_reconnect()

    assert old_sub not in actor._subs  # old sub was cleared


@pytest.mark.asyncio
async def test_reconnect_tolerates_subscribe_failure():
    """_on_nats_reconnect() logs errors but doesn't propagate subscribe failures."""
    actor = _make_actor()
    actor.nc.subscribe = AsyncMock(side_effect=Exception("subscribe failed"))
    actor._handlers = {"events.task.created": AsyncMock()}

    # Must not raise
    await actor._on_nats_reconnect()
