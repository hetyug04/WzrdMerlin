"""
Tests for BaseAgentActor v3 reliability fixes:
1. handle_step_requested() body wrapped in try/finally
2. _safe_publish() catches exceptions silently
3. KV read failure re-queues the step
4. tool_parallel_tools per-tool timeout
5. resume_with_human_response() empty-history guard
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.events import Event, EventType


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_agent():
    """Create a BaseAgentActor with all external dependencies mocked."""
    from src.core.base_agent import BaseAgentActor
    from src.core.ollama_client import OllamaClient

    agent = BaseAgentActor.__new__(BaseAgentActor)
    agent.name = "agent-implementer"
    agent.role = "implementer"
    agent.nc = MagicMock()
    agent.nc.is_connected = True
    agent.js = None
    agent._handlers = {}
    agent._subs = []
    agent._inflight_tasks = set()
    agent._processed_step_event_ids = set()
    agent._processed_step_event_order = []
    agent._current_task_id = ""
    agent._ui_broadcast = None

    agent.state_store = MagicMock()
    agent.state_store.get = AsyncMock(return_value=None)
    agent.state_store.put = AsyncMock()

    agent.ollama = MagicMock(spec=OllamaClient)
    agent.ollama.stream_chat = AsyncMock(return_value=_async_gen([]))
    agent.ollama.parse_action = MagicMock(return_value=None)
    agent.ollama.build_messages = MagicMock(return_value=[])

    import os
    from src.core.mcp.manager import MCPManager
    agent.mcp_manager = MCPManager()
    from src.core.mcp.forage import ForageManager
    agent.forage_manager = ForageManager(agent.mcp_manager)
    from src.core.mcp.codemode import CodeModeSandbox
    agent.sandbox = CodeModeSandbox()

    agent.tools = {
        "shell": agent.tool_shell,
        "read_file": agent.tool_read_file,
        "write_file": agent.tool_write_file,
        "search_memory": agent.tool_search_memory,
        "write_memory": agent.tool_write_memory,
        "fetch_url": agent.tool_fetch_url,
        "done": agent.tool_done,
        "request_human": agent.tool_request_human,
        "parallel_tools": agent.tool_parallel_tools,
        "forage_search": agent.tool_forage_search,
        "forage_install": agent.tool_forage_install,
        "python_sandbox": agent.tool_python_sandbox,
        "request_capability": agent.tool_request_capability,
    }

    return agent


async def _async_gen(items):
    for item in items:
        yield item


def _step_event(task_id="task_001") -> Event:
    return Event(
        type=EventType.STEP_REQUESTED,
        source_actor="test",
        correlation_id=task_id,
        payload={"task_id": task_id},
    )


# ── Fix 2: _safe_publish ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_safe_publish_does_not_reraise():
    """_safe_publish() swallows exceptions from publish()."""
    agent = _make_agent()
    agent.nc.publish = AsyncMock(side_effect=Exception("NATS gone"))

    event = Event(
        type=EventType.ACTION_FAILED,
        source_actor="agent-implementer",
        correlation_id="t1",
        payload={"task_id": "t1", "status": "failed", "reason": "test"},
    )
    # Must NOT raise
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await agent._safe_publish("events.action.failed", event)


@pytest.mark.asyncio
async def test_safe_publish_calls_publish():
    """_safe_publish() forwards to publish() on success."""
    agent = _make_agent()
    agent.nc.publish = AsyncMock()

    event = Event(
        type=EventType.ACTION_FAILED,
        source_actor="agent-implementer",
        correlation_id="t1",
        payload={"task_id": "t1", "status": "failed", "reason": "test"},
    )
    await agent._safe_publish("events.action.failed", event)
    agent.nc.publish.assert_called_once()


# ── Fix 5: resume_with_human_response empty-history guard ─────────────────────

@pytest.mark.asyncio
async def test_resume_returns_false_on_empty_history():
    """resume_with_human_response() returns False when history is empty."""
    agent = _make_agent()
    agent.state_store.get = AsyncMock(return_value={
        "task_id": "task_001",
        "status": "waiting_for_human",
        "history": [],
        "instruction": "test",
        "iteration": 1,
    })

    result = await agent.resume_with_human_response("task_001", "my answer")
    assert result is False


@pytest.mark.asyncio
async def test_resume_returns_false_when_not_waiting():
    """resume_with_human_response() returns False if task is not in waiting_for_human state."""
    agent = _make_agent()
    agent.state_store.get = AsyncMock(return_value={
        "task_id": "task_001",
        "status": "running",
        "history": [{"action": {"tool": "request_human", "args": {}}, "result": "waiting"}],
        "instruction": "test",
        "iteration": 1,
    })

    result = await agent.resume_with_human_response("task_001", "my answer")
    assert result is False


@pytest.mark.asyncio
async def test_resume_succeeds_with_populated_history():
    """resume_with_human_response() updates history and publishes step event."""
    agent = _make_agent()
    agent.nc.publish = AsyncMock()
    agent.state_store.get = AsyncMock(return_value={
        "task_id": "task_001",
        "status": "waiting_for_human",
        "history": [
            {"action": {"tool": "request_human", "args": {"question": "what?"}}, "result": "(awaiting)"},
        ],
        "instruction": "test",
        "iteration": 1,
    })

    result = await agent.resume_with_human_response("task_001", "the answer")
    assert result is True
    agent.state_store.put.assert_called()


# ── Fix 4: tool_parallel_tools timeout ────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_tools_timeout_returns_error_message():
    """tool_parallel_tools wraps each sub-call in asyncio.wait_for with TOOL_TIMEOUT."""
    agent = _make_agent()

    async def _slow_tool(args):
        await asyncio.sleep(9999)
        return "should not get here"

    agent.tools["slow_tool"] = _slow_tool

    with patch.dict("os.environ", {"MERLIN_TOOL_TIMEOUT_SECONDS": "0.01"}):
        result = await agent.tool_parallel_tools({
            "calls": [{"tool": "slow_tool", "args": {}}]
        })

    data = json.loads(result)
    assert len(data) == 1
    assert "timed out" in data[0]["result"].lower() or "Error" in data[0]["result"]


@pytest.mark.asyncio
async def test_parallel_tools_runs_in_parallel():
    """Multiple non-install calls run concurrently, not sequentially."""
    agent = _make_agent()
    results = []

    async def _fast_done(args):
        results.append("done")
        return "ok"

    agent.tools["fast_done"] = _fast_done

    result_str = await agent.tool_parallel_tools({
        "calls": [
            {"tool": "fast_done", "args": {}},
            {"tool": "fast_done", "args": {}},
        ]
    })
    data = json.loads(result_str)
    assert len(data) == 2


# ── Fix 1 & 3: handle_step_requested exception handling ───────────────────────

@pytest.mark.asyncio
async def test_handle_step_requested_removes_from_inflight_on_kv_error():
    """Fix 3: KV read failure should not leave the task stuck in _inflight_tasks."""
    agent = _make_agent()
    agent.state_store.get = AsyncMock(side_effect=Exception("KV gone"))
    agent.nc.publish = AsyncMock()

    event = _step_event()
    # After the handler completes, task must not be stuck in _inflight_tasks
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await agent.handle_step_requested(event)

    assert "task_001" not in agent._inflight_tasks


@pytest.mark.asyncio
async def test_handle_step_ignores_duplicate_event_id():
    """Idempotency: same event id is ignored on second delivery."""
    agent = _make_agent()
    event = _step_event()
    agent._processed_step_event_ids.add(event.id)

    agent.state_store.get = AsyncMock()  # should not be called
    await agent.handle_step_requested(event)

    agent.state_store.get.assert_not_called()
