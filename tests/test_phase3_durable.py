import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.core.base_agent import BaseAgentActor
from src.core.events import Event, EventType

@pytest.mark.asyncio
async def test_actor_state_persistence():
    # Setup agent
    agent = BaseAgentActor(nats_url="nats://localhost:4222")
    agent.state_store = MagicMock()
    agent.state_store.put = AsyncMock()
    agent.state_store.get = AsyncMock()
    
    # Simulate a new task entry
    evt = Event(
        type=EventType.ACTION_REQUESTED,
        source_actor="disco-router",
        correlation_id="task-123",
        payload={"task_id": "task-123", "instruction": "Test instruction"}
    )
    
    with patch.object(agent, 'publish', new_callable=AsyncMock) as mock_publish:
        await agent.handle_action_requested(evt)
        
        # Verify initial state was persisted
        agent.state_store.put.assert_called_once()
        args = agent.state_store.put.call_args[0]
        assert args[0] == "actor_state.task-123"
        assert args[1]["instruction"] == "Test instruction"
        
        # Verify STEP_REQUESTED was published
        mock_publish.assert_called_once()
        pub_evt = mock_publish.call_args[0][1]
        assert pub_evt.type == EventType.STEP_REQUESTED

@pytest.mark.asyncio
async def test_observation_masking():
    agent = BaseAgentActor()
    history = [
        {"action": {"tool": "shell"}, "result": "Very long output 1"},
        {"action": {"tool": "shell"}, "result": "Very long output 2"},
        {"action": {"tool": "shell"}, "result": "Very long output 3"},
        {"action": {"tool": "shell"}, "result": "Very long output 4"},
        {"action": {"tool": "shell"}, "result": "Very long output 5"},
        {"action": {"tool": "shell"}, "result": "Very long output 6"},
    ]
    
    # Masking with keep_last=3
    masked = agent._mask_observations(history, keep_last=3)
    
    assert len(masked) == 6
    assert "Output Omitted" in masked[0]["result"]
    assert "Output Omitted" in masked[1]["result"]
    assert "Output Omitted" in masked[2]["result"]
    assert masked[3]["result"] == "Very long output 4"
    assert masked[5]["result"] == "Very long output 6"
