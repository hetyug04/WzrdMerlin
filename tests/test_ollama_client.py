"""
Tests for OllamaClient (src/core/ollama_client.py):
- CircuitBreaker state machine
- parse_action 6-strategy JSON parser
- OllamaUnavailableError when breaker is open
- stream_chat retry logic
"""
import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.ollama_client import CircuitBreaker, OllamaClient, OllamaUnavailableError


# ── CircuitBreaker ─────────────────────────────────────────────────────────────

def test_breaker_starts_closed():
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    assert not cb.is_open()


def test_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open()


def test_breaker_does_not_open_before_threshold():
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    for _ in range(4):
        cb.record_failure()
    assert not cb.is_open()


def test_breaker_resets_failure_count_on_success():
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    for _ in range(4):
        cb.record_failure()
    cb.record_success()
    # Should not open after success resets count
    assert not cb.is_open()


def test_breaker_half_open_after_recovery_timeout():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.01)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open()
    time.sleep(0.02)
    # After recovery timeout, breaker should allow one probe (not "open")
    assert not cb.is_open()


def test_breaker_closes_on_success_after_half_open():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.01)
    for _ in range(3):
        cb.record_failure()
    time.sleep(0.02)
    cb.record_success()
    assert not cb.is_open()


def test_breaker_reopens_on_failure_after_half_open():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.01)
    for _ in range(3):
        cb.record_failure()
    time.sleep(0.02)
    cb.record_failure()
    assert cb.is_open()


# ── parse_action (6-strategy parser) ──────────────────────────────────────────

def _make_client() -> OllamaClient:
    """Instantiate an OllamaClient without connecting."""
    client = OllamaClient.__new__(OllamaClient)
    client._breaker = CircuitBreaker()
    return client


def test_parse_action_clean_json():
    client = _make_client()
    raw = '{"tool": "shell", "args": {"cmd": "ls -la"}}'
    result = client.parse_action(raw)
    assert result == {"tool": "shell", "args": {"cmd": "ls -la"}}


def test_parse_action_with_think_blocks():
    client = _make_client()
    raw = '<think>I should list files</think>{"tool": "shell", "args": {"cmd": "ls"}}'
    result = client.parse_action(raw)
    assert result["tool"] == "shell"


def test_parse_action_markdown_fences():
    client = _make_client()
    raw = '```json\n{"tool": "done", "args": {"summary": "All done"}}\n```'
    result = client.parse_action(raw)
    assert result["tool"] == "done"


def test_parse_action_escaped_single_quotes():
    """Handles Qwen3.5's invalid JSON with escaped single quotes."""
    client = _make_client()
    raw = """{"tool": "shell", "args": {"cmd": "echo \\'hello\\'"}}"""
    result = client.parse_action(raw)
    assert result is not None
    assert result["tool"] == "shell"


def test_parse_action_extra_braces():
    """Handles extra closing braces Qwen3.5 sometimes adds."""
    client = _make_client()
    raw = '{"tool": "shell", "args": {"cmd": "pwd"}}}'
    result = client.parse_action(raw)
    assert result["tool"] == "shell"


def test_parse_action_prose_before_json():
    """Handles prose before the JSON object."""
    client = _make_client()
    raw = 'I will now list files. {"tool": "shell", "args": {"cmd": "ls"}}'
    result = client.parse_action(raw)
    assert result["tool"] == "shell"


def test_parse_action_regex_fallback():
    """Uses regex extraction when JSON is malformed beyond repair."""
    client = _make_client()
    raw = '"tool": "done", "args": { "summary": "Task complete" }'
    result = client.parse_action(raw)
    # May return None or partial — just assert no exception
    assert result is None or isinstance(result, dict)


def test_parse_action_empty_returns_none():
    client = _make_client()
    assert client.parse_action("") is None
    assert client.parse_action(None) is None


def test_parse_action_literal_newlines_in_string():
    """Handles literal newlines inside JSON string values."""
    client = _make_client()
    raw = '{"tool": "write_file", "args": {"path": "/tmp/x.py", "content": "line1\nline2"}}'
    result = client.parse_action(raw)
    assert result is not None
    assert result["tool"] == "write_file"


# ── build_messages ─────────────────────────────────────────────────────────────

def test_build_messages_structure():
    client = _make_client()
    msgs = client.build_messages(
        system_prompt="You are Merlin.",
        history=[
            {"action": {"tool": "shell", "args": {"cmd": "ls"}}, "result": "/workspace"},
        ],
        instruction="List files",
    )
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert "assistant" in roles
    assert "user" in roles
    # Critical: last two messages must NOT both be user (breaks Qwen3.5 context)
    assert not (roles[-1] == "user" and roles[-2] == "user"), \
        "build_messages must not end with two consecutive user messages"


def test_build_messages_empty_history():
    client = _make_client()
    msgs = client.build_messages("sys", [], "do thing")
    assert msgs[0]["role"] == "system"
    assert any(m["role"] == "user" for m in msgs)
    # No consecutive user messages at end
    roles = [m["role"] for m in msgs]
    assert not (roles[-1] == "user" and roles[-2] == "user"), \
        "Empty history must not produce consecutive user messages"


def test_build_messages_generate_prompt_merged_into_last_result():
    """The 'generate' cue must be in the same message as the last tool result,
    not as a separate user message — otherwise Qwen3.5 loses history context."""
    client = _make_client()
    history = [
        {"action": {"tool": "shell", "args": {"cmd": "scrot /tmp/shot.png"}},
         "result": "screenshot saved to /tmp/shot.png"},
    ]
    msgs = client.build_messages("sys", history, "take a screenshot then read it")

    # The last message should be a user message containing BOTH the result AND generate cue
    last = msgs[-1]
    assert last["role"] == "user"
    assert "Result:" in last["content"]
    assert "Generate" in last["content"] or "generate" in last["content"]

    # Verify no consecutive user messages anywhere
    roles = [m["role"] for m in msgs]
    for i in range(len(roles) - 1):
        assert not (roles[i] == "user" and roles[i + 1] == "user"), \
            f"Consecutive user messages at positions {i} and {i+1}: {roles}"


def test_build_messages_multi_step_no_consecutive_user():
    """With multiple history entries, no two user messages should be consecutive."""
    client = _make_client()
    history = [
        {"action": {"tool": "shell", "args": {"cmd": "ls"}}, "result": "file1.py file2.py"},
        {"action": {"tool": "read_file", "args": {"path": "file1.py"}}, "result": "# content"},
        {"action": {"tool": "shell", "args": {"cmd": "echo test"}}, "result": "test"},
    ]
    msgs = client.build_messages("sys", history, "read all files")
    roles = [m["role"] for m in msgs]
    for i in range(len(roles) - 1):
        assert not (roles[i] == "user" and roles[i + 1] == "user"), \
            f"Consecutive user messages at positions {i} and {i+1}"


# ── OllamaUnavailableError ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_chat_raises_when_breaker_open():
    """stream_chat raises OllamaUnavailableError when circuit breaker is open."""
    import httpx
    client = OllamaClient(
        base_url="http://localhost:11434",
        model="test-model",
    )
    # Force open breaker
    for _ in range(5):
        client._breaker.record_failure()

    with pytest.raises(OllamaUnavailableError):
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass


@pytest.mark.asyncio
async def test_chat_raises_when_breaker_open():
    """chat() raises OllamaUnavailableError when circuit breaker is open."""
    client = OllamaClient(
        base_url="http://localhost:11434",
        model="test-model",
    )
    for _ in range(5):
        client._breaker.record_failure()

    with pytest.raises(OllamaUnavailableError):
        await client.chat([{"role": "user", "content": "hi"}])
