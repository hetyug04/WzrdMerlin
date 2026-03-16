"""
Tests for SQLiteMemoryStore (src/core/memory.py):
- add / search / recall / prune / count
- get_memory() factory returns SQLite store when chromadb absent
"""
import asyncio
import os
import sys
import pytest
import tempfile


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def store(db_path, monkeypatch):
    """Fresh SQLiteMemoryStore backed by a temp DB."""
    from src.core.memory import SQLiteMemoryStore
    s = SQLiteMemoryStore(db_path=db_path)
    return s


# ── add / search ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_and_search(store):
    """add() stores a document; search() retrieves it by keyword."""
    doc_id = await store.add(content="Python asyncio coroutines are fast", tags=["python"])
    assert doc_id

    results = await store.search("asyncio coroutines", top_k=5)
    assert len(results) >= 1
    assert any("asyncio" in r["content"].lower() for r in results)


@pytest.mark.asyncio
async def test_search_returns_empty_for_no_match(store):
    await store.add(content="Red foxes live in forests", tags=[])
    results = await store.search("quantum physics", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_search_top_k_limits_results(store):
    for i in range(10):
        await store.add(content=f"Document about Python topic {i}", tags=[])

    results = await store.search("Python", top_k=3)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_add_with_metadata(store):
    meta = {"source": "test", "importance": "high"}
    doc_id = await store.add(content="Important config note", metadata=meta)
    results = await store.search("config note", top_k=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_collections_are_isolated(store):
    """Documents in 'episodic' should not show in 'trajectories' search."""
    await store.add(content="Episodic memory: shell commands work", collection="episodic")
    await store.add(content="Trajectory: Task completed with shell", collection="trajectories")

    ep_results = await store.search("shell", collection="episodic")
    tr_results = await store.search("shell", collection="trajectories")

    ep_contents = [r["content"] for r in ep_results]
    assert any("Episodic" in c for c in ep_contents)
    assert not any("Trajectory" in c for c in ep_contents)


# ── recall ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recall_returns_formatted_string(store):
    await store.add(content="Use httpx for async HTTP calls in Python", tags=["http"])
    result = await store.recall("async http", top_k=3)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_recall_empty_returns_empty_string(store):
    result = await store.recall("nonexistent topic xyz123", top_k=3)
    assert result == ""


# ── prune ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prune_removes_old_entries(store):
    import time
    import sqlite3

    # Add a doc and then manually set its timestamp to be very old
    doc_id = await store.add(content="Old memory about API patterns", tags=[])
    old_ts = int(time.time()) - (100 * 86400)  # 100 days ago

    conn = sqlite3.connect(store.db_path)
    conn.execute("UPDATE memories SET timestamp = ? WHERE id = ?", (old_ts, doc_id))
    conn.commit()
    conn.close()

    removed = await store.prune(max_age_days=90)
    assert removed >= 1


@pytest.mark.asyncio
async def test_prune_keeps_recent_entries(store):
    await store.add(content="Recent memory about deployment", tags=[])
    removed = await store.prune(max_age_days=90)
    assert removed == 0
    assert store.count() == 1


# ── count ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_count_reflects_adds(store):
    assert store.count() == 0
    await store.add(content="First entry", tags=[])
    assert store.count() == 1
    await store.add(content="Second entry", tags=[])
    assert store.count() == 2


@pytest.mark.asyncio
async def test_count_separate_per_collection(store):
    await store.add(content="Episodic entry", collection="episodic")
    await store.add(content="Trajectory entry", collection="trajectories")
    assert store.count("episodic") == 1
    assert store.count("trajectories") == 1


# ── get_memory factory ─────────────────────────────────────────────────────────

def test_get_memory_returns_sqlite_when_chromadb_absent(monkeypatch, tmp_path):
    """get_memory() must fall back to SQLiteMemoryStore when chromadb is not installed."""
    import importlib
    import src.core.memory as mem_module

    # Clear singleton
    mem_module._memory_instance = None

    # Block chromadb import
    monkeypatch.setenv("MERLIN_WORKSPACE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "chromadb", None)

    mem = mem_module.get_memory()
    from src.core.memory import SQLiteMemoryStore
    assert isinstance(mem, SQLiteMemoryStore)

    # Reset singleton for other tests
    mem_module._memory_instance = None
