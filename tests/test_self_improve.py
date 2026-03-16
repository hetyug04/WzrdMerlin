"""
Tests for the self-improvement loop:
  - ImprovementManager: worktree setup, patch application, validation, rollback
  - GardenerActor: fact extraction, dedup, idle detection, consolidation
"""
import asyncio
import json
import os
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from src.core.events import Event, EventType


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_improvement_event(gap="missing_tool_x", task="test-task-1"):
    return Event(
        type=EventType.CAPABILITY_GAP,
        source_actor="test-agent",
        correlation_id="corr-1",
        payload={"gap_description": gap, "triggering_task": task},
    )


def _make_gardener():
    from src.core.gardener import GardenerActor
    g = GardenerActor.__new__(GardenerActor)
    g.name = "gardener"
    g.nats_url = "nats://localhost:4222"
    g.nc = MagicMock()
    g.nc.is_connected = True
    g.js = None
    g._handlers = {}
    g._subs = []
    g._last_run = 0.0
    g._last_active_task = time.time()
    g._running = False
    g._llm = None
    from src.core.state import StateStore
    g.state_store = MagicMock(spec=StateStore)
    return g


def _make_improvement_manager(tmp_path):
    from src.core.self_improve import ImprovementManager
    mgr = ImprovementManager.__new__(ImprovementManager)
    mgr.name = "improvement-manager"
    mgr.nats_url = "nats://localhost:4222"
    mgr.nc = MagicMock()
    mgr.nc.is_connected = True
    mgr.nc.publish = AsyncMock()
    mgr.js = None
    mgr._handlers = {}
    mgr._subs = []
    mgr.repo_workspace = str(tmp_path)
    from src.core.state import StateStore
    mgr.state_store = MagicMock(spec=StateStore)
    return mgr


# ══════════════════════════════════════════════════════════════════════════════
# ImprovementManager tests
# ══════════════════════════════════════════════════════════════════════════════

class TestImprovementManagerFailureTracking:

    def test_record_failure_creates_log(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        mgr._record_failure("missing_tool_x", "patch_generation_failed")

        log_path = os.path.join(str(tmp_path), "improvement_failures.json")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            failures = json.load(f)
        assert len(failures) == 1
        assert failures[0]["gap_description"] == "missing_tool_x"
        assert failures[0]["reason"] == "patch_generation_failed"

    def test_record_failure_appends(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        mgr._record_failure("gap_a", "reason_a")
        mgr._record_failure("gap_b", "reason_b")

        log_path = os.path.join(str(tmp_path), "improvement_failures.json")
        with open(log_path) as f:
            failures = json.load(f)
        assert len(failures) == 2

    def test_record_failure_truncates_reason(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        long_reason = "x" * 1000
        mgr._record_failure("gap", long_reason)

        log_path = os.path.join(str(tmp_path), "improvement_failures.json")
        with open(log_path) as f:
            failures = json.load(f)
        assert len(failures[0]["reason"]) <= 500

    def test_record_failure_caps_at_100_entries(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        for i in range(110):
            mgr._record_failure(f"gap_{i}", f"reason_{i}")

        log_path = os.path.join(str(tmp_path), "improvement_failures.json")
        with open(log_path) as f:
            failures = json.load(f)
        assert len(failures) == 100
        # Oldest entries should be trimmed
        assert failures[0]["gap_description"] == "gap_10"


class TestImprovementManagerCooldown:

    def test_is_recently_failed_true(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        mgr._record_failure("gap_x", "failed")
        assert mgr._is_recently_failed("gap_x", cooldown_hours=24) is True

    def test_is_recently_failed_false_different_gap(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        mgr._record_failure("gap_x", "failed")
        assert mgr._is_recently_failed("gap_y", cooldown_hours=24) is False

    def test_is_recently_failed_false_no_log(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        assert mgr._is_recently_failed("anything") is False

    def test_is_recently_failed_expired(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        # Write a failure with an old timestamp
        log_path = os.path.join(str(tmp_path), "improvement_failures.json")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        old_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 86400 * 2))
        with open(log_path, "w") as f:
            json.dump([{"timestamp": old_ts, "gap_description": "gap_x", "reason": "r"}], f)
        assert mgr._is_recently_failed("gap_x", cooldown_hours=24) is False


class TestImprovementManagerRollback:

    def test_record_rollback_point(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="abc123def\n", returncode=0)
            result = mgr._record_rollback_point("improve-test123")

        assert result == "abc123def"
        rollback_path = os.path.join(os.getenv("MERLIN_WORKSPACE", "/workspace"), "rollback.json")
        # The rollback is written to ROLLBACK_LOG which uses MERLIN_WORKSPACE

    def test_rollback_caps_at_50(self, tmp_path):
        from src.core.self_improve import ROLLBACK_LOG
        os.makedirs(os.path.dirname(ROLLBACK_LOG), exist_ok=True)
        # Pre-fill with 50 entries
        entries = [{"timestamp": "2026-01-01T00:00:00", "branch": f"b-{i}", "pre_merge_hash": f"h{i}"} for i in range(50)]
        with open(ROLLBACK_LOG, "w") as f:
            json.dump(entries, f)

        mgr = _make_improvement_manager(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="newhash\n", returncode=0)
            mgr._record_rollback_point("improve-new")

        with open(ROLLBACK_LOG) as f:
            data = json.load(f)
        assert len(data) == 50  # capped, oldest trimmed


class TestImprovementManagerValidation:

    def test_validate_compile_failure(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        candidate = str(tmp_path / "candidate")
        os.makedirs(os.path.join(candidate, "src", "core"), exist_ok=True)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="SyntaxError: bad code")
            valid, log = mgr._validate_candidate(candidate)

        assert valid is False
        assert "Compile Error" in log

    def test_validate_test_failure(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        candidate = str(tmp_path / "candidate")
        os.makedirs(os.path.join(candidate, "tests"), exist_ok=True)

        compile_result = MagicMock(returncode=0, stderr="")
        test_result = MagicMock(returncode=1, stdout="FAILED test_x", stderr="")
        import_result = MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=[compile_result, test_result]):
            valid, log = mgr._validate_candidate(candidate)

        assert valid is False
        assert "Test Failure" in log

    def test_validate_all_pass(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        candidate = str(tmp_path / "candidate")
        os.makedirs(os.path.join(candidate, "tests"), exist_ok=True)

        ok = MagicMock(returncode=0, stdout="3 passed", stderr="")

        with patch("subprocess.run", return_value=ok):
            valid, log = mgr._validate_candidate(candidate)

        assert valid is True
        assert "compileall: passed" in log
        assert "import check: passed" in log


class TestImprovementManagerApplyPatch:

    @pytest.mark.asyncio
    async def test_apply_patch_writes_files(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        candidate = str(tmp_path / "candidate")
        os.makedirs(os.path.join(candidate, "src", "core"), exist_ok=True)

        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value=json.dumps({
            "files": [
                {"path": "src/core/new_tool.py", "content": "# new tool\nprint('hello')"}
            ],
            "test_file": {"path": "tests/test_new_tool.py", "content": "def test_it(): pass"},
            "description": "Added new tool",
        }))
        mock_ollama.parse_action = lambda raw: json.loads(raw)
        mock_ollama.aclose = AsyncMock()

        with patch("src.core.ollama_client.OllamaClient", return_value=mock_ollama), \
             patch("src.core.config.get_config") as mock_cfg, \
             patch("src.core.memory.get_memory") as mock_mem, \
             patch("subprocess.run"):
            mock_profile = MagicMock()
            mock_profile.model_name = "test-model"
            mock_profile.context_window = 4096
            mock_profile.temperature = 0.3
            mock_cfg.return_value.get_active_model.return_value = mock_profile
            mock_cfg.return_value.ollama.base_url = "http://localhost:11434"
            mock_mem_inst = AsyncMock()
            mock_mem_inst.recall = AsyncMock(return_value="")
            mock_mem.return_value = mock_mem_inst

            result = await mgr._apply_patch(candidate, "need tool X", "task-1")

        assert result is True
        assert os.path.exists(os.path.join(candidate, "src", "core", "new_tool.py"))
        assert os.path.exists(os.path.join(candidate, "tests", "test_new_tool.py"))

    @pytest.mark.asyncio
    async def test_apply_patch_empty_response_returns_false(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        candidate = str(tmp_path / "candidate")
        os.makedirs(candidate, exist_ok=True)

        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value="")
        mock_ollama.aclose = AsyncMock()

        with patch("src.core.ollama_client.OllamaClient", return_value=mock_ollama), \
             patch("src.core.config.get_config") as mock_cfg, \
             patch("src.core.memory.get_memory") as mock_mem:
            mock_profile = MagicMock()
            mock_profile.model_name = "test-model"
            mock_profile.context_window = 4096
            mock_profile.temperature = 0.3
            mock_cfg.return_value.get_active_model.return_value = mock_profile
            mock_cfg.return_value.ollama.base_url = "http://localhost:11434"
            mock_mem_inst = AsyncMock()
            mock_mem_inst.recall = AsyncMock(return_value="")
            mock_mem.return_value = mock_mem_inst

            result = await mgr._apply_patch(candidate, "need tool X", "task-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_apply_patch_injects_error_context_on_retry(self, tmp_path):
        mgr = _make_improvement_manager(tmp_path)
        candidate = str(tmp_path / "candidate")
        os.makedirs(os.path.join(candidate, "src", "core"), exist_ok=True)

        captured_prompts = []

        async def capture_chat(messages):
            captured_prompts.append(messages[-1]["content"])
            return json.dumps({
                "files": [{"path": "src/core/fix.py", "content": "# fix"}],
                "description": "retry fix",
            })

        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(side_effect=capture_chat)
        mock_ollama.parse_action = lambda raw: json.loads(raw)
        mock_ollama.aclose = AsyncMock()

        with patch("src.core.ollama_client.OllamaClient", return_value=mock_ollama), \
             patch("src.core.config.get_config") as mock_cfg, \
             patch("src.core.memory.get_memory") as mock_mem, \
             patch("subprocess.run"):
            mock_profile = MagicMock()
            mock_profile.model_name = "test-model"
            mock_profile.context_window = 4096
            mock_profile.temperature = 0.3
            mock_cfg.return_value.get_active_model.return_value = mock_profile
            mock_cfg.return_value.ollama.base_url = "http://localhost:11434"
            mock_mem_inst = AsyncMock()
            mock_mem_inst.recall = AsyncMock(return_value="")
            mock_mem.return_value = mock_mem_inst

            result = await mgr._apply_patch(
                candidate, "need tool X", "task-1",
                error_context="ImportError: no module named foo"
            )

        assert result is True
        assert "PREVIOUS ATTEMPT FAILED WITH" in captured_prompts[0]
        assert "ImportError" in captured_prompts[0]


# ══════════════════════════════════════════════════════════════════════════════
# GardenerActor tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGardenerFactExtraction:

    @pytest.mark.asyncio
    async def test_extract_facts_returns_list(self):
        g = _make_gardener()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value='["Fact one about the task", "Fact two about an error"]')
        g._llm = mock_llm

        trace = {
            "instruction": "Search for files matching pattern",
            "result": "Found 3 files",
            "history": [
                {"action": "tool:glob", "result": "file1.py, file2.py, file3.py"},
            ],
        }
        facts = await g._extract_facts(trace)
        assert len(facts) == 2
        assert "Fact one" in facts[0]

    @pytest.mark.asyncio
    async def test_extract_facts_no_llm_returns_empty(self):
        g = _make_gardener()
        g._llm = None
        facts = await g._extract_facts({"instruction": "test", "history": [{"action": "x"}]})
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_short_trace_returns_empty(self):
        g = _make_gardener()
        g._llm = AsyncMock()
        facts = await g._extract_facts({"instruction": "hi"})
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_filters_short_strings(self):
        g = _make_gardener()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value='["ok", "This is a real fact about something useful"]')
        g._llm = mock_llm

        trace = {
            "instruction": "Do something meaningful with the codebase",
            "result": "Completed successfully",
            "history": [{"action": "tool:read", "result": "file content here for testing"}],
        }
        facts = await g._extract_facts(trace)
        assert len(facts) == 1  # "ok" is < 10 chars, filtered out

    @pytest.mark.asyncio
    async def test_extract_facts_handles_markdown_wrapped_json(self):
        g = _make_gardener()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value='```json\n["Fact wrapped in markdown"]\n```')
        g._llm = mock_llm

        trace = {
            "instruction": "Process something with the codebase here",
            "result": "Done processing the thing",
            "history": [{"action": "tool:run", "result": "output data here for testing"}],
        }
        facts = await g._extract_facts(trace)
        assert len(facts) == 1
        assert "markdown" in facts[0]

    @pytest.mark.asyncio
    async def test_extract_facts_bad_json_returns_empty(self):
        g = _make_gardener()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="not valid json at all")
        g._llm = mock_llm

        trace = {
            "instruction": "Do something that produces bad LLM output",
            "result": "Completed",
            "history": [{"action": "tool:test", "result": "some result data here"}],
        }
        facts = await g._extract_facts(trace)
        assert facts == []


class TestGardenerDedup:

    @pytest.mark.asyncio
    async def test_is_duplicate_true(self):
        g = _make_gardener()
        mock_mem = AsyncMock()
        mock_mem.search = AsyncMock(return_value=[{"content": "existing fact", "score": 0.9}])
        result = await g._is_duplicate("existing fact", mock_mem)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_duplicate_false(self):
        g = _make_gardener()
        mock_mem = AsyncMock()
        mock_mem.search = AsyncMock(return_value=[])
        result = await g._is_duplicate("new fact", mock_mem)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_duplicate_handles_exception(self):
        g = _make_gardener()
        mock_mem = AsyncMock()
        mock_mem.search = AsyncMock(side_effect=Exception("search failed"))
        result = await g._is_duplicate("any fact", mock_mem)
        assert result is False


class TestGardenerTrajectory:

    def test_format_trajectory_valid(self):
        g = _make_gardener()
        trace = {
            "instruction": "Find all Python files",
            "result": "Found 10 files",
            "status": "completed",
            "history": [
                {"action": "glob *.py", "result": "10 files found"},
            ],
        }
        traj = g._format_trajectory("task-1", trace)
        assert traj is not None
        assert "Find all Python files" in traj
        assert "completed" in traj

    def test_format_trajectory_none_for_missing_instruction(self):
        g = _make_gardener()
        trace = {"result": "something", "history": [{"action": "x"}]}
        assert g._format_trajectory("task-1", trace) is None

    def test_format_trajectory_none_for_empty_history(self):
        g = _make_gardener()
        trace = {"instruction": "do X", "history": []}
        assert g._format_trajectory("task-1", trace) is None

    def test_format_trajectory_truncates_to_3000(self):
        g = _make_gardener()
        trace = {
            "instruction": "Big task",
            "result": "Big result",
            "status": "completed",
            "history": [{"action": "x" * 500, "result": "y" * 500} for _ in range(20)],
        }
        traj = g._format_trajectory("task-1", trace)
        assert traj is not None
        assert len(traj) <= 3000


class TestGardenerIdleDetection:

    def test_set_llm(self):
        g = _make_gardener()
        mock_llm = MagicMock()
        g.set_llm(mock_llm)
        assert g._llm is mock_llm

    @pytest.mark.asyncio
    async def test_task_activity_updates_timestamp(self):
        g = _make_gardener()
        old_ts = g._last_active_task
        await asyncio.sleep(0.01)
        await g._on_task_activity(Event(
            type=EventType.ACTION_COMPLETED,
            source_actor="agent",
            correlation_id="c1",
            payload={"task_id": "t1"},
        ))
        assert g._last_active_task > old_ts
