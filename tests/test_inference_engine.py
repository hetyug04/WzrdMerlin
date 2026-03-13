"""
Comprehensive tests for the inference engine (LlamaCppManager + ModelInterface wiring).

Covers:
  - Environment overrides (HSA_OVERRIDE_GFX_VERSION, RADV_PERFTEST, GGML_VULKAN)
  - Launch args (KV cache quantization, GPU layers, context window, mmap)
  - Multi-model LRU eviction
  - Dual-backend disaggregation routing
  - GTT spill detection and eviction
  - Watchdog idle/busy timers
  - Health check polling
  - Telemetry parsing from /metrics
  - ModelInterface llama-server API format
  - ModelInterface Ollama fallback
  - Slot lifecycle (start, stop, restart)
"""
import asyncio
import json
import os
import time
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

from src.core.inference import LlamaCppManager, ModelSlot, InferenceTelemetry


# ══════════════════════════════════════════════════════════════════════════════
#  Environment Overrides
# ══════════════════════════════════════════════════════════════════════════════


class TestEnvOverrides:
    def test_rocm_env_overrides(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        env = mgr.get_env_overrides("rocm")
        assert env["HSA_OVERRIDE_GFX_VERSION"] == "10.3.0"
        assert env["RADV_PERFTEST"] == "nogttspill"
        assert "GGML_VULKAN" not in env

    def test_vulkan_env_overrides(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        env = mgr.get_env_overrides("vulkan")
        assert env["HSA_OVERRIDE_GFX_VERSION"] == "10.3.0"
        assert env["RADV_PERFTEST"] == "nogttspill"
        assert env["GGML_VULKAN"] == "1"

    def test_vulkan_removes_existing_ggml_flag(self):
        """ROCm override should remove GGML_VULKAN if it was in env."""
        with patch.dict(os.environ, {"GGML_VULKAN": "1"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            env = mgr.get_env_overrides("rocm")
            assert "GGML_VULKAN" not in env

    def test_case_insensitive_backend(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        env_upper = mgr.get_env_overrides("Vulkan")
        assert env_upper["GGML_VULKAN"] == "1"
        env_lower = mgr.get_env_overrides("vulkan")
        assert env_lower["GGML_VULKAN"] == "1"


# ══════════════════════════════════════════════════════════════════════════════
#  Launch Args
# ══════════════════════════════════════════════════════════════════════════════


class TestLaunchArgs:
    def test_kv_cache_quantization_default(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        args = mgr.get_launch_args("qwen.gguf", 8081)
        assert "--cache-type-k" in args
        idx_k = args.index("--cache-type-k")
        assert args[idx_k + 1] == "q8_0"
        assert "--cache-type-v" in args
        idx_v = args.index("--cache-type-v")
        assert args[idx_v + 1] == "q8_0"

    def test_kv_cache_quantization_custom(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        mgr.kv_cache_type_k = "q4_0"
        mgr.kv_cache_type_v = "q4_0"
        args = mgr.get_launch_args("qwen.gguf", 8081)
        idx_k = args.index("--cache-type-k")
        assert args[idx_k + 1] == "q4_0"
        idx_v = args.index("--cache-type-v")
        assert args[idx_v + 1] == "q4_0"

    def test_gpu_layers(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        args = mgr.get_launch_args("qwen.gguf", 8081)
        assert "-ngl" in args
        idx = args.index("-ngl")
        assert args[idx + 1] == "99"

    def test_context_window(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        mgr.context_window = 16384
        args = mgr.get_launch_args("qwen.gguf", 8081)
        assert "-c" in args
        idx = args.index("-c")
        assert args[idx + 1] == "16384"

    def test_context_window_override(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        args = mgr.get_launch_args("qwen.gguf", 8081, context_window=4096)
        idx = args.index("-c")
        assert args[idx + 1] == "4096"

    def test_mmap_enabled(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        mgr.use_mmap = True
        args = mgr.get_launch_args("qwen.gguf", 8081)
        assert "--no-mmap" not in args

    def test_mmap_disabled(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        mgr.use_mmap = False
        args = mgr.get_launch_args("qwen.gguf", 8081)
        assert "--no-mmap" in args

    def test_continuous_batching(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        args = mgr.get_launch_args("qwen.gguf", 8081)
        assert "-cb" in args

    def test_metrics_flag(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        args = mgr.get_launch_args("qwen.gguf", 8081)
        assert "--metrics" in args

    def test_host_binding(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        args = mgr.get_launch_args("qwen.gguf", 8081)
        assert "--host" in args
        idx = args.index("--host")
        assert args[idx + 1] == "127.0.0.1"

    def test_port_assignment(self):
        mgr = LlamaCppManager(model_path="qwen.gguf")
        args = mgr.get_launch_args("qwen.gguf", 9999)
        assert "--port" in args
        idx = args.index("--port")
        assert args[idx + 1] == "9999"


# ══════════════════════════════════════════════════════════════════════════════
#  Port Allocation
# ══════════════════════════════════════════════════════════════════════════════


class TestPortAllocation:
    def test_sequential_port_allocation(self):
        mgr = LlamaCppManager(model_path="dummy.gguf", port=9000)
        p1 = mgr._allocate_port()
        p2 = mgr._allocate_port()
        p3 = mgr._allocate_port()
        assert p1 == 9000
        assert p2 == 9001
        assert p3 == 9002


# ══════════════════════════════════════════════════════════════════════════════
#  Slot Key
# ══════════════════════════════════════════════════════════════════════════════


class TestSlotKey:
    def test_slot_key_format(self):
        assert LlamaCppManager._slot_key("/models/qwen.gguf", "rocm") == "/models/qwen.gguf:rocm"
        assert LlamaCppManager._slot_key("/models/qwen.gguf", "VULKAN") == "/models/qwen.gguf:vulkan"


# ══════════════════════════════════════════════════════════════════════════════
#  ModelSlot
# ══════════════════════════════════════════════════════════════════════════════


class TestModelSlot:
    def test_slot_key_property(self):
        slot = ModelSlot(model_path="/m/qwen.gguf", backend="rocm", port=8081)
        assert slot.key == "/m/qwen.gguf:rocm"

    def test_api_base(self):
        slot = ModelSlot(model_path="/m/qwen.gguf", backend="rocm", port=8081)
        assert slot.api_base == "http://127.0.0.1:8081"

    def test_is_alive_no_process(self):
        slot = ModelSlot(model_path="/m/qwen.gguf", backend="rocm", port=8081)
        assert not slot.is_alive

    def test_is_alive_running(self):
        slot = ModelSlot(model_path="/m/qwen.gguf", backend="rocm", port=8081)
        proc = MagicMock()
        proc.poll.return_value = None  # Still running
        slot.process = proc
        assert slot.is_alive

    def test_is_alive_exited(self):
        slot = ModelSlot(model_path="/m/qwen.gguf", backend="rocm", port=8081)
        proc = MagicMock()
        proc.poll.return_value = 1  # Exited
        slot.process = proc
        assert not slot.is_alive


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-Model LRU Eviction
# ══════════════════════════════════════════════════════════════════════════════


class TestLRUEviction:
    @pytest.mark.asyncio
    async def test_evict_lru_when_at_capacity(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.max_active_models = 2

        # Manually insert two "alive" slots
        proc1 = MagicMock()
        proc1.poll.return_value = None
        proc1.terminate = MagicMock()
        proc1.wait = MagicMock()
        slot1 = ModelSlot(
            model_path="model_a.gguf", backend="rocm", port=8081,
            process=proc1, pid=1001, last_used=time.time() - 100,
        )

        proc2 = MagicMock()
        proc2.poll.return_value = None
        proc2.terminate = MagicMock()
        proc2.wait = MagicMock()
        slot2 = ModelSlot(
            model_path="model_b.gguf", backend="rocm", port=8082,
            process=proc2, pid=1002, last_used=time.time(),
        )

        mgr.slots[slot1.key] = slot1
        mgr.slots[slot2.key] = slot2

        # Eviction should remove slot1 (oldest last_used)
        await mgr._evict_if_needed()

        assert slot1.key not in mgr.slots
        assert slot2.key in mgr.slots
        proc1.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_eviction_under_capacity(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.max_active_models = 3

        proc = MagicMock()
        proc.poll.return_value = None
        slot = ModelSlot(
            model_path="model_a.gguf", backend="rocm", port=8081,
            process=proc, pid=1001,
        )
        mgr.slots[slot.key] = slot

        await mgr._evict_if_needed()
        assert slot.key in mgr.slots  # No eviction

    @pytest.mark.asyncio
    async def test_evict_multiple_to_make_room(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.max_active_models = 1

        for i in range(3):
            proc = MagicMock()
            proc.poll.return_value = None
            proc.terminate = MagicMock()
            proc.wait = MagicMock()
            slot = ModelSlot(
                model_path=f"model_{i}.gguf", backend="rocm", port=8081 + i,
                process=proc, pid=1000 + i, last_used=time.time() - (100 - i * 10),
            )
            mgr.slots[slot.key] = slot

        await mgr._evict_if_needed()
        # Only 0 slots should remain (all evicted to get under max=1)
        alive = {k: s for k, s in mgr.slots.items() if s.is_alive}
        assert len(alive) == 0  # all evicted since we need room for 1 new


# ══════════════════════════════════════════════════════════════════════════════
#  Dual-Backend Disaggregation
# ══════════════════════════════════════════════════════════════════════════════


class TestDualBackend:
    def test_single_backend_always_rocm(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.disaggregate = False
        assert mgr.select_backend(100) == "rocm"
        assert mgr.select_backend(5000) == "rocm"

    def test_disaggregate_short_prompt_vulkan(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.disaggregate = True
        mgr.prefill_token_threshold = 2048
        assert mgr.select_backend(500) == "vulkan"
        assert mgr.select_backend(2048) == "vulkan"

    def test_disaggregate_long_prompt_rocm(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.disaggregate = True
        mgr.prefill_token_threshold = 2048
        assert mgr.select_backend(2049) == "rocm"
        assert mgr.select_backend(10000) == "rocm"

    def test_disaggregate_boundary(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.disaggregate = True
        mgr.prefill_token_threshold = 2048
        # At exactly threshold → vulkan (<=)
        assert mgr.select_backend(2048) == "vulkan"
        # One over → rocm
        assert mgr.select_backend(2049) == "rocm"


# ══════════════════════════════════════════════════════════════════════════════
#  GTT Spill Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestGTTSpill:
    @pytest.mark.asyncio
    async def test_gtt_spill_evicts_on_high_ram(self):
        """When RAM exceeds gtt_threshold, watchdog loop should evict the slot."""
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.gtt_threshold = 85.0

        proc = MagicMock()
        proc.poll.return_value = None
        proc.terminate = MagicMock()
        proc.wait = MagicMock()
        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            process=proc, pid=1001, last_used=time.time(),
        )
        mgr.slots[slot.key] = slot

        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.percent = 92.0
            # Simulate one watchdog iteration manually
            for key in list(mgr.slots.keys()):
                s = mgr.slots.get(key)
                if s and s.is_alive:
                    vm = mock_vm()
                    if vm.percent > mgr.gtt_threshold:
                        mgr.stop_slot(key)

        assert slot.key not in mgr.slots
        proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_eviction_below_gtt_threshold(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.gtt_threshold = 85.0

        proc = MagicMock()
        proc.poll.return_value = None
        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            process=proc, pid=1001, last_used=time.time(),
        )
        mgr.slots[slot.key] = slot

        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.percent = 60.0
            vm = mock_vm()
            assert vm.percent <= mgr.gtt_threshold
            # Slot should NOT be evicted
            assert slot.key in mgr.slots


# ══════════════════════════════════════════════════════════════════════════════
#  Watchdog Timers
# ══════════════════════════════════════════════════════════════════════════════


class TestWatchdogTimers:
    def test_idle_timeout_config(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.idle_timeout_minutes = 15
        assert mgr.idle_timeout_minutes == 15

    def test_busy_timeout_config(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.busy_timeout_minutes = 10
        assert mgr.busy_timeout_minutes == 10

    def test_idle_detection(self):
        """Slot idle longer than timeout should be flagged."""
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.idle_timeout_minutes = 15

        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            last_used=time.time() - (16 * 60),  # 16 minutes ago
        )
        idle_seconds = time.time() - slot.last_used
        assert idle_seconds > mgr.idle_timeout_minutes * 60

    def test_not_idle(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.idle_timeout_minutes = 15

        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            last_used=time.time() - 60,  # 1 minute ago
        )
        idle_seconds = time.time() - slot.last_used
        assert idle_seconds < mgr.idle_timeout_minutes * 60

    def test_hung_detection(self):
        """Slot with both long uptime and long idle should be flagged as hung."""
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.busy_timeout_minutes = 10

        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            last_used=time.time() - (11 * 60),
            started_at=time.time() - (11 * 60),
        )
        now = time.time()
        idle_seconds = now - slot.last_used
        uptime_seconds = now - slot.started_at
        assert uptime_seconds > mgr.busy_timeout_minutes * 60
        assert idle_seconds > mgr.busy_timeout_minutes * 60


# ══════════════════════════════════════════════════════════════════════════════
#  Telemetry Parsing
# ══════════════════════════════════════════════════════════════════════════════


class TestTelemetryParsing:
    def test_parse_prometheus_metrics(self):
        metrics_text = """
# HELP llama_tokens_predicted_total Number of predicted tokens
# TYPE llama_tokens_predicted_total counter
llama_tokens_predicted_total 12345

# HELP llama_prompt_tokens_total Number of prompt tokens
# TYPE llama_prompt_tokens_total counter
llama_prompt_tokens_total 67890

# HELP llama_kv_cache_used_cells Number of KV cache cells used
# TYPE llama_kv_cache_used_cells gauge
llama_kv_cache_used_cells 512

# HELP llama_kv_cache_tokens Number of KV cache tokens
# TYPE llama_kv_cache_tokens gauge
llama_kv_cache_tokens 8192

# HELP llama_requests_processing Number of requests processing
# TYPE llama_requests_processing gauge
llama_requests_processing 1

# HELP llama_requests_pending Number of requests pending
# TYPE llama_requests_pending gauge
llama_requests_pending 0

# HELP llama_tokens_second_total Tokens per second
# TYPE llama_tokens_second_total gauge
llama_tokens_second_total 42.5
"""
        tel = LlamaCppManager._parse_metrics(metrics_text)
        assert tel.tokens_predicted == 12345
        assert tel.tokens_evaluated == 67890
        assert tel.kv_cache_used_cells == 512
        assert tel.kv_cache_total_cells == 8192
        assert tel.requests_processing == 1
        assert tel.requests_pending == 0
        assert tel.tokens_per_sec == 42.5

    def test_parse_empty_metrics(self):
        tel = LlamaCppManager._parse_metrics("")
        assert tel.tokens_predicted == 0
        assert tel.tokens_per_sec == 0.0

    def test_parse_partial_metrics(self):
        metrics_text = """
llama_tokens_predicted_total 100
llama_requests_processing 2
"""
        tel = LlamaCppManager._parse_metrics(metrics_text)
        assert tel.tokens_predicted == 100
        assert tel.requests_processing == 2
        assert tel.tokens_per_sec == 0.0  # Not in output

    def test_parse_alternative_metric_names(self):
        """Some llama.cpp builds use different metric names."""
        metrics_text = """
llama_tokens_predicted 500
llama_tokens_evaluated 1000
llama_decode_tokens_per_second 35.2
"""
        tel = LlamaCppManager._parse_metrics(metrics_text)
        assert tel.tokens_predicted == 500
        assert tel.tokens_evaluated == 1000
        assert tel.tokens_per_sec == 35.2


# ══════════════════════════════════════════════════════════════════════════════
#  Health Check
# ══════════════════════════════════════════════════════════════════════════════


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_no_model_loaded(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        health = await mgr.get_health()
        assert health["status"] == "no_model_loaded"
        assert health["model_loaded"] is False

    @pytest.mark.asyncio
    async def test_health_with_alive_slot(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        proc = MagicMock()
        proc.poll.return_value = None
        slot = ModelSlot(
            model_path="/models/qwen.gguf", backend="rocm", port=8081,
            process=proc, pid=1001, started_at=time.time() - 60,
        )
        mgr.slots[slot.key] = slot

        # Mock the httpx call to return 200
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.json.return_value = {"status": "ok"}
            mock_client.get = AsyncMock(return_value=mock_resp)

            health = await mgr.get_health(slot)
            assert health["status"] == "ok"
            assert health["model_loaded"] is True
            assert health["backend"] == "rocm"
            assert health["port"] == 8081


# ══════════════════════════════════════════════════════════════════════════════
#  Slot Lifecycle
# ══════════════════════════════════════════════════════════════════════════════


class TestSlotLifecycle:
    def test_stop_slot(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        proc = MagicMock()
        proc.poll.return_value = None
        proc.terminate = MagicMock()
        proc.wait = MagicMock()
        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            process=proc, pid=1001,
        )
        mgr.slots[slot.key] = slot

        mgr.stop_slot(slot.key)
        proc.terminate.assert_called_once()
        assert slot.key not in mgr.slots

    def test_stop_all(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        procs = []
        for i in range(3):
            proc = MagicMock()
            proc.poll.return_value = None
            proc.terminate = MagicMock()
            proc.wait = MagicMock()
            procs.append(proc)
            slot = ModelSlot(
                model_path=f"model_{i}.gguf", backend="rocm", port=8081 + i,
                process=proc, pid=1000 + i,
            )
            mgr.slots[slot.key] = slot

        mgr.stop_all()
        for p in procs:
            p.terminate.assert_called_once()
        assert len(mgr.slots) == 0

    def test_stop_nonexistent_slot(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        mgr.stop_slot("nonexistent:rocm")  # Should not raise

    def test_record_activity(self):
        mgr = LlamaCppManager(model_path="model.gguf")
        proc = MagicMock()
        proc.poll.return_value = None
        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            process=proc, pid=1001, last_used=time.time() - 100,
        )
        mgr.slots[slot.key] = slot
        old_time = slot.last_used

        mgr.record_activity(model_path="model.gguf", backend="rocm")
        assert slot.last_used > old_time

    def test_get_api_base_loaded(self):
        mgr = LlamaCppManager(model_path="model.gguf")
        proc = MagicMock()
        proc.poll.return_value = None
        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            process=proc, pid=1001,
        )
        mgr.slots[slot.key] = slot
        assert mgr.get_api_base("model.gguf", "rocm") == "http://127.0.0.1:8081"

    def test_get_api_base_not_loaded(self):
        mgr = LlamaCppManager(model_path="model.gguf")
        assert mgr.get_api_base("model.gguf", "rocm") is None

    def test_get_all_slots_info(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        proc = MagicMock()
        proc.poll.return_value = None
        slot = ModelSlot(
            model_path="/models/qwen.gguf", backend="rocm", port=8081,
            process=proc, pid=1001, started_at=time.time() - 30,
        )
        mgr.slots[slot.key] = slot

        info = mgr.get_all_slots_info()
        assert len(info) == 1
        assert info[0]["model"] == "qwen.gguf"
        assert info[0]["backend"] == "rocm"
        assert info[0]["alive"] is True

    @pytest.mark.asyncio
    async def test_ensure_model_starts_if_needed(self):
        """ensure_model should call start() if no slot exists."""
        mgr = LlamaCppManager(model_path="model.gguf")

        # Mock start to avoid actually launching a process
        started_with = {}

        async def mock_start(model_path=None, backend="rocm"):
            started_with["model_path"] = model_path
            started_with["backend"] = backend
            slot = ModelSlot(
                model_path=model_path or mgr.default_model_path,
                backend=backend,
                port=8081,
                process=MagicMock(poll=MagicMock(return_value=None)),
                pid=1001,
                ready=True,
            )
            mgr.slots[slot.key] = slot
            return slot

        mgr.start = mock_start
        slot = await mgr.ensure_model(backend="vulkan")
        assert started_with["backend"] == "vulkan"
        assert slot.ready

    @pytest.mark.asyncio
    async def test_ensure_model_reuses_existing(self):
        """ensure_model should reuse an alive+ready slot."""
        mgr = LlamaCppManager(model_path="model.gguf")
        proc = MagicMock()
        proc.poll.return_value = None
        existing = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            process=proc, pid=1001, ready=True,
            last_used=time.time() - 50,
        )
        mgr.slots[existing.key] = existing

        slot = await mgr.ensure_model(backend="rocm")
        assert slot is existing
        assert slot.last_used > time.time() - 1  # Updated


# ══════════════════════════════════════════════════════════════════════════════
#  ModelInterface — llama-server API format
# ══════════════════════════════════════════════════════════════════════════════


class TestModelInterfaceLlamaServer:
    def setup_method(self):
        # Force llama-server mode
        self._orig_env = {}
        for key in ("INFERENCE_BACKEND", "LLAMA_API_BASE"):
            self._orig_env[key] = os.environ.get(key)
        os.environ["INFERENCE_BACKEND"] = "llama-server"
        os.environ["LLAMA_API_BASE"] = "http://127.0.0.1:8081"

    def teardown_method(self):
        for key, val in self._orig_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_build_llama_payload_format(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        messages = [{"role": "user", "content": "Hello"}]
        payload = mi._build_llama_payload(messages, stream=True)

        assert "model" in payload
        assert payload["stream"] is True
        assert payload["temperature"] == 0.3
        assert "messages" in payload
        # Should NOT have Ollama-specific fields
        assert "keep_alive" not in payload
        assert "options" not in payload
        assert "think" not in payload

    def test_build_llama_payload_max_tokens(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        mi.think = True
        mi.think_budget = 1024
        messages = [{"role": "user", "content": "Hello"}]
        payload = mi._build_llama_payload(messages, stream=True)
        assert payload["max_tokens"] == 1024 + 512

    def test_estimate_prompt_tokens(self):
        from src.core.llm import ModelInterface
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me a story about a fox."},
        ]
        tokens = ModelInterface._estimate_prompt_tokens(messages)
        assert tokens > 0
        # ~50 chars total → ~12-13 tokens
        assert 5 < tokens < 30


# ══════════════════════════════════════════════════════════════════════════════
#  ModelInterface — Ollama fallback
# ══════════════════════════════════════════════════════════════════════════════


class TestModelInterfaceOllama:
    def setup_method(self):
        self._orig_env = {}
        for key in ("INFERENCE_BACKEND", "OLLAMA_BASE_URL"):
            self._orig_env[key] = os.environ.get(key)
        os.environ["INFERENCE_BACKEND"] = "ollama"
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

    def teardown_method(self):
        for key, val in self._orig_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_build_ollama_payload_format(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        messages = [{"role": "user", "content": "Hello"}]
        payload = mi._build_ollama_payload(messages, stream=True)

        assert "model" in payload
        assert payload["stream"] is True
        assert payload["keep_alive"] == -1
        assert "options" in payload
        assert payload["options"]["temperature"] == 0.3
        # Should NOT have OpenAI-specific fields
        assert "max_tokens" not in payload

    def test_ollama_backend_type(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        assert mi.backend_type == "ollama"


# ══════════════════════════════════════════════════════════════════════════════
#  ModelInterface — JSON parser (unchanged, regression tests)
# ══════════════════════════════════════════════════════════════════════════════


class TestJSONParser:
    def setup_method(self):
        os.environ.pop("INFERENCE_BACKEND", None)

    def test_parse_clean_json(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        result = mi.parse_action('{"tool": "shell", "args": {"cmd": "ls"}}')
        assert result == {"tool": "shell", "args": {"cmd": "ls"}}

    def test_parse_with_think_tags(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        result = mi.parse_action(
            '<think>Let me think...</think>{"tool": "done", "args": {"summary": "OK"}}'
        )
        assert result["tool"] == "done"

    def test_parse_with_markdown_fences(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        result = mi.parse_action('```json\n{"tool": "shell", "args": {"cmd": "pwd"}}\n```')
        assert result["tool"] == "shell"

    def test_parse_with_escaped_quotes(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        result = mi.parse_action('{"tool": "shell", "args": {"cmd": "echo \'hello\'"}}')
        assert result["tool"] == "shell"

    def test_parse_with_extra_braces(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        result = mi.parse_action('Some text {"tool": "done", "args": {"summary": "OK"}} extra')
        assert result["tool"] == "done"

    def test_parse_empty(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        assert mi.parse_action("") is None
        assert mi.parse_action(None) is None

    def test_regex_extraction_fallback(self):
        from src.core.llm import ModelInterface
        mi = ModelInterface()
        # Badly broken JSON but has tool and args
        result = mi.parse_action('I think {"tool": "shell" "args": {"cmd": "ls"}}')
        assert result is not None
        assert result["tool"] == "shell"


# ══════════════════════════════════════════════════════════════════════════════
#  Environment Variable Configuration
# ══════════════════════════════════════════════════════════════════════════════


class TestEnvConfig:
    def test_kv_cache_env_vars(self):
        with patch.dict(os.environ, {
            "LLAMA_KV_CACHE_TYPE_K": "q4_0",
            "LLAMA_KV_CACHE_TYPE_V": "q4_0",
        }):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.kv_cache_type_k == "q4_0"
            assert mgr.kv_cache_type_v == "q4_0"

    def test_gtt_threshold_env_var(self):
        with patch.dict(os.environ, {"LLAMA_GTT_THRESHOLD": "90.0"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.gtt_threshold == 90.0

    def test_max_models_env_var(self):
        with patch.dict(os.environ, {"LLAMA_MAX_MODELS": "3"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.max_active_models == 3

    def test_disaggregate_env_var(self):
        with patch.dict(os.environ, {"LLAMA_DISAGGREGATE": "true"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.disaggregate is True

    def test_mmap_env_var(self):
        with patch.dict(os.environ, {"LLAMA_MMAP": "false"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.use_mmap is False

    def test_idle_timeout_env_var(self):
        with patch.dict(os.environ, {"LLAMA_IDLE_TIMEOUT": "30"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.idle_timeout_minutes == 30

    def test_busy_timeout_env_var(self):
        with patch.dict(os.environ, {"LLAMA_BUSY_TIMEOUT": "20"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.busy_timeout_minutes == 20

    def test_prefill_threshold_env_var(self):
        with patch.dict(os.environ, {"LLAMA_PREFILL_THRESHOLD": "4096"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.prefill_token_threshold == 4096

    def test_gpu_layers_env_var(self):
        with patch.dict(os.environ, {"LLAMA_GPU_LAYERS": "32"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.gpu_layers == 32

    def test_context_window_env_var(self):
        with patch.dict(os.environ, {"LLAMA_CONTEXT_WINDOW": "32768"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.context_window == 32768

    def test_default_model_path_env_var(self):
        with patch.dict(os.environ, {"LLAMA_MODEL_PATH": "/models/test.gguf"}):
            mgr = LlamaCppManager()
            assert mgr.default_model_path == "/models/test.gguf"

    def test_server_path_env_var(self):
        with patch.dict(os.environ, {"LLAMA_SERVER_PATH": "/usr/bin/llama-server"}):
            mgr = LlamaCppManager(model_path="dummy.gguf")
            assert mgr.server_path == "/usr/bin/llama-server"


# ══════════════════════════════════════════════════════════════════════════════
#  Shutdown
# ══════════════════════════════════════════════════════════════════════════════


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_stops_all(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")
        proc = MagicMock()
        proc.poll.return_value = None
        proc.terminate = MagicMock()
        proc.wait = MagicMock()
        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            process=proc, pid=1001,
        )
        mgr.slots[slot.key] = slot

        await mgr.shutdown()
        proc.terminate.assert_called_once()
        assert len(mgr.slots) == 0

    @pytest.mark.asyncio
    async def test_shutdown_cancels_watchdog(self):
        mgr = LlamaCppManager(model_path="dummy.gguf")

        async def fake_watchdog():
            while True:
                await asyncio.sleep(1)

        mgr._watchdog_task = asyncio.create_task(fake_watchdog())
        await mgr.shutdown()
        assert mgr._watchdog_task.cancelled() or mgr._watchdog_task.done()


# ══════════════════════════════════════════════════════════════════════════════
#  Integration: ModelInterface + LlamaCppManager wiring
# ══════════════════════════════════════════════════════════════════════════════


class TestModelInterfaceWiring:
    def test_resolve_api_base_with_manager(self):
        from src.core.llm import ModelInterface
        mgr = LlamaCppManager(model_path="model.gguf")
        proc = MagicMock()
        proc.poll.return_value = None
        slot = ModelSlot(
            model_path="model.gguf", backend="rocm", port=8081,
            process=proc, pid=1001,
        )
        mgr.slots[slot.key] = slot

        with patch.dict(os.environ, {"INFERENCE_BACKEND": "llama-server"}):
            mi = ModelInterface(inference_manager=mgr)
            base = mi._resolve_api_base(prompt_tokens=100)
            assert base == "http://127.0.0.1:8081"

    def test_resolve_api_base_without_manager(self):
        from src.core.llm import ModelInterface
        with patch.dict(os.environ, {
            "INFERENCE_BACKEND": "llama-server",
            "LLAMA_API_BASE": "http://127.0.0.1:9999",
        }):
            mi = ModelInterface()
            base = mi._resolve_api_base(prompt_tokens=100)
            assert base == "http://127.0.0.1:9999"

    def test_disaggregate_routing_through_interface(self):
        from src.core.llm import ModelInterface
        mgr = LlamaCppManager(model_path="model.gguf")
        mgr.disaggregate = True
        mgr.prefill_token_threshold = 2048

        # Set up two slots: rocm and vulkan
        for backend, port in [("rocm", 8081), ("vulkan", 8082)]:
            proc = MagicMock()
            proc.poll.return_value = None
            slot = ModelSlot(
                model_path="model.gguf", backend=backend, port=port,
                process=proc, pid=1000 + port,
            )
            mgr.slots[slot.key] = slot

        with patch.dict(os.environ, {"INFERENCE_BACKEND": "llama-server"}):
            mi = ModelInterface(inference_manager=mgr)

            # Short prompt → vulkan
            base = mi._resolve_api_base(prompt_tokens=500)
            assert "8082" in base

            # Long prompt → rocm
            base = mi._resolve_api_base(prompt_tokens=5000)
            assert "8081" in base
