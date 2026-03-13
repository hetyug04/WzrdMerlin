import pytest
import os
import asyncio
from unittest.mock import MagicMock, patch
from src.core.inference import LlamaCppManager

@pytest.mark.asyncio
async def test_inference_env_overrides():
    manager = LlamaCppManager(model_path="dummy.gguf")

    # Test ROCm (default)
    env = manager.get_env_overrides("rocm")
    assert env["HSA_OVERRIDE_GFX_VERSION"] == "10.3.0"
    assert env["RADV_PERFTEST"] == "nogttspill"
    assert "GGML_VULKAN" not in env

    # Test Vulkan
    env_vulkan = manager.get_env_overrides("vulkan")
    assert env_vulkan["GGML_VULKAN"] == "1"
    assert env_vulkan["RADV_PERFTEST"] == "nogttspill"

def test_inference_launch_args():
    manager = LlamaCppManager(model_path="qwen.gguf")
    args = manager.get_launch_args("qwen.gguf", 8081)

    # Verify KV Cache Quantization
    assert "--cache-type-k" in args
    assert "q8_0" in args
    assert "--cache-type-v" in args

    # Verify Context Window
    assert "-c" in args
    assert str(manager.context_window) in args

    # Verify GPU offloading
    assert "-ngl" in args
    assert "99" in args

@pytest.mark.asyncio
async def test_memory_watchdog_trigger():
    manager = LlamaCppManager(model_path="dummy.gguf")
    proc = MagicMock()
    proc.poll.return_value = None  # Process is "running"
    proc.terminate = MagicMock()
    proc.wait = MagicMock()

    from src.core.inference import ModelSlot
    slot = ModelSlot(
        model_path="dummy.gguf", backend="rocm", port=8081,
        process=proc, pid=1001,
    )
    manager.slots[slot.key] = slot

    # Mock psutil to simulate 95% RAM usage (over the 85% default threshold)
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.percent = 95.0

        # Simulate one watchdog check
        for key in list(manager.slots.keys()):
            s = manager.slots.get(key)
            if s and s.is_alive:
                vm = mock_vm()
                if vm.percent > manager.gtt_threshold:
                    manager.stop_slot(key)

        # Slot should have been evicted
        assert slot.key not in manager.slots
        proc.terminate.assert_called_once()
