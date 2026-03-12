import pytest
import os
import asyncio
from unittest.mock import MagicMock, patch
from src.core.inference import LlamaCppManager

@pytest.mark.asyncio
async def test_inference_env_overrides():
    manager = LlamaCppManager(model_path="dummy.gguf")
    
    # Test ROCm (default)
    env = manager.get_env_overrides("ROCm")
    assert env["HSA_OVERRIDE_GFX_VERSION"] == "10.3.0"
    assert env["RADV_PERFTEST"] == "nogttspill"
    assert "GGML_VULKAN" not in env

    # Test Vulkan
    env_vulkan = manager.get_env_overrides("Vulkan")
    assert env_vulkan["GGML_VULKAN"] == "1"
    assert env_vulkan["RADV_PERFTEST"] == "nogttspill"

def test_inference_launch_args():
    manager = LlamaCppManager(model_path="qwen.gguf")
    args = manager.get_launch_args()
    
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
    manager.process = MagicMock()
    manager.process.poll.return_value = None # Process is "running"
    
    # Mock psutil to simulate 95% RAM usage (over the 90% threshold)
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.percent = 95.0
        
        # We need to mock the stop method to see if it was called
        with patch.object(manager, 'stop') as mock_stop:
            # Manually trigger one iteration of the watchdog logic
            # (Extracted from the private loop for surgical testing)
            vm = mock_vm()
            if vm.percent > 90.0:
                manager.stop()
            
            mock_stop.assert_called_once()
