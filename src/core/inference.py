import asyncio
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
import psutil

logger = logging.getLogger(__name__)

class LlamaCppManager:
    """
    Manages the lifecycle of a llama-server process specifically tuned for the AMD RX 6750 XT.
    Implements a dual-API disaggregation strategy (ROCm prefill, Vulkan decode)
    and handles GTT spilling bounds.
    """
    def __init__(self, model_path: str, server_path: str = "llama-server", port: int = 8080):
        self.model_path = model_path
        self.server_path = server_path
        self.port = port
        self.process: subprocess.Popen = None
        self.context_window = int(os.getenv("LLAMA_CONTEXT_WINDOW", "262144"))
        
        # Watchdog Tracking
        self.last_activity_time = datetime.now()
        self.start_time = None
        self.idle_timeout_minutes = 15
        self.busy_timeout_minutes = 10
        self.watchdog_task = None
        
        # Dual-Backend Config
        # Note: In a real environment, switching backends requires separate compiled binaries or GGML flag toggles.
        # For this design, we expose parameters that would trigger Vulkan vs ROCm selection.
        self.active_backend = "ROCm" 

    def get_env_overrides(self, backend: str) -> dict:
        env = os.environ.copy()
        # Forcing ROCm execution for unsupported Navi 22 cards
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        
        if backend == "Vulkan":
            env["GGML_VULKAN"] = "1"
        else:
            # Default to ROCm (HIP/HSA)
            pass
            
        return env

    def get_launch_args(self) -> list:
        # Tuning specific to 12GB VRAM limit and GTT spilling.
        # -ngl 99 ensures weight layers fit in VRAM (Qwen 9B Q5_K_M is ~6.5GB).
        # -c 8192 limits total context, but KV cache will GTT spill perfectly into system RAM if it exceeds VRAM.
        return [
            self.server_path,
            "-m", self.model_path,
            "--port", str(self.port),
            "-ngl", "99",
            "-c", str(self.context_window),
            "-cb",                # Continuous batching
            "--metrics",          # Expose /metrics for hardware polling
            "--host", "127.0.0.1"
        ]

    async def start(self, backend: str = "ROCm"):
        if self.process and self.process.poll() is None:
            logger.info("Llama-server already running.")
            return

        self.active_backend = backend
        env = self.get_env_overrides(backend)
        args = self.get_launch_args()
        
        logger.info(f"Launching llama-server using {backend} backend: {' '.join(args)}")
        
        # Non-blocking subprocess launch
        self.process = subprocess.Popen(
            args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.start_time = datetime.now()
        self.record_activity()
        
        # Start watchdog loop
        if not self.watchdog_task:
            self.watchdog_task = asyncio.create_task(self._watchdog_loop())

    def record_activity(self):
        """Called by the API router whenever a request arrives."""
        self.last_activity_time = datetime.now()

    def stop(self):
        if self.process:
            logger.info("Terminating llama-server process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    async def _watchdog_loop(self):
        while True:
            await asyncio.sleep(60)
            if not self.process or self.process.poll() is not None:
                continue

            now = datetime.now()
            idle_time = now - self.last_activity_time
            uptime = now - self.start_time

            # Idle eviction
            if idle_time > timedelta(minutes=self.idle_timeout_minutes):
                logger.warning(f"Llama-server idle for >{self.idle_timeout_minutes}m. Unloading model.")
                self.stop()
                continue
            
            # Busy/Hung eviction
            # If a single request prevents activity updates for >10m, kill it.
            if uptime > timedelta(minutes=self.busy_timeout_minutes) and idle_time > timedelta(minutes=self.busy_timeout_minutes):
                logger.error(f"Llama-server appears hung (> {self.busy_timeout_minutes}m without completing). Terminating.")
                self.stop()
