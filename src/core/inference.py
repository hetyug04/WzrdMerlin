"""
WzrdMerlin v2 — LlamaCppManager

Manages llama-server processes for the AMD RX 6750 XT (Navi 22, gfx1031).
Features:
  - Multi-model LRU eviction (max_active_models slots)
  - Dual-backend disaggregation (ROCm prefill, Vulkan decode)
  - KV cache quantization (Q8_0 / Q4_0 configurable)
  - GTT spill budget control (--mmap / --no-mmap, RAM threshold)
  - Health endpoint polling (/health)
  - Telemetry from /metrics (tokens/s, KV cache usage)
  - Watchdog timers (idle eviction, busy/hung kill)
"""
import asyncio
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import httpx
import psutil

from src.core.config import get_config, HardwareConfig

logger = logging.getLogger(__name__)


# ── Configuration from environment ────────────────────────────────────────────

def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ── Model Slot ────────────────────────────────────────────────────────────────

@dataclass
class ModelSlot:
    """Tracks a single running llama-server instance."""
    model_path: str
    backend: str            # "rocm" or "vulkan"
    port: int
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    last_used: float = field(default_factory=time.time)
    started_at: float = field(default_factory=time.time)
    ready: bool = False

    @property
    def key(self) -> str:
        return f"{self.model_path}:{self.backend}"

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


# ── Telemetry snapshot ────────────────────────────────────────────────────────

@dataclass
class InferenceTelemetry:
    """Parsed from llama-server /metrics endpoint."""
    tokens_per_sec: float = 0.0
    tokens_predicted: int = 0
    tokens_evaluated: int = 0
    kv_cache_used_cells: int = 0
    kv_cache_total_cells: int = 0
    requests_processing: int = 0
    requests_pending: int = 0
    timestamp: float = field(default_factory=time.time)


# ── LlamaCppManager ──────────────────────────────────────────────────────────

class LlamaCppManager:
    """
    Manages llama-server process lifecycle for AMD RX 6750 XT.

    Key capabilities:
      - start/stop/restart llama-server with correct env overrides
      - Multi-model LRU eviction when VRAM budget is exhausted
      - Dual-backend disaggregation: ROCm for prefill, Vulkan for decode
      - KV cache quantization to save VRAM
      - GTT spill monitoring and eviction
      - Health polling and readiness checks
      - Telemetry collection from /metrics
      - Watchdog: idle eviction + hung process kill
    """

    def __init__(
        self,
        model_path: str = None,
        server_path: str = None,
        port: int = None,
    ):
        cfg = get_config()
        active = cfg.get_active_model()
        inf = cfg.inference

        # Priority: constructor arg > env var > config value
        # Paths
        self.default_model_path = model_path or _env("LLAMA_MODEL_PATH", "") or active.model_path
        self.server_path = server_path or _env("LLAMA_SERVER_PATH", "") or inf.server_path

        # Port allocation
        self._base_port = port or _env_int("LLAMA_PORT", inf.port)
        self._next_port = self._base_port

        # GPU / context — env var overrides config profile
        self.context_window = _env_int("LLAMA_CONTEXT_WINDOW", active.context_window)
        self.gpu_layers = _env_int("LLAMA_GPU_LAYERS", active.gpu_layers)
        self.kv_cache_type_k = _env("LLAMA_KV_CACHE_TYPE_K", active.kv_cache_type_k)
        self.kv_cache_type_v = _env("LLAMA_KV_CACHE_TYPE_V", active.kv_cache_type_v)

        # Memory / capacity — env var overrides inference config
        self.use_mmap = _env_bool("LLAMA_MMAP", inf.mmap)
        self.gtt_threshold = _env_float("LLAMA_GTT_THRESHOLD", inf.gtt_threshold)
        self.max_active_models = _env_int("LLAMA_MAX_MODELS", inf.max_active_models)

        # Dual-backend disaggregation
        self.disaggregate = _env_bool("LLAMA_DISAGGREGATE", inf.disaggregate)
        self.prefill_token_threshold = _env_int("LLAMA_PREFILL_THRESHOLD", inf.prefill_token_threshold)

        # Watchdog
        self.idle_timeout_minutes = _env_int("LLAMA_IDLE_TIMEOUT", inf.idle_timeout_minutes)
        self.busy_timeout_minutes = _env_int("LLAMA_BUSY_TIMEOUT", inf.busy_timeout_minutes)

        # Hardware config for env overrides (AMD GPU)
        self._hw: HardwareConfig = cfg.hardware

        # State
        self.slots: Dict[str, ModelSlot] = {}
        self._watchdog_task: Optional[asyncio.Task] = None
        self._last_telemetry: Optional[InferenceTelemetry] = None

    # ── Environment overrides ─────────────────────────────────────────────

    def get_env_overrides(self, backend: str = "rocm") -> dict:
        """Build env dict with AMD GPU overrides for the given backend."""
        env = os.environ.copy()
        # AMD GPU compatibility overrides (from hardware config)
        env["HSA_OVERRIDE_GFX_VERSION"] = self._hw.hsa_override_gfx_version
        env["RADV_PERFTEST"] = self._hw.radv_perftest

        if backend.lower() == "vulkan":
            env["GGML_VULKAN"] = "1"
        else:
            # ROCm (HIP/HSA) — ensure Vulkan flag is NOT set
            env.pop("GGML_VULKAN", None)

        return env

    # ── Launch args ───────────────────────────────────────────────────────

    def get_launch_args(
        self,
        model_path: str,
        port: int,
        context_window: int = None,
    ) -> List[str]:
        """Build llama-server CLI arguments."""
        ctx = context_window or self.context_window
        args = [
            self.server_path,
            "-m", model_path,
            "--port", str(port),
            "-ngl", str(self.gpu_layers),
            "-c", str(ctx),
            "-cb",                          # Continuous batching
            "--metrics",                     # Expose /metrics for telemetry
            "--host", "127.0.0.1",
            "--cache-type-k", self.kv_cache_type_k,
            "--cache-type-v", self.kv_cache_type_v,
        ]

        # GTT spill budget control
        if not self.use_mmap:
            args.append("--no-mmap")

        return args

    # ── Port allocation ───────────────────────────────────────────────────

    def _allocate_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    # ── Slot key helpers ──────────────────────────────────────────────────

    @staticmethod
    def _slot_key(model_path: str, backend: str) -> str:
        return f"{model_path}:{backend.lower()}"

    # ── Start a model ─────────────────────────────────────────────────────

    async def start(
        self,
        model_path: str = None,
        backend: str = "rocm",
    ) -> ModelSlot:
        """Start a llama-server instance for the given model and backend."""
        model_path = model_path or self.default_model_path
        if not model_path:
            raise ValueError("No model path provided and LLAMA_MODEL_PATH not set")

        key = self._slot_key(model_path, backend)

        # Already running?
        if key in self.slots and self.slots[key].is_alive:
            slot = self.slots[key]
            slot.last_used = time.time()
            logger.info(f"Model already loaded: {key} on port {slot.port}")
            return slot

        # Evict LRU if at capacity
        await self._evict_if_needed()

        port = self._allocate_port()
        env = self.get_env_overrides(backend)
        args = self.get_launch_args(model_path, port)

        logger.info(
            f"Launching llama-server: backend={backend} port={port} "
            f"model={os.path.basename(model_path)} "
            f"kv_cache={self.kv_cache_type_k}/{self.kv_cache_type_v} "
            f"mmap={self.use_mmap} ctx={self.context_window}"
        )

        process = subprocess.Popen(
            args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        slot = ModelSlot(
            model_path=model_path,
            backend=backend.lower(),
            port=port,
            process=process,
            pid=process.pid,
            last_used=time.time(),
            started_at=time.time(),
            ready=False,
        )
        self.slots[key] = slot

        # Wait for server readiness
        await self._wait_for_health(slot)

        # Start watchdog if not running
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

        return slot

    async def _wait_for_health(self, slot: ModelSlot, timeout: float = 120.0):
        """Poll /health until the server is ready or timeout."""
        deadline = time.time() + timeout
        url = f"{slot.api_base}/health"

        while time.time() < deadline:
            if not slot.is_alive:
                stderr = ""
                try:
                    stderr = slot.process.stderr.read() if slot.process else ""
                except Exception:
                    pass
                raise RuntimeError(
                    f"llama-server exited during startup (code={slot.process.returncode}): {stderr[:500]}"
                )
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                        status = data.get("status", "ok")
                        if status in ("ok", "no slot available"):
                            slot.ready = True
                            logger.info(
                                f"llama-server ready on port {slot.port} "
                                f"(startup {time.time() - slot.started_at:.1f}s)"
                            )
                            return
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
                pass
            await asyncio.sleep(1.0)

        raise TimeoutError(
            f"llama-server on port {slot.port} not ready after {timeout}s"
        )

    # ── Stop a slot ───────────────────────────────────────────────────────

    def stop_slot(self, key: str) -> None:
        """Terminate a specific model slot."""
        slot = self.slots.get(key)
        if not slot:
            return
        if slot.process:
            logger.info(f"Terminating llama-server: {key} (pid={slot.pid})")
            slot.process.terminate()
            try:
                slot.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                slot.process.kill()
                slot.process.wait(timeout=3)
            slot.process = None
            slot.ready = False
        self.slots.pop(key, None)

    def stop_all(self) -> None:
        """Terminate all running llama-server instances."""
        for key in list(self.slots.keys()):
            self.stop_slot(key)

    # ── LRU eviction ──────────────────────────────────────────────────────

    async def _evict_if_needed(self) -> None:
        """Evict LRU model(s) until we're under max_active_models."""
        alive = {k: s for k, s in self.slots.items() if s.is_alive}
        while len(alive) >= self.max_active_models:
            lru_key = min(alive, key=lambda k: alive[k].last_used)
            logger.warning(
                f"LRU eviction: stopping {lru_key} "
                f"(last_used {time.time() - alive[lru_key].last_used:.0f}s ago) "
                f"to make room (active={len(alive)}, max={self.max_active_models})"
            )
            self.stop_slot(lru_key)
            alive.pop(lru_key)

    # ── Ensure model is loaded ────────────────────────────────────────────

    async def ensure_model(
        self,
        model_path: str = None,
        backend: str = None,
    ) -> ModelSlot:
        """
        Ensure a model is loaded and ready. Returns the slot.
        If dual-backend disaggregation is off, uses default backend.
        """
        model_path = model_path or self.default_model_path
        backend = (backend or "rocm").lower()
        key = self._slot_key(model_path, backend)

        slot = self.slots.get(key)
        if slot and slot.is_alive and slot.ready:
            slot.last_used = time.time()
            return slot

        # Need to start
        return await self.start(model_path, backend)

    def select_backend(self, prompt_tokens: int) -> str:
        """
        Dual-backend routing: pick ROCm or Vulkan based on prompt length.
        ROCm is better for prefill-heavy (long prompt) workloads.
        Vulkan is better for decode-heavy (short prompt, long generation).
        """
        if not self.disaggregate:
            return "rocm"  # Default single-backend

        if prompt_tokens > self.prefill_token_threshold:
            return "rocm"   # Long prompt → ROCm for fast prefill
        else:
            return "vulkan"  # Short prompt → Vulkan for fast decode

    async def switch_model(self, profile_name: str) -> "ModelSlot":
        """
        Hot-swap to a different model profile.

        1. Validates the profile exists in the config.
        2. Updates all model-specific settings on this manager.
        3. Stops all currently running slots.
        4. Starts the new model.
        """
        cfg = get_config()
        cfg.switch_active_model(profile_name)   # raises ValueError if unknown
        profile = cfg.get_active_model()

        logger.info(
            f"Switching model to '{profile_name}': "
            f"path={profile.model_path} "
            f"ctx={profile.context_window} layers={profile.gpu_layers}"
        )

        # Apply new model settings
        self.default_model_path = profile.model_path
        self.context_window = profile.context_window
        self.gpu_layers = profile.gpu_layers
        self.kv_cache_type_k = profile.kv_cache_type_k
        self.kv_cache_type_v = profile.kv_cache_type_v

        # Evict all current slots before loading the new model
        self.stop_all()

        return await self.start(profile.model_path)

    def get_api_base(self, model_path: str = None, backend: str = None) -> Optional[str]:
        """Return the base URL for a loaded model, or None if not loaded."""
        model_path = model_path or self.default_model_path
        backend = (backend or "rocm").lower()
        key = self._slot_key(model_path, backend)
        slot = self.slots.get(key)
        if slot and slot.is_alive:
            return slot.api_base
        return None

    def record_activity(self, model_path: str = None, backend: str = None) -> None:
        """Record activity timestamp for watchdog idle tracking."""
        model_path = model_path or self.default_model_path
        backend = (backend or "rocm").lower()
        key = self._slot_key(model_path, backend)
        slot = self.slots.get(key)
        if slot:
            slot.last_used = time.time()

    # ── Telemetry ─────────────────────────────────────────────────────────

    async def poll_telemetry(self, slot: ModelSlot = None) -> Optional[InferenceTelemetry]:
        """
        Parse Prometheus metrics from llama-server /metrics endpoint.
        Returns InferenceTelemetry or None if unavailable.
        """
        if slot is None:
            # Use the first alive slot
            alive = [s for s in self.slots.values() if s.is_alive]
            if not alive:
                return None
            slot = alive[0]

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{slot.api_base}/metrics")
                if resp.status_code != 200:
                    return None
                return self._parse_metrics(resp.text)
        except (httpx.ConnectError, httpx.ReadTimeout):
            return None

    @staticmethod
    def _parse_metrics(text: str) -> InferenceTelemetry:
        """Parse Prometheus-format metrics text into InferenceTelemetry."""
        tel = InferenceTelemetry()

        def _get(name: str) -> Optional[float]:
            # Match both counter and gauge lines: metric_name value
            m = re.search(rf"^{re.escape(name)}\s+([\d.eE+\-]+)", text, re.MULTILINE)
            return float(m.group(1)) if m else None

        tel.tokens_predicted = int(_get("llama_tokens_predicted_total") or _get("llama_tokens_predicted") or 0)
        tel.tokens_evaluated = int(_get("llama_prompt_tokens_total") or _get("llama_tokens_evaluated") or 0)
        tel.kv_cache_used_cells = int(_get("llama_kv_cache_used_cells") or 0)
        tel.kv_cache_total_cells = int(_get("llama_kv_cache_tokens") or 0)
        tel.requests_processing = int(_get("llama_requests_processing") or 0)
        tel.requests_pending = int(_get("llama_requests_pending") or 0)

        # tokens/s: derived from llama_tokens_second or computed
        tps = _get("llama_tokens_second_total") or _get("llama_tokens_second")
        if tps is not None:
            tel.tokens_per_sec = tps
        # Some builds use a different name
        if tel.tokens_per_sec == 0:
            tps2 = _get("llama_decode_tokens_per_second")
            if tps2:
                tel.tokens_per_sec = tps2

        tel.timestamp = time.time()
        return tel

    async def get_health(self, slot: ModelSlot = None) -> Dict[str, Any]:
        """Return health status for a slot (or first alive slot)."""
        if slot is None:
            alive = [s for s in self.slots.values() if s.is_alive]
            if not alive:
                return {"status": "no_model_loaded", "model_loaded": False}
            slot = alive[0]

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{slot.api_base}/health")
                if resp.status_code == 200:
                    return {
                        "status": "ok",
                        "model_loaded": True,
                        "active_model": os.path.basename(slot.model_path),
                        "backend": slot.backend,
                        "port": slot.port,
                        "pid": slot.pid,
                        "uptime_seconds": int(time.time() - slot.started_at),
                    }
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass

        return {
            "status": "unhealthy",
            "model_loaded": False,
            "active_model": os.path.basename(slot.model_path),
            "backend": slot.backend,
        }

    def get_all_slots_info(self) -> List[Dict[str, Any]]:
        """Summary of all slots for debug endpoints."""
        return [
            {
                "key": slot.key,
                "model": os.path.basename(slot.model_path),
                "backend": slot.backend,
                "port": slot.port,
                "pid": slot.pid,
                "alive": slot.is_alive,
                "ready": slot.ready,
                "idle_seconds": int(time.time() - slot.last_used),
                "uptime_seconds": int(time.time() - slot.started_at),
            }
            for slot in self.slots.values()
        ]

    # ── Watchdog loop ─────────────────────────────────────────────────────

    async def _watchdog_loop(self) -> None:
        """
        Periodic watchdog checks:
          1. GTT spill detection — evict if system RAM > threshold
          2. Idle eviction — unload models that haven't been used
          3. Hung process detection — kill processes stuck too long
          4. Dead process cleanup — remove slots whose process exited
        """
        while True:
            await asyncio.sleep(30)

            for key in list(self.slots.keys()):
                slot = self.slots.get(key)
                if not slot:
                    continue

                # Dead process cleanup
                if not slot.is_alive:
                    logger.warning(f"WATCHDOG: Dead process detected for {key}, cleaning up")
                    self.slots.pop(key, None)
                    continue

                now = time.time()
                idle_seconds = now - slot.last_used
                uptime_seconds = now - slot.started_at

                # 1. GTT spill detection — system RAM pressure
                try:
                    vm = psutil.virtual_memory()
                    if vm.percent > self.gtt_threshold:
                        logger.error(
                            f"WATCHDOG: System RAM at {vm.percent:.1f}% "
                            f"(threshold {self.gtt_threshold}%). "
                            f"GTT spilling detected — evicting {key}"
                        )
                        self.stop_slot(key)
                        continue
                except Exception as e:
                    logger.warning(f"WATCHDOG: Failed to check RAM: {e}")

                # 2. Idle eviction
                if idle_seconds > self.idle_timeout_minutes * 60:
                    logger.warning(
                        f"WATCHDOG: {key} idle for {idle_seconds/60:.1f}m "
                        f"(timeout={self.idle_timeout_minutes}m). Evicting."
                    )
                    self.stop_slot(key)
                    continue

                # 3. Hung process detection
                if (
                    uptime_seconds > self.busy_timeout_minutes * 60
                    and idle_seconds > self.busy_timeout_minutes * 60
                ):
                    logger.error(
                        f"WATCHDOG: {key} appears hung — "
                        f"uptime={uptime_seconds/60:.1f}m, "
                        f"idle={idle_seconds/60:.1f}m. Killing."
                    )
                    self.stop_slot(key)
                    continue

            # Collect telemetry
            try:
                self._last_telemetry = await self.poll_telemetry()
            except Exception:
                pass

    @property
    def last_telemetry(self) -> Optional[InferenceTelemetry]:
        return self._last_telemetry

    # ── Shutdown ──────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Clean shutdown: stop all servers and cancel watchdog."""
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        self.stop_all()
        logger.info("LlamaCppManager shut down")
