"""
WzrdMerlin v2 — Central Configuration

Loads merlin.config.yaml and exposes typed config objects.
Model switching is as simple as changing `active_model` in the config file,
or setting the MERLIN_MODEL environment variable.

Config file search path (first found wins):
  1. $MERLIN_CONFIG              (env var pointing to a yaml file)
  2. /workspace/merlin.config.yaml  (Docker volume — persists restarts)
  3. ./merlin.config.yaml           (local dev / project root)
  4. Built-in defaults + env var fallbacks (LLAMA_MODEL_PATH, etc.)
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Model Profile ─────────────────────────────────────────────────────────────

class ModelProfile(BaseModel):
    """Settings for a single local model."""
    model_path: str
    model_name: str
    context_window: int = 8192
    gpu_layers: int = 99
    kv_cache_type_k: str = "q8_0"
    kv_cache_type_v: str = "q8_0"
    think: bool = True
    think_budget: int = 1024
    temperature: float = 0.3


# ── Sub-configs ───────────────────────────────────────────────────────────────

class InferenceConfig(BaseModel):
    backend: str = "llama-server"       # "llama-server" | "ollama"
    server_path: str = "llama-server"
    port: int = 8081
    max_active_models: int = 1
    mmap: bool = True
    gtt_threshold: float = 85.0
    idle_timeout_minutes: int = 15
    busy_timeout_minutes: int = 10
    disaggregate: bool = False
    prefill_token_threshold: int = 2048


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"


class HardwareConfig(BaseModel):
    hsa_override_gfx_version: str = "10.3.0"
    radv_perftest: str = "nogttspill"


# ── Root Config ───────────────────────────────────────────────────────────────

class MerlinConfig(BaseModel):
    active_model: str = "default"
    models: Dict[str, ModelProfile] = Field(default_factory=dict)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    def get_active_model(self) -> ModelProfile:
        """
        Return the active model profile.
        MERLIN_MODEL env var overrides the config file's active_model field.
        Falls back to LLAMA_* env vars if no matching profile is defined.
        """
        name = os.getenv("MERLIN_MODEL", self.active_model)
        if name in self.models:
            return self.models[name]

        # Graceful fallback: synthesize a profile from legacy env vars
        if name != "default":
            logger.warning(
                f"Model profile '{name}' not found in config — "
                "falling back to LLAMA_MODEL_PATH / LLAMA_MODEL_NAME env vars"
            )
        return ModelProfile(
            model_path=os.getenv("LLAMA_MODEL_PATH", ""),
            model_name=os.getenv("LLAMA_MODEL_NAME", "qwen3.5:9b"),
            context_window=int(os.getenv("LLAMA_CONTEXT_WINDOW", "8192")),
            gpu_layers=int(os.getenv("LLAMA_GPU_LAYERS", "99")),
            kv_cache_type_k=os.getenv("LLAMA_KV_CACHE_TYPE_K", "q8_0"),
            kv_cache_type_v=os.getenv("LLAMA_KV_CACHE_TYPE_V", "q8_0"),
            think=os.getenv("LLAMA_THINK", "true").lower() == "true",
            think_budget=int(os.getenv("LLAMA_THINK_BUDGET", "1024")),
        )

    def list_models(self) -> List[str]:
        """Return all defined profile names."""
        return list(self.models.keys())

    def switch_active_model(self, name: str) -> None:
        """
        Switch the active model in-memory (does not write to disk).
        Raises ValueError if the profile is not defined in the config.
        """
        if name not in self.models:
            raise ValueError(
                f"Unknown model profile: '{name}'. "
                f"Available: {self.list_models()}"
            )
        self.active_model = name
        logger.info(f"Active model switched to: {name}")


# ── Singleton loader ──────────────────────────────────────────────────────────

_config: Optional[MerlinConfig] = None
_config_path: Optional[str] = None


def _find_config_file() -> Optional[Path]:
    candidates = [
        os.getenv("MERLIN_CONFIG", ""),
        "/workspace/merlin.config.yaml",
        "./merlin.config.yaml",
        str(Path(__file__).parent.parent.parent / "merlin.config.yaml"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            return Path(p)
    return None


def load_config() -> MerlinConfig:
    """Load config from disk (or return cached instance)."""
    global _config, _config_path
    if _config is not None:
        return _config

    config_file = _find_config_file()
    if config_file:
        try:
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
            _config = MerlinConfig(**data)
            _config_path = str(config_file)
            logger.info(
                f"Loaded config from {config_file} "
                f"(active_model={_config.active_model}, "
                f"profiles={_config.list_models()})"
            )
            return _config
        except Exception as e:
            logger.error(f"Failed to parse {config_file}: {e} — using defaults")

    logger.info("No merlin.config.yaml found — using built-in defaults with env var fallbacks")
    _config = MerlinConfig()
    return _config


def get_config() -> MerlinConfig:
    """Return the loaded config singleton (loads on first call)."""
    return load_config()


def reload_config() -> MerlinConfig:
    """Force a fresh reload from disk."""
    global _config, _config_path
    _config = None
    _config_path = None
    return load_config()


def get_config_path() -> Optional[str]:
    """Return the path to the loaded config file, or None if using defaults."""
    load_config()
    return _config_path
