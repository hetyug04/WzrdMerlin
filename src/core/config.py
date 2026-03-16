"""
WzrdMerlin v3 — Central Configuration

Stripped down from v2: InferenceConfig removed entirely.
Only Ollama backend is supported.

Config file search path (first found wins):
  1. $MERLIN_CONFIG
  2. /workspace/merlin.config.yaml
  3. ./merlin.config.yaml
  4. Built-in defaults + env var fallbacks
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelProfile(BaseModel):
    model_name: str
    context_window: int = 32768
    temperature: float = 0.3
    think: bool = False
    think_budget: int = 0


class AgentPersonality(BaseModel):
    """A named behavioral profile that can be hot-swapped onto an agent at runtime."""
    name: str
    description: str
    system_prompt_override: str
    model_profile_key: str = "merlin-brain"   # key into MerlinConfig.models
    tool_allowlist: List[str] = Field(default_factory=list)  # empty = all tools allowed


class WorkersConfig(BaseModel):
    """Configuration for spawned sub-agent workers."""
    default_model: str = "qwen3.5-2b"
    max_concurrent: int = 4
    max_depth: int = 3
    max_iterations: int = 15
    timeout_seconds: float = 300.0


class OllamaConfig(BaseModel):
    base_url: str = "http://host.docker.internal:11434"
    keep_alive: str = "-1"          # never unload model
    read_timeout: float = 120.0     # per-chunk timeout


class HardwareConfig(BaseModel):
    hsa_override_gfx_version: str = "10.3.0"
    radv_perftest: str = "nogttspill"


class MerlinConfig(BaseModel):
    active_model: str = "merlin-brain"
    models: Dict[str, ModelProfile] = Field(default_factory=dict)
    personalities: Dict[str, AgentPersonality] = Field(default_factory=dict)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    def get_active_model(self) -> ModelProfile:
        name = os.getenv("MERLIN_MODEL", self.active_model)
        if name in self.models:
            return self.models[name]
        if name != "merlin-brain":
            logger.warning(
                f"Model profile '{name}' not found in config — "
                "falling back to env var MERLIN_MODEL_NAME"
            )
        return ModelProfile(
            model_name=os.getenv("MERLIN_MODEL_NAME", "merlin-model:latest"),
            context_window=int(os.getenv("MERLIN_CONTEXT_WINDOW", "32768")),
            temperature=float(os.getenv("MERLIN_TEMPERATURE", "0.3")),
            think=os.getenv("MERLIN_THINK", "false").lower() == "true",
        )

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def switch_active_model(self, name: str) -> None:
        if name not in self.models:
            raise ValueError(
                f"Unknown model profile: '{name}'. Available: {self.list_models()}"
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

    logger.info("No merlin.config.yaml found — using built-in defaults")
    _config = MerlinConfig()
    return _config


def get_config() -> MerlinConfig:
    return load_config()


def reload_config() -> MerlinConfig:
    global _config, _config_path
    _config = None
    _config_path = None
    return load_config()


def get_config_path() -> Optional[str]:
    load_config()
    return _config_path
