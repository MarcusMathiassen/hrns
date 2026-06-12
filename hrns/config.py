"""Configuration: where data lives, how we reach DeepSeek, and model pricing.

Resolution order for the API key:
    1. DEEPSEEK_API_KEY in the environment
    2. DEEPSEEK_API_KEY in a project-local .env (current working dir)
    3. api_key saved in ~/.hrns/config.json (written by `/connect`)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from hrns import storage

DEFAULT_BASE_URL = "https://api.deepseek.com"
# deepseek-chat is the broadly-available alias (non-thinking deepseek-v4-flash).
# Swap to deepseek-reasoner for thinking mode, or deepseek-v4-pro for the larger model.
DEFAULT_MODEL = "deepseek-chat"

# Per-1M-token pricing in USD. Source: https://api-docs.deepseek.com/quick_start/pricing
# The whole point of this harness is to push tokens into the cheap "cache hit" column.
PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
    # Legacy aliases (deprecate 2026-07-24) — priced as v4-flash.
    "deepseek-chat": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "deepseek-reasoner": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
}


def pricing_for(model: str) -> dict[str, float]:
    return PRICING.get(model, PRICING[DEFAULT_MODEL])


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser — KEY=VALUE lines, ignores comments and blanks."""
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip().strip("'\"")
    except FileNotFoundError:
        pass
    return out


@dataclass
class Config:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    max_tool_iters: int = 12
    home: Path = field(default_factory=lambda: Path(os.environ.get("HRNS_HOME", Path.home() / ".hrns")))

    # --- derived paths -------------------------------------------------
    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def memory_path(self) -> Path:
        return self.home / "memory" / "memory.json"

    @property
    def config_path(self) -> Path:
        return self.home / "config.json"

    # --- load / save ---------------------------------------------------
    @classmethod
    def load(cls) -> "Config":
        home = Path(os.environ.get("HRNS_HOME", Path.home() / ".hrns"))
        saved = storage.read_json(home / "config.json", default={}) or {}

        cfg = cls(home=home)
        cfg.base_url = saved.get("base_url", cfg.base_url)
        cfg.model = saved.get("model", cfg.model)
        cfg.temperature = saved.get("temperature", cfg.temperature)
        cfg.max_tool_iters = saved.get("max_tool_iters", cfg.max_tool_iters)

        dotenv = _load_dotenv(Path.cwd() / ".env")
        cfg.api_key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or dotenv.get("DEEPSEEK_API_KEY")
            or saved.get("api_key")
        )
        return cfg

    def save(self) -> None:
        """Persist non-derived settings (including the key) to ~/.hrns/config.json."""
        storage.write_json(
            self.config_path,
            {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "model": self.model,
                "temperature": self.temperature,
                "max_tool_iters": self.max_tool_iters,
            },
        )
        try:
            os.chmod(self.config_path, 0o600)
        except OSError:
            pass
