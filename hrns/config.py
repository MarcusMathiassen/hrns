"""Configuration: where data lives, how we reach DeepSeek, and model pricing.

Resolution order for the API key:
    1. DEEPSEEK_API_KEY in the environment   (explicit per-process override)
    2. api_key saved in ~/.hrns/config.json  (remembered from a previous /connect)
    3. DEEPSEEK_API_KEY in a project-local .env (first-run bootstrap only)
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


# Context window in tokens. DeepSeek's current models share a 1M-token window.
# Source: https://api-docs.deepseek.com/quick_start/pricing
DEFAULT_CONTEXT_WINDOW = 1_000_000
CONTEXT_WINDOW: dict[str, int] = {
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
}


def context_window(model: str) -> int:
    return CONTEXT_WINDOW.get(model, DEFAULT_CONTEXT_WINDOW)


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
    approval_mode: str = "confirm"
    balance: float | None = None  # cached from /user/balance
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
        cfg.approval_mode = saved.get("approval_mode", cfg.approval_mode)

        dotenv = _load_dotenv(Path.cwd() / ".env")
        cfg.api_key = (
            os.environ.get("DEEPSEEK_API_KEY")   # explicit per-process override
            or saved.get("api_key")              # remembered from a previous /connect
            or dotenv.get("DEEPSEEK_API_KEY")    # project .env (first-run bootstrap)
        )
        return cfg

    def save(self, *, include_key: bool = False) -> None:
        """Persist settings to ~/.hrns/config.json so the next run remembers them.

        Merges with any existing file. The API key is written only when
        include_key=True (i.e. from `/connect`), so a key that merely came from
        the environment or a project .env is never silently copied in — and a
        stale saved key never shadows an updated .env.
        """
        data = storage.read_json(self.config_path, default={}) or {}
        data.update({
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "approval_mode": self.approval_mode,
        })
        if include_key:
            data["api_key"] = self.api_key
        storage.write_json(self.config_path, data)
        try:
            os.chmod(self.config_path, 0o600)
        except OSError:
            pass
