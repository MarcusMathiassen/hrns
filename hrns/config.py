"""Configuration: where data lives, how we reach the API, and model pricing.

Resolution order for the API key:
    1. PROVIDER_API_KEY in the environment   (explicit per-process override)
    2. api_key saved in ~/.hrns/config.json  (remembered from a previous /connect)
    3. PROVIDER_API_KEY in a project-local .env (first-run bootstrap only)

The provider is auto-detected from the base_url — "deepseek" for
api.deepseek.com, "openrouter" for openrouter.ai.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from hrns import storage

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"

# Per-1M-token pricing in USD.
PRICING: dict[str, dict[str, float]] = {
    # DeepSeek  (cache_hit / cache_miss / output)
    "deepseek-v4-flash":  {"cache_hit": 0.0028,   "cache_miss": 0.14,   "output": 0.28},
    "deepseek-v4-pro":    {"cache_hit": 0.003625, "cache_miss": 0.435,  "output": 0.87},
    "deepseek-chat":      {"cache_hit": 0.0028,   "cache_miss": 0.14,   "output": 0.28},
    "deepseek-reasoner":  {"cache_hit": 0.0028,   "cache_miss": 0.14,   "output": 0.28},
    # OpenRouter — prompt / completion (no per-model cache-hit column; cache_hit ≈ prompt)
    "deepseek/deepseek-chat":          {"cache_hit": 0.14, "cache_miss": 0.14, "output": 0.28},
    "deepseek/deepseek-r1":            {"cache_hit": 0.35, "cache_miss": 0.35, "output": 2.19},
    "deepseek/deepseek-r1-distill":    {"cache_hit": 0.07, "cache_miss": 0.07, "output": 0.28},
    "anthropic/claude-3.5-sonnet":     {"cache_hit": 0.69, "cache_miss": 0.69, "output": 2.75},
    "anthropic/claude-3-opus":         {"cache_hit": 2.75, "cache_miss": 2.75, "output": 8.25},
    "openai/gpt-4o":                   {"cache_hit": 2.07, "cache_miss": 2.07, "output": 6.90},
    "openai/gpt-4o-mini":              {"cache_hit": 0.10, "cache_miss": 0.10, "output": 0.34},
    "google/gemini-2.0-flash":         {"cache_hit": 0.08, "cache_miss": 0.08, "output": 0.29},
    "google/gemini-2.5-pro":           {"cache_hit": 1.38, "cache_miss": 1.38, "output": 5.50},
    "meta-llama/llama-4-maverick":     {"cache_hit": 0.17, "cache_miss": 0.17, "output": 0.76},
}


def pricing_for(model: str) -> dict[str, float]:
    return PRICING.get(model, PRICING[DEFAULT_MODEL])


# Context window in tokens.
DEFAULT_CONTEXT_WINDOW = 1_000_000
CONTEXT_WINDOW: dict[str, int] = {
    "deepseek-chat":               1_000_000,
    "deepseek-reasoner":           1_000_000,
    "deepseek-v4-flash":           1_000_000,
    "deepseek-v4-pro":             1_000_000,
    "deepseek/deepseek-chat":        128_000,
    "deepseek/deepseek-r1":          128_000,
    "deepseek/deepseek-r1-distill":  128_000,
    "anthropic/claude-3.5-sonnet":   200_000,
    "anthropic/claude-3-opus":       200_000,
    "openai/gpt-4o":                 128_000,
    "openai/gpt-4o-mini":            128_000,
    "google/gemini-2.0-flash":     1_000_000,
    "google/gemini-2.5-pro":       1_000_000,
    "meta-llama/llama-4-maverick": 1_000_000,
}


def context_window(model: str) -> int:
    return CONTEXT_WINDOW.get(model, DEFAULT_CONTEXT_WINDOW)


# --- provider detection --------------------------------------------------
Provider = str  # "deepseek" | "openrouter"


def detect_provider(base_url: str) -> Provider:
    """Return the provider key for a base_url."""
    if "openrouter.ai" in base_url:
        return "openrouter"
    return "deepseek"


# Provider→env var name for the API key.  The saved key is shared across
# providers (one key in ~/.hrns/config.json), but env-var lookups are
# provider-specific so you can keep both keys set without conflict.
_PROVIDER_API_KEY_ENV: dict[Provider, str] = {
    "deepseek":   "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


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
    balance: float | None = None  # cached from balance endpoint
    home: Path = field(default_factory=lambda: Path(os.environ.get("HRNS_HOME", Path.home() / ".hrns")))

    # --- derived paths -------------------------------------------------
    @property
    def provider(self) -> Provider:
        return detect_provider(self.base_url)

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

        provider = cfg.provider
        key_env = _PROVIDER_API_KEY_ENV.get(provider, "DEEPSEEK_API_KEY")
        dotenv = _load_dotenv(Path.cwd() / ".env")
        cfg.api_key = (
            os.environ.get(key_env)              # explicit per-process override
            or saved.get("api_key")              # remembered from a previous /connect
            or dotenv.get(key_env)               # project .env (first-run bootstrap)
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
