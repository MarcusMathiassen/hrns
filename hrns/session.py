"""A Session is an append-only conversation persisted to disk.

WHY THIS SHAPE (the caching strategy):
DeepSeek's context cache, OpenRouter's response cache, and MiMo's upstream
cache are all keyed on the *prefix* of the request. A cache hit happens when
the leading bytes of `messages` are identical to a previous request the
platform has already seen. So this harness treats the message log as immutable
and append-only:

  * messages[0] is a STATIC system prompt — the stable anchor. Memory is
    snapshotted into it at creation time, never mutated afterward, so the
    prefix never shifts under us.
  * Every turn re-sends the full history; the unchanged prefix is served from
    the provider's cache (DeepSeek on-disk KV, OpenRouter proxy-layer, MiMo
    upstream).
  * Because the log is persisted byte-for-byte, RESUMING a session days later
    still replays an identical prefix and still hits the cache (within each
    provider's cache TTL; DeepSeek ~5 min on-disk, OpenRouter/MiMo varies by
    upstream — both reset the clock when the prefix is re-sent).
  * Volatile data (timestamps, "today is...") is kept OUT of the prefix so it
    never invalidates the cache.

We never edit or reorder past messages. That discipline is the feature.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hrns import storage
from hrns.config import DEFAULT_MODEL


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


@dataclass
class Session:
    id: str
    model: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    # everything the user saw, as rendered lines — replayed verbatim on resume
    # so a loaded session looks exactly like it did when they left it.
    # Entries: {"kind": "user"|"assistant"|"meta", "text": <rendered line(s)>}
    # LOCAL DISPLAY ONLY: never sent to the API — requests send `messages`
    # alone, so the transcript cannot disturb the cached prefix.
    transcript: list[dict[str, Any]] = field(default_factory=list)
    # cumulative, append-only counters
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        }
    )
    # current context size (tokens) — the latest request's prompt + completion.
    # This is the exact size from the API, not an estimate.
    context_tokens: int = 0

    # --- lifecycle -----------------------------------------------------
    @classmethod
    def new(cls, model: str, system_prompt: str) -> "Session":
        ts = _now()
        return cls(
            id=_new_id(),
            model=model,
            created_at=ts,
            updated_at=ts,
            messages=[{"role": "system", "content": system_prompt}],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        s = cls(
            id=data["id"],
            model=data.get("model", DEFAULT_MODEL),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
            messages=data.get("messages", []),
            transcript=data.get("transcript", []),
        )
        s.usage.update(data.get("usage", {}))
        s.context_tokens = int(data.get("context_tokens", 0) or 0)
        # Heal logs poisoned by an empty stream before the client guarded
        # against it: an assistant message with neither content nor tool_calls
        # is rejected by the API ("content or tool_calls must be set"), so
        # every request containing it 400s. Setting content to "" makes the
        # log valid again at the cost of one cache miss on the next turn.
        for m in s.messages:
            if (m.get("role") == "assistant"
                    and m.get("content") is None and not m.get("tool_calls")):
                m["content"] = ""
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "transcript": self.transcript,
            "usage": self.usage,
            "context_tokens": self.context_tokens,
        }

    # --- append-only mutation -----------------------------------------
    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.updated_at = _now()

    def log(self, kind: str, text: str) -> None:
        """Record a rendered line exactly as the user saw it."""
        self.transcript.append({"kind": kind, "text": text})

    def record_usage(self, usage: dict[str, Any]) -> None:
        if not usage:
            return
        self.usage["requests"] += 1
        for k in ("prompt_tokens", "completion_tokens",
                  "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            self.usage[k] += int(usage.get(k, 0) or 0)
        # latest request reflects the live context size
        self.context_tokens = (int(usage.get("prompt_tokens", 0) or 0)
                               + int(usage.get("completion_tokens", 0) or 0))

    def cost(self, pricing: dict[str, float]) -> float:
        u = self.usage
        return (u["prompt_cache_hit_tokens"] / 1e6 * pricing["cache_hit"]
                + u["prompt_cache_miss_tokens"] / 1e6 * pricing["cache_miss"]
                + u["completion_tokens"] / 1e6 * pricing["output"])

    # --- persistence ---------------------------------------------------
    def path(self, sessions_dir: Path) -> Path:
        return sessions_dir / f"{self.id}.json"

    def save(self, sessions_dir: Path) -> None:
        storage.write_json(self.path(sessions_dir), self.to_dict())

    @staticmethod
    def load(sessions_dir: Path, session_id: str) -> Optional["Session"]:
        data = storage.read_json(sessions_dir / f"{session_id}.json")
        return Session.from_dict(data) if data else None

    # --- helpers for display ------------------------------------------
    @property
    def cache_hit_rate(self) -> float:
        hit = self.usage["prompt_cache_hit_tokens"]
        total = hit + self.usage["prompt_cache_miss_tokens"]
        return (hit / total) if total else 0.0

    def title(self) -> str:
        """First line of the first user message, for listings."""
        for m in self.messages:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                first = m["content"].strip().splitlines()[0] if m["content"].strip() else ""
                return (first[:60] + "…") if len(first) > 60 else (first or "(empty)")
        return "(no messages yet)"

    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "user")


def list_sessions(sessions_dir: Path) -> list["Session"]:
    """All saved sessions, newest first."""
    if not sessions_dir.exists():
        return []
    out: list[Session] = []
    for p in sessions_dir.glob("*.json"):
        data = storage.read_json(p)
        if data:
            out.append(Session.from_dict(data))
    out.sort(key=lambda s: s.updated_at, reverse=True)
    return out
