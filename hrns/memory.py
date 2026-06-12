"""Persistent, cross-session memory.

Memory is a flat list of notes shared across every session. It is *snapshotted*
into a new session's system prompt at creation time (see cli.build_system_prompt)
and then frozen for that session's life — editing memory never mutates an existing
session's prefix, so it can't silently invalidate that session's cache. New
sessions pick up the updated memory.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hrns import storage


def _load(path: Path) -> list[dict[str, Any]]:
    return storage.read_json(path, default=[]) or []


def add(path: Path, text: str) -> dict[str, Any]:
    notes = _load(path)
    note = {
        "id": uuid.uuid4().hex[:8],
        "text": text.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    notes.append(note)
    storage.write_json(path, notes)
    return note


def list_notes(path: Path) -> list[dict[str, Any]]:
    return _load(path)


def remove(path: Path, note_id: str) -> bool:
    notes = _load(path)
    kept = [n for n in notes if n["id"] != note_id]
    if len(kept) == len(notes):
        return False
    storage.write_json(path, kept)
    return True


def clear(path: Path) -> None:
    storage.write_json(path, [])


def as_prompt_block(path: Path) -> str:
    """Render memory as a stable text block for the system prompt, or ''."""
    notes = _load(path)
    if not notes:
        return ""
    lines = "\n".join(f"- {n['text']}" for n in notes)
    return f"\n\nPersistent memory (durable facts the user wants you to remember):\n{lines}"
