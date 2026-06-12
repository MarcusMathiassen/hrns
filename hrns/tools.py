"""Coding tools exposed to the model via OpenAI-style function calling.

These make `hrns` an agentic *coding* harness rather than a plain chat loop.
Read-only tools (read_file, list_dir) run automatically; mutating/executing
tools (write_file, run_bash) go through a confirm() gate supplied by the CLI.

Tool loops are extremely cache-friendly: each step re-sends the whole growing
prefix, so every tool round-trip after the first is mostly served from
DeepSeek's cache.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

MAX_OUTPUT = 12_000  # truncate tool output to keep the context (and cost) bounded

# --- schemas sent to the model ----------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path, relative to cwd or absolute."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the entries of a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path. Defaults to '.'."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a shell command in the current working directory and return combined stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

READ_ONLY = {"read_file", "list_dir"}


def _truncate(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + f"\n… [truncated {len(s) - MAX_OUTPUT} chars]"


def _read_file(args: dict) -> str:
    p = Path(args["path"]).expanduser()
    try:
        return _truncate(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — surface any error back to the model
        return f"ERROR reading {p}: {e}"


def _list_dir(args: dict) -> str:
    p = Path(args.get("path", ".")).expanduser()
    try:
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        return "\n".join(f"{'d' if e.is_dir() else 'f'} {e.name}" for e in entries) or "(empty)"
    except Exception as e:  # noqa: BLE001
        return f"ERROR listing {p}: {e}"


def _write_file(args: dict) -> str:
    p = Path(args["path"]).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        return f"Wrote {len(args['content'])} chars to {p}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR writing {p}: {e}"


def _run_bash(args: dict) -> str:
    cmd = args["command"]
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return _truncate(f"$ {cmd}\n[exit {proc.returncode}]\n{out}")
    except subprocess.TimeoutExpired:
        return f"ERROR: `{cmd}` timed out after 120s"
    except Exception as e:  # noqa: BLE001
        return f"ERROR running `{cmd}`: {e}"


_DISPATCH: dict[str, Callable[[dict], str]] = {
    "read_file": _read_file,
    "list_dir": _list_dir,
    "write_file": _write_file,
    "run_bash": _run_bash,
}


def execute(name: str, raw_args: str, confirm: Callable[[str, dict], bool]) -> str:
    """Run a tool. `confirm(name, args) -> bool` gates mutating tools."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as e:
        return f"ERROR: could not parse arguments for {name}: {e}"

    if name not in READ_ONLY and not confirm(name, args):
        return f"User declined to run {name}."
    return fn(args)
