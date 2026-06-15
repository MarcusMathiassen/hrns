"""Coding tools exposed to the model via OpenAI-style function calling.

The design follows what modern coding agents converge on:

  * read with line numbers + paging (offset/limit), so the model can cite
    exact locations and chunk large files;
  * edit by EXACT string replacement with a uniqueness guard (not blind
    whole-file overwrites), returning a unified diff of what changed;
  * create new files separately from editing, so nothing is clobbered by
    accident;
  * first-class search: grep (regex, ripgrep-backed) and glob (by pattern);
  * shell execution with exit status, timeout, and bounded output.

Read-only tools (read_file, list_dir, glob, grep) run automatically; mutating
tools (edit_file, create_file, run_bash) pass through a confirm() gate that the
CLI uses to show a diff/preview before applying.

Tool loops are very cache-friendly: each step re-sends the whole growing
prefix, so every round-trip after the first is mostly served from DeepSeek's
cache.
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

MAX_OUTPUT = 12_000      # cap any tool result (chars) to keep context/cost bounded
MAX_READ_LINES = 2_000   # default read window


def _kill_process_group(proc: subprocess.Popen, timeout: int) -> None:
    """Kill a process and all its children, then wait for cleanup."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # gave it our best shot

MAX_LINE_LEN = 2_000     # truncate pathological single lines
GREP_CAP = 200           # max match lines returned
GLOB_CAP = 200           # max paths returned

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".idea", ".codegraph",
    ".egg-info",
}


# --- workspace containment --------------------------------------------
# hrns stays inside the folder it was opened in. Any tool whose target path
# escapes this root (read-only tools included) must ask the user first.
WORKSPACE_ROOT = Path(os.path.realpath(os.getcwd()))

# which argument holds the path each tool touches. run_bash is unconstrained
# (a shell can reach anywhere), so it is always confirmed instead.
_PATH_ARG = {
    "read_file": "path", "edit_file": "path", "create_file": "path",
    "list_dir": "path", "glob": "path", "grep": "path",
}
_DEFAULTS_TO_DOT = {"list_dir", "glob", "grep"}


def set_workspace_root(path: Any) -> None:
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = Path(os.path.realpath(str(path)))


def workspace_root() -> Path:
    return WORKSPACE_ROOT


def _is_outside(path_str: str) -> bool:
    # Resolve relative paths against the workspace root (not the process cwd),
    # then realpath to collapse '..' and symlinks so neither can sneak past.
    p = Path(str(path_str)).expanduser()
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p
    rp = Path(os.path.realpath(str(p)))
    try:
        return not rp.is_relative_to(WORKSPACE_ROOT)
    except ValueError:
        # Python 3.9–3.12 raise ValueError for non-relative paths
        return True


def _escape_path(name: str, args: dict) -> str | None:
    """The tool's target path if it escapes the workspace, else None."""
    key = _PATH_ARG.get(name)
    if key is None:
        return None
    raw = args.get(key, "." if name in _DEFAULTS_TO_DOT else None)
    if raw in (None, ""):
        return None
    return str(raw) if _is_outside(raw) else None


def resolve_target(path_str: str) -> Path:
    """The absolute, symlink-resolved path a tool's `path` argument refers to,
    anchored to the workspace root for relative paths — the same resolution the
    containment check uses. Lets the CLI show and remember real targets."""
    p = Path(str(path_str)).expanduser()
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p
    return Path(os.path.realpath(str(p)))


# --- schemas advertised to the model ----------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file, returned with line numbers. Use `offset` "
                "and `limit` to page through large files. ALWAYS read a file before "
                "editing it so your edit_file old_string matches exactly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to cwd or absolute."},
                    "offset": {"type": "integer", "description": "1-based line to start at. Default 1."},
                    "limit": {"type": "integer", "description": f"Max lines to return. Default {MAX_READ_LINES}."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the entries of a directory (directories first, with file sizes).",
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
            "name": "glob",
            "description": (
                "Find files by glob pattern (e.g. '**/*.py', 'src/**/test_*.py'). "
                "Returns matching paths, most-recently-modified first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, evaluated under `path`."},
                    "path": {"type": "string", "description": "Root directory. Defaults to '.'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents with a regular expression (ripgrep-backed). "
                "Returns matching lines as 'file:line:text'. Prefer this over "
                "reading whole files to locate code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "File or directory to search. Defaults to '.'."},
                    "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py'."},
                    "case_insensitive": {"type": "boolean", "description": "Case-insensitive match. Default false."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact substring in an existing file. `old_string` must "
                "match EXACTLY (including whitespace/indentation) and be UNIQUE — "
                "include a few surrounding lines for context — unless `replace_all` "
                "is true. Read the file first. Returns a unified diff of the change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string", "description": "Exact text to find. Must be unique unless replace_all."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": (
                "Create a NEW text file. Fails if the path already exists unless "
                "`overwrite` is true. To modify an existing file, use edit_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean", "description": "Allow overwriting an existing file. Default false."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a shell command in the current working directory. Returns "
                "combined stdout/stderr and the exit status. Use for builds, tests, "
                "git, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds before the command is killed. Default 120, max 600."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save a fact, preference, convention, or pattern the user has "
                "expressed to persistent cross-session memory. Applies to FUTURE "
                "sessions only — it does NOT modify the current conversation. The "
                "user can view, remove, or clear memories with /memory commands. "
                "Use sparingly: only save things the user explicitly stated they "
                "want remembered, or clear recurring preferences/patterns they've "
                "demonstrated across multiple turns. Do NOT save one-off requests, "
                "temporary context, or speculative inferences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The fact to remember, in 1-2 concise sentences. "
                            "Write in third person about the user. "
                            'E.g. "The user prefers TypeScript over JavaScript." '
                            'or "The user uses pytest for testing in all Python projects."'
                        ),
                    },
                },
                "required": ["text"],
            },
        },
    },
]

READ_ONLY = {"read_file", "list_dir", "glob", "grep"}


# --- helpers -----------------------------------------------------------
def _truncate(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + f"\n… [truncated {len(s) - MAX_OUTPUT} chars]"


def _resolve(path: str) -> Path:
    # Relative paths are anchored to the workspace root — the same anchor
    # _is_outside uses — so the containment check and the actual file
    # operation always refer to the same target.
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else WORKSPACE_ROOT / p


def _human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if f < 1024 or unit == "T":
            return f"{int(f)}{unit}" if unit == "B" else f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}P"


def _atomic_write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".hrns.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def _unified(before: str, after: str, path: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=2,
    )
    return "\n".join(diff)


def _walk(root: Path, glob_filter: str | None) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for name in files:
            if glob_filter and not fnmatch.fnmatch(name, glob_filter):
                continue
            out.append(Path(dirpath) / name)
    return out


# --- tool implementations ---------------------------------------------
def _read_file(args: dict) -> str:
    p = _resolve(args["path"])
    offset = max(1, int(args.get("offset", 1) or 1))
    limit = int(args.get("limit", MAX_READ_LINES) or MAX_READ_LINES)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: file not found: {p}"
    except IsADirectoryError:
        return f"ERROR: {p} is a directory. Use list_dir."
    except UnicodeDecodeError:
        return f"ERROR: {p} is not UTF-8 text (binary file?)."
    except Exception as e:  # noqa: BLE001
        return f"ERROR reading {p}: {e}"

    lines = raw.splitlines()
    total = len(lines)
    if total == 0:
        return f"{p} (0 lines, empty)"
    if offset > total:
        return f"ERROR: offset {offset} is past end of file ({total} lines)."

    window = lines[offset - 1: offset - 1 + limit]
    rendered = []
    for i, ln in enumerate(window, start=offset):
        if len(ln) > MAX_LINE_LEN:
            ln = ln[:MAX_LINE_LEN] + f"… [+{len(ln) - MAX_LINE_LEN} chars]"
        rendered.append(f"{i:>6}\t{ln}")
    shown_end = offset - 1 + len(window)
    header = f"{p} ({total} lines"
    if offset > 1 or shown_end < total:
        header += f"; showing {offset}-{shown_end}"
    header += ")"
    return _truncate(header + "\n" + "\n".join(rendered))


def _list_dir(args: dict) -> str:
    p = _resolve(args.get("path", "."))
    if not p.exists():
        return f"ERROR: path not found: {p}"
    if p.is_file():
        return f"ERROR: {p} is a file, not a directory."
    try:
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except Exception as e:  # noqa: BLE001
        return f"ERROR listing {p}: {e}"
    rows = []
    for e in entries:
        if e.is_dir():
            rows.append(f"{e.name}/")
        else:
            try:
                rows.append(f"{e.name}  {_human_bytes(e.stat().st_size)}")
            except OSError:
                rows.append(e.name)
    body = "\n".join(rows) if rows else "(empty)"
    return _truncate(f"{p} ({len(entries)} entries)\n{body}")


def _glob(args: dict) -> str:
    pattern = str(args["pattern"])
    root = _resolve(args.get("path", "."))
    try:
        matches = [m for m in root.glob(pattern) if m.is_file()]
    except Exception as e:  # noqa: BLE001
        return f"ERROR globbing '{pattern}': {e}"
    matches = [m for m in matches if not any(part in IGNORE_DIRS for part in m.parts)]
    if not matches:
        return f"No files match '{pattern}' under {root}"
    matches.sort(key=lambda m: m.stat().st_mtime if m.exists() else 0.0, reverse=True)
    shown = matches[:GLOB_CAP]
    body = "\n".join(str(m) for m in shown)
    if len(matches) > GLOB_CAP:
        body += f"\n… [{len(matches) - GLOB_CAP} more]"
    return body


def _grep(args: dict) -> str:
    pattern = str(args["pattern"])
    path = str(args.get("path", "."))
    glob_filter = args.get("glob")
    ci = bool(args.get("case_insensitive", False))

    rg = shutil.which("rg")
    lines: list[str] = []
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color=never"]
        if ci:
            cmd.append("--ignore-case")
        if glob_filter:
            cmd += ["--glob", str(glob_filter)]
        cmd += ["--", pattern, str(_resolve(path))]
        proc = None
        try:
            proc = subprocess.Popen(
                cmd, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            if proc is not None:
                _kill_process_group(proc, 5)
            return "ERROR: grep timed out after 30s"
        if proc.returncode > 1:  # 0=matches, 1=no matches, >1=error
            return f"ERROR (ripgrep): {(err or '').strip()[:300]}"
        lines = (out or "").splitlines()
    else:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ci else 0)
        except re.error as e:
            return f"ERROR: invalid regex: {e}"
        root = _resolve(path)
        files = [root] if root.is_file() else _walk(root, str(glob_filter) if glob_filter else None)
        for f in files:
            try:
                for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(ln):
                        lines.append(f"{f}:{i}:{ln}")
            except (UnicodeDecodeError, OSError):
                continue
            if len(lines) >= GREP_CAP * 2:
                break

    if not lines:
        return f"No matches for /{pattern}/"
    total = len(lines)
    body = "\n".join(
        (ln[:MAX_LINE_LEN] + "…") if len(ln) > MAX_LINE_LEN else ln
        for ln in lines[:GREP_CAP]
    )
    if total > GREP_CAP:
        body += f"\n… [{total - GREP_CAP} more matches]"
    return _truncate(f"{total} match(es)\n{body}")


def _edit_file(args: dict) -> str:
    p = _resolve(args["path"])
    old = args.get("old_string", "")
    new = args.get("new_string", "")
    replace_all = bool(args.get("replace_all", False))
    if old == new:
        return "ERROR: old_string and new_string are identical; nothing to do."
    if old == "":
        return "ERROR: old_string is empty. Use create_file to make a new file."
    try:
        content = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: file not found: {p}. Use create_file for new files."
    except Exception as e:  # noqa: BLE001
        return f"ERROR reading {p}: {e}"

    count = content.count(old)
    if count == 0:
        return ("ERROR: old_string not found. It must match the file exactly, "
                "including whitespace and indentation — read the file and copy it verbatim.")
    if count > 1 and not replace_all:
        return (f"ERROR: old_string matches {count} places. Add surrounding lines "
                f"to make it unique, or set replace_all=true to change all of them.")

    new_content = content.replace(old, new) if replace_all else content.replace(old, new, 1)
    try:
        _atomic_write(p, new_content)
    except Exception as e:  # noqa: BLE001
        return f"ERROR writing {p}: {e}"
    n = count if replace_all else 1
    diff = _unified(content, new_content, str(p))
    return _truncate(f"Edited {p} ({n} replacement{'s' if n != 1 else ''}).\n{diff}")


def _create_file(args: dict) -> str:
    p = _resolve(args["path"])
    content = args.get("content", "")
    overwrite = bool(args.get("overwrite", False))
    if p.exists() and not overwrite:
        return f"ERROR: {p} already exists. Use edit_file to modify it, or set overwrite=true."
    try:
        _atomic_write(p, content)
    except Exception as e:  # noqa: BLE001
        return f"ERROR writing {p}: {e}"
    return f"{'Overwrote' if overwrite else 'Created'} {p} ({len(content)} bytes, {len(content.splitlines())} lines)."


def _run_bash(args: dict) -> str:
    cmd = str(args["command"])
    timeout = min(max(1, int(args.get("timeout", 120) or 120)), 600)
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, shell=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if proc is not None:
            _kill_process_group(proc, 5)
        return f"ERROR: `{cmd}` timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return f"ERROR running `{cmd}`: {e}"
    out_text = (out or "") + (err or "")
    status = "exit 0" if proc.returncode == 0 else f"exit {proc.returncode}"
    return _truncate(f"[{status}]\n{out_text}".rstrip())


# --- memory path, set by the CLI on startup --------------------------
MEMORY_PATH: Optional[Path] = None


def _save_memory(args: dict) -> str:
    """Save a fact or preference to persistent memory for future sessions.
    Does NOT alter the current session's system prompt, so the prefix cache
    is never invalidated."""
    text = str(args.get("text", "")).strip()
    if not text:
        return "ERROR: memory text must not be empty."
    from hrns import memory
    if MEMORY_PATH is None:
        return "ERROR: memory path is not configured."
    note = memory.add(MEMORY_PATH, text)
    return f"Saved memory {note['id']}: {text[:120]}"


_DISPATCH: dict[str, Callable[[dict], str]] = {
    "read_file": _read_file,
    "list_dir": _list_dir,
    "glob": _glob,
    "grep": _grep,
    "edit_file": _edit_file,
    "create_file": _create_file,
    "run_bash": _run_bash,
    "save_memory": _save_memory,
}


def execute(name: str, raw_args: str, confirm: Callable[..., bool]) -> str:
    """Run a tool.

    `confirm(name, args, reason) -> bool` gates (a) every mutating tool and
    (b) any tool — read-only included — whose target path escapes the workspace
    root. `reason` is "outside-workspace" for the latter case, otherwise None.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as e:
        return f"ERROR: could not parse arguments for {name}: {e}"
    if not isinstance(args, dict):
        return f"ERROR: arguments for {name} must be a JSON object."

    escape = _escape_path(name, args)
    reason = "outside-workspace" if escape is not None else None
    needs_confirm = name not in READ_ONLY or reason is not None
    if needs_confirm and not confirm(name, args, reason):
        if reason:
            return (f"User declined: {name} would access '{escape}', which is outside "
                    f"the workspace ({WORKSPACE_ROOT}). Stay within the project directory.")
        return f"User declined to run {name}."
    try:
        return fn(args)
    except KeyError as e:
        return f"ERROR: {name} is missing required argument {e}."
    except Exception as e:  # noqa: BLE001 — a bad tool call must not crash the loop
        return f"ERROR: {name} failed: {e}"
