"""hrns REPL: chat loop, slash commands, and the cache-aware status line.

Input is never blocked: while a turn is running, a TypeAhead reader keeps
capturing keystrokes — Enter queues the line as the next prompt, and anything
left half-typed pre-fills the prompt once the turn finishes.
"""

from __future__ import annotations

import base64
import codecs
import difflib
import getpass
import itertools
import json
import mimetypes
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

try:  # raw key reading for Shift+Tab; absent on non-Unix terminals
    import termios
    import tty
except ImportError:  # pragma: no cover
    termios = None  # type: ignore
    tty = None  # type: ignore

from hrns import __version__, memory
from hrns.client import ChatResult, DeepSeekClient, DeepSeekError
from hrns.config import Config, PROVIDER_LABELS, Provider, context_window, pricing_for
from hrns.session import Session, list_sessions
from hrns.tools import (
    TOOL_SCHEMAS,
    execute,
    resolve_target,
    set_workspace_root,
    workspace_root,
)
import hrns.tools as tools

# --- ANSI styling (degrades to no-op if not a tty) --------------------
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def dim(t: str) -> str: return _c("2", t)
def gray(t: str) -> str: return _c("38;5;242", t)
def bold(t: str) -> str: return _c("1", t)
def cyan(t: str) -> str: return _c("36", t)
def green(t: str) -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str) -> str: return _c("31", t)
def blue(t: str) -> str: return _c("34", t)
def magenta(t: str) -> str: return _c("35", t)


def _human(n: int) -> str:
    """Compact token counts: 950, 12.3k, 1.4M."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def _money(c: float) -> str:
    if c >= 1:
        return f"${c:.2f}"
    if c >= 0.01:
        return f"${c:.4f}"
    return f"${c:.6f}"


_CACHE_TTL = 300  # DeepSeek KV cache TTL in seconds


def _cache_age(session: Session) -> str:
    """Colored countdown until the KV cache expires. Each API request resets
    the clock; DeepSeek's TTL is ~5 minutes.

    green > 3 min, yellow > 1 min, red ≤ 1 min."""
    try:
        updated = datetime.fromisoformat(session.updated_at)
    except (ValueError, TypeError):
        return dim("cache --")
    age = (datetime.now(timezone.utc) - updated).total_seconds()
    remaining = max(0, _CACHE_TTL - age)
    if age < 0 or remaining <= 0:
        return red("cache expired")
    m, s = divmod(int(remaining), 60)
    label = f"cache {m}m {s}s" if m else f"cache {s}s"
    if remaining > 180:
        return green(label)
    if remaining > 60:
        return yellow(label)
    return red(label)


def _divider() -> str:
    """A dim rule spanning the full terminal width (re-measured per call,
    so it tracks live resizes)."""
    return dim("─" * shutil.get_terminal_size().columns)


# --- the input box (drawn inside the bottom dock) -----------------------
#   ┌─ ⏺ 3s · $0.000042 · 12.3k · 3.4k/88 · calling api
#   │ the user types in here
#   └─ hrns · main · +2 -1 · chat · 95.0% · cache 4m 58s · $0.01 · $10.00 · 12.3k ctx / 48k cum · 1.2% · auto
def _box_top(content: str) -> str:
    return dim("┌─ ") + content


def _box_mid(content: str) -> str:
    return dim("│ ") + content


def _box_bottom(content: str = "") -> str:
    if content:
        return dim("└─ ") + content
    return dim("└─")


def _layout(text: str, cursor: int, width: int) -> "tuple[list[str], int, int]":
    """Lay `text` out into display rows of at most `width` columns — wrapping
    long logical lines and honoring embedded newlines — and locate the cursor.

    Returns (rows, cursor_row, cursor_col). A cursor sitting exactly on a wrap
    boundary belongs to column 0 of the following row, and a trailing newline
    yields a final empty row so the cursor stays visible after it."""
    width = max(1, width)
    rows: list[str] = []
    cur_row = cur_col = 0
    idx = 0                                   # global index of the line's start
    for line in text.split("\n"):
        start = 0
        while True:
            chunk = line[start:start + width]
            lo = idx + start                  # global index of the chunk's head
            hi = lo + len(chunk)
            more = start + width < len(line)  # this logical line wraps further
            if lo <= cursor <= hi and not (more and cursor == hi):
                cur_row, cur_col = len(rows), cursor - lo
            rows.append(chunk)
            if not more:
                break
            start += width
        idx += len(line) + 1                  # +1 for the consumed newline
    return rows, cur_row, cur_col


# --- the bottom dock ----------------------------------------------------
class _Dock:
    """The UI pinned to the bottom of the terminal: a top status border, one
    or more input rows, and a bottom border.

    A DECSTBM scroll region confines normal output to the rows above, so
    replies and meta scroll there while the status row, divider, and input
    field stay put on the screen's last rows — even mid-reasoning. The dock
    grows and shrinks as the input field gains and loses lines (see
    `_dock_ensure`), so a pasted or multi-line draft is shown in full.
    """
    active = False
    rows = 0          # terminal row count when the dock was last established
    height = 3        # dock rows: top border + N input rows + bottom border


def _dock_ensure(parts: list[str], height: int = 3) -> int:
    """Append the escapes that (re)establish the dock at `height` rows (≥3:
    top border + ≥1 input rows + bottom border), growing or shrinking it in
    place. Returns the terminal row count, or 0 when the terminal is too short
    to dock."""
    size = shutil.get_terminal_size()
    if size.lines < 8:
        return 0
    height = max(3, min(height, size.lines - 5))   # keep some flow visible
    if _Dock.active and _Dock.rows == size.lines and _Dock.height == height:
        return size.lines
    if _Dock.active and _Dock.rows == size.lines:
        _dock_reflow(parts, _Dock.height, height, size.lines)
    else:
        if _Dock.active:                  # resized — release the old margins
            parts.append("\0337\033[r\0338")
        parts.append("\n" * height + f"\033[{height}A")   # free the bottom rows
        parts.append("\0337" + f"\033[1;{size.lines - height}r" + "\0338")
    _Dock.active = True
    _Dock.rows = size.lines
    _Dock.height = height
    return size.lines


def _dock_reflow(parts: list[str], old: int, new: int, lines: int) -> None:
    """Resize the dock from `old` to `new` rows on the same terminal. Growth
    scrolls the flow up so its tail slides into scrollback; shrink scrolls it
    back down to fill the rows the dock gives up. Either way the cursor is left
    on the flow's last usable row (just above the new dock) so a following
    print lands correctly."""
    delta = new - old
    parts.append("\033[r")                     # full screen, so the scroll lands
    if delta > 0:
        parts.append(f"\033[{lines};1H" + "\n" * delta)   # bottom: scroll up
    else:
        parts.append("\033[1;1H" + "\033M" * (-delta))    # top: scroll down
    parts.append(f"\033[1;{lines - new}r")     # re-pin the resized flow region
    parts.append(f"\033[{lines - new};1H")     # park on the flow's last row


def _undock() -> None:
    """Clear the docked rows and release the scroll margins (on exit)."""
    if not _Dock.active:
        return
    top = _Dock.rows - _Dock.height + 1
    sys.stdout.write(f"\0337\033[{top};1H\033[J\033[r\0338\033[?25h")
    sys.stdout.flush()
    _Dock.active = False


def _block(row: str, col: int, ghost: str = "") -> str:
    """Render `row` with an inverse-video block standing in for the hardware
    cursor at `col` (used while the hw cursor is hidden in the flow). Past the
    end of `row` the block sits on the first ghost char, or on a trailing
    space."""
    if col < len(row):
        return row[:col] + _c("7", row[col]) + row[col + 1:]
    if ghost:
        return row + _c("7", ghost[0]) + dim(ghost[1:])
    return row + _c("7", " ")


def _confirm_body(prompt: str, text: str, cursor: int, cols: int) -> str:
    """One-row body for a confirm question borrowing the field: the question,
    then the typed answer windowed so the cursor stays on screen, with an
    inverse block cursor."""
    plain_len = len(_ANSI.sub("", prompt))
    disp = text.replace("\n", " ")
    dcur = min(cursor - text[:cursor].count("\n"), len(disp))
    width = max(10, cols - plain_len - 5)
    start = 0
    if len(disp) >= width:
        start = min(max(0, dcur - width + 1), len(disp) - width)
    visible = disp[start:start + width]
    return prompt + ("…" if start else "") + _block(visible, dcur - start)


def _paint_input_box(parts: list[str], head: str, text: str, cursor: int,
                     *, prompt: str = "", block_cursor: bool = False,
                     foot: str = "") -> int:
    """Draw the bottom input dock — `head` on the top border, the field body,
    then `foot` on the bottom border — growing the dock to fit. Returns the
    terminal row count, or 0 when the terminal is too short (caller falls back
    to an inline prompt).

    Without `prompt` the body is `text` wrapped across as many input rows as it
    needs, so a multi-line or pasted draft shows in full. With `prompt` (a
    confirm question borrowing the field) the body stays on one row.

    block_cursor=True draws the cursor cell inverse-video and leaves the hw
    cursor hidden (during-turn type-ahead); False parks the visible hw cursor
    at the cell (interactive prompt)."""
    size = shutil.get_terminal_size()
    cols, lines = size.columns, size.lines
    ghost = ""
    if prompt:
        rows: list[str] = [_confirm_body(prompt, text, cursor, cols)]
        cur_row = cur_col = 0
    else:
        width = max(8, cols - 2)              # the "│ " mid-row prefix is 2 cols
        rows, cur_row, cur_col = _layout(text, cursor, width)
        if len(rows) == 1 and cursor == len(text):   # faded /command preview
            ghost = _command_ghost(text)[:max(0, width - len(rows[0]))]

    max_in = max(1, lines - 7)                # most input rows the dock can show
    if len(rows) > max_in:                    # window so the cursor stays shown
        start = min(max(0, cur_row - max_in + 1), len(rows) - max_in)
        rows, cur_row = rows[start:start + max_in], cur_row - start

    rows_n = _dock_ensure(parts, len(rows) + 2)
    if rows_n == 0:
        return 0
    if len(rows) > _Dock.height - 2:          # dock capped tighter than asked
        keep = _Dock.height - 2
        start = min(max(0, cur_row - keep + 1), len(rows) - keep)
        rows, cur_row = rows[start:start + keep], cur_row - start
    top = rows_n - _Dock.height + 1           # the top-border row
    # block_cursor: hide the hw cursor and stash the flow spot (so the turn's
    # next print lands there); the field paints its own inverse block. Hide
    # BEFORE saving — some terminals fold visibility into the DECSC state, so a
    # save-then-hide gets undone by every later restore.
    parts.append("\033[?25l\0337" if block_cursor else "\033[?25h")
    parts.append(f"\033[{top};1H\033[K" + _box_top(head))
    for i, row in enumerate(rows):
        g = ghost if i == len(rows) - 1 else ""
        if prompt:
            body = row                        # block already embedded
        elif block_cursor and i == cur_row:
            body = _block(row, cur_col, g)
        else:
            body = row + (dim(g) if g else "")
        parts.append(f"\033[{top + 1 + i};1H\033[K" + _box_mid(body))
    parts.append(f"\033[{rows_n};1H\033[K" + _box_bottom(foot))
    if block_cursor:
        parts.append("\0338")                 # hw cursor back to the flow
    else:
        parts.append(f"\033[{top + 1 + cur_row};{3 + cur_col}H")   # park on cell
    return rows_n


# --- "working…" spinner with an elapsed timer -------------------------
class Spinner:
    """A self-erasing live region animated from a background thread.

    The main thread does the blocking work and calls .set(label) to describe
    what's happening ("thinking", "reading config.py"). The whole noisy
    intermediate stream (reasoning, tool I/O) is collapsed into the status row.

    When `input_state` is set (type-ahead active), it paints the input box
    into the bottom dock, growing it to fit a multi-line draft:

        ┌─ auto · ⠋ reasoning 3s [1 queued]   <- mode (via `lead`) + status
        │ the user's draft                     <- input row(s), block cursor
        └─

    Without `input_state` it degrades to the classic single status row.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, state: State) -> None:
        self.enabled = _TTY
        self.state = state
        self._label = "working"
        self._lock = threading.Lock()
        self._io = threading.Lock()  # serialises stdout between threads
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._frame = self.FRAMES[0]
        self._region_live = False  # the 3-row region is currently on screen
        self.elapsed = 0.0
        # optional extra text appended to the status row (queued badge)
        self.extra: Optional[Callable[[], str]] = None
        # leading segment on the box's top border (the approval mode)
        self.lead: Optional[Callable[[], str]] = None
        # when set, supplies the input field's (text, cursor, confirm_prompt)
        self.input_state: Optional[Callable[[], "tuple[str, int, Optional[str]]"]] = None
        # prompt token count accumulated during this turn (set from run_turn)
        self.prompt_tokens: Optional[int] = None
        # prompt cost accumulated during this turn (set from run_turn)
        self.prompt_cost: Optional[float] = None
        # per-request cache hit / miss tokens (set from run_turn)
        self.cache_hit_tokens: Optional[int] = None
        self.cache_miss_tokens: Optional[int] = None

    def start(self, label: str = "working") -> None:
        self._label = label
        self._t0 = time.monotonic()
        if not self.enabled:
            return
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set(self, label: str) -> None:
        with self._lock:
            self._label = label

    def set_prompt_tokens(self, n: int) -> None:
        self.prompt_tokens = n

    def set_prompt_cost(self, c: float) -> None:
        self.prompt_cost = c

    def set_cache_stats(self, hit: int, miss: int) -> None:
        self.cache_hit_tokens = hit
        self.cache_miss_tokens = miss

    def last_head(self) -> str:
        """The final top-border line, dimmed — shown after the turn ends."""
        elapsed = self.elapsed
        parts = [f"⏺ {dim(f'{elapsed:.0f}s')}"]
        if self.prompt_cost is not None:
            parts.append(dim(f"${self.prompt_cost:.6f}"))
        if self.prompt_tokens is not None:
            parts.append(dim(_human(self.prompt_tokens)))
        if self.cache_hit_tokens is not None and self.cache_miss_tokens is not None:
            parts.append(dim(f"{_human(self.cache_hit_tokens)}/{_human(self.cache_miss_tokens)}"))
        parts.append(dim(self._label))
        extra_fn = self.extra
        if extra_fn:
            parts.append(dim(extra_fn()))
        return dim(" · ").join(parts)

    def _run(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            self._frame = frame
            if not self._paused.is_set():
                self._render()
            time.sleep(0.09)

    def _render(self) -> None:
        with self._lock:
            label = self._label
        extra_fn = self.extra
        extra = extra_fn() if extra_fn else ""
        state_fn = self.input_state
        elapsed = time.monotonic() - self._t0
        self.elapsed = elapsed
        head = f"{cyan(self._frame)} {dim(f'{elapsed:.0f}s')}"
        if self.prompt_cost is not None:
            head += dim(" · ") + yellow(f"${self.prompt_cost:.6f}")
        if self.prompt_tokens is not None:
            head += dim(" · ") + magenta(_human(self.prompt_tokens))
        if self.cache_hit_tokens is not None and self.cache_miss_tokens is not None:
            head += dim(" · ") + green(_human(self.cache_hit_tokens)) \
                + dim("/") + red(_human(self.cache_miss_tokens))
        head += dim(" · ") + label
        if extra:
            head += " " + extra
        with self._io:
            if self._paused.is_set() or self._stop.is_set():
                return
            if state_fn is None:
                sys.stdout.write(f"\r\033[K{head}")
                sys.stdout.flush()
                return
            text, cursor, req = state_fn()
            foot = statusline(self.state)
            if self.lead:
                foot += dim(" · ") + self.lead()
            parts: list[str] = []
            rows = _paint_input_box(parts, head, text, cursor,
                                    prompt=req or "", block_cursor=True,
                                    foot=foot)
            if rows == 0:                 # terminal too short to dock
                flat = (req or "") + text.replace("\n", " ")
                sys.stdout.write(f"\r\033[K{head}  {flat}")
                sys.stdout.flush()
                return
            sys.stdout.write("".join(parts))
            sys.stdout.flush()
            self._region_live = True

    def render_now(self) -> None:
        """Redraw immediately — used to echo a type-ahead keystroke."""
        if self.enabled and self._thread is not None \
                and not self._stop.is_set() and not self._paused.is_set():
            self._render()

    def _clear_live(self) -> None:
        """Stop painting the live rows. The dock itself persists (the next
        owner repaints it); just unhide the hw cursor. Must hold _io."""
        if self._region_live:
            sys.stdout.write("\033[?25h")
            self._region_live = False
        else:
            sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def pause(self) -> None:
        """Hide the live rendering so the main thread can prompt for input."""
        if not self.enabled:
            return
        self._paused.set()
        with self._io:
            self._clear_live()

    def resume(self) -> None:
        self._paused.clear()  # next tick repaints

    def println(self, text: str) -> None:
        """Print a permanent line into the scrolling flow above the dock."""
        if self._region_live:
            with self._io:
                print(text)  # margins keep it above the docked rows
            return
        self.pause()
        with self._io:
            print(text)
        self.resume()

    def stop(self) -> None:
        self.elapsed = time.monotonic() - self._t0
        if not self.enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join()
        with self._io:
            self._clear_live()


# --- lightweight markdown rendering for the terminal ------------------
_FENCE = re.compile(r"^\s*```(\w*)\s*$")
_HEADER = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_HRULE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_CODE_SPAN = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")


def _inline(text: str) -> str:
    text = _CODE_SPAN.sub(lambda m: _c("36", m.group(1)), text)        # `code`
    text = _BOLD.sub(lambda m: _c("1", m.group(1)), text)               # **bold**
    return text


def format_markdown(md: str) -> str:
    """Render a subset of markdown to ANSI. Returns raw text when not a tty."""
    md = (md or "").strip("\n")
    if not _TTY:
        return md.strip()

    out: list[str] = []
    in_code = False
    for line in md.split("\n"):
        fence = _FENCE.match(line)
        if fence:
            if not in_code:
                in_code = True
                out.append(dim("  ┌─ " + (fence.group(1) or "code")))
            else:
                in_code = False
                out.append(dim("  └─"))
            continue
        if in_code:
            out.append(dim("  │ ") + _c("36", line))
            continue
        if _HRULE.match(line):
            out.append(dim("─" * 48))
            continue
        h = _HEADER.match(line)
        if h:
            out.append(bold(cyan(h.group(2))))
            continue
        b = _BULLET.match(line)
        if b:
            out.append(f"{b.group(1)}{cyan('•')} {_inline(b.group(2))}")
            continue
        o = _ORDERED.match(line)
        if o:
            out.append(f"{o.group(1)}{cyan(o.group(2) + '.')} {_inline(o.group(3))}")
            continue
        out.append(_inline(line))
    return "\n".join(out)


BASE_SYSTEM_PROMPT = """\
You are hrns, a precise command-line coding assistant powered by DeepSeek,
working inside the user's project in the current working directory.

# Tools
- grep — search file contents by regex to locate code fast.
- glob — find files by name pattern (e.g. **/*.py).
- list_dir — see what's in a directory.
- read_file — read a file (returned with line numbers) before changing it.
- edit_file — change a file via an exact, unique string replacement.
- create_file — add a new file.
- run_bash — run builds, tests, git, and other shell commands. Always preface the command with a one-liner explaining why it needs to run (e.g. "check the build passes" or "verify no regressions").
- save_memory — save a persistent fact or preference for future sessions (applies to new sessions only, never the current one).

# How to work
- Explore before you act. Use grep/glob/list_dir/read_file to understand the
  code and match its existing conventions, libraries, and style. Don't guess at
  APIs or file contents — go look.
- Always read a file immediately before editing it, and copy its whitespace and
  indentation verbatim into edit_file's old_string so the match is exact and
  unique. If an edit fails, re-read and try again with more surrounding context.
- Make the smallest correct change. Don't reformat or refactor unrelated code,
  and don't add comments unless they earn their place.
- Verify your work. When it's possible, run the relevant tests, build, or linter
  with run_bash and fix anything you broke before declaring done.
- edit_file, create_file, and run_bash require user confirmation. If the user
  declines a step, stop and ask rather than working around it.

# Safety
- Some actions are destructive or hard to undo. Do NOT run them unless the user
  explicitly asked for that specific action in this conversation:
  - deleting files/dirs (`rm -rf`, `find … -delete`, `git clean -fd`);
  - remote or history-rewriting git ops (`git push`, especially `--force`;
    `git reset --hard`; `git checkout -- .`; amending/rebasing pushed commits;
    deleting branches);
  - discarding or overwriting uncommitted work, or mass find-and-replace across
    many files at once;
  - `sudo`, `chmod`/`chown -R`, package publishes/deploys, or piping the network
    into a shell (`curl … | sh`).
- Prefer the reversible path: show a dry run or a diff, scope changes narrowly,
  and confirm the exact target before any one-way operation. If you're unsure
  whether something can be undone, ask first.
- Stay inside the workspace (the folder hrns was opened in). Access to any path
  outside it is blocked unless the user approves that specific path, so don't
  plan on reaching outside. Don't read or print secrets (`.env`, credentials,
  keys, tokens), and never send code or data to external services.
- The confirmation gate is a backstop, not a license: it's on you not to propose
  a dangerous command in the first place. Flag the risk and offer a safer
  alternative instead.

# Style
- Be concise and direct. Briefly say why before a tool call, not after.
- Reply in GitHub-flavored markdown. Reference code as `path:line`.
- When you finish, give a short summary of what changed — not a play-by-play.

# Memory
- You have a `save_memory` tool for persistent cross-session memory. Use it to
  remember user preferences, conventions, and patterns they explicitly stated or
  demonstrated across multiple turns.
- Memory applies to FUTURE sessions only — it does NOT alter the current
  conversation, so the prefix cache stays valid.
- Rules for what to save:
  • DO save: explicit user preferences ("I prefer tabs"), tech-stack choices
    ("we use pytest"), recurring patterns (always uses `git commit -m` not `-m`),
    project conventions the user enforces.
  • DO NOT save: one-off requests, temporary workarounds, the current task's
    context, anything the user hasn't confirmed they want long-term, speculative
    inferences about the user.
- When in doubt, don't save. The user can always `/memory add` manually.
- Write memories in third person: "The user prefers X over Y."

Note: do not assume the current date or environment; inspect via tools when it
matters. Volatile facts are deliberately kept out of this prompt so the cached
prefix stays identical across turns and resumed sessions."""


def build_system_prompt(cfg: Config) -> str:
    """Static base + a snapshot of persistent memory. Frozen per session."""
    return BASE_SYSTEM_PROMPT + memory.as_prompt_block(cfg.memory_path)


@dataclass
class State:
    cfg: Config
    session: Session
    client: Optional[DeepSeekClient]
    approval_mode: str = "confirm"
    # directories outside the workspace the user has approved for this session
    # (so repeat access to the same area isn't re-prompted). Not persisted.
    approved_dirs: set[Path] = field(default_factory=set)
    # the last turn's top-border stats, shown dimmed when idle
    last_head: str = ""


# --- approval modes (cycled with Shift+Tab) ---------------------------
APPROVAL_MODES = ["confirm", "auto-edit", "auto"]
_MODE_DESC = {
    "confirm": "ask before every edit, file write, and command",
    "auto-edit": "auto-approve file edits/creates · still ask for shell & outside-workspace",
    "auto": "auto-approve all in-workspace actions · still ask for outside-workspace",
}


def _cycle_mode(state: State) -> None:
    i = APPROVAL_MODES.index(state.approval_mode)
    state.approval_mode = APPROVAL_MODES[(i + 1) % len(APPROVAL_MODES)]
    state.cfg.approval_mode = state.approval_mode
    state.cfg.save()


def mode_badge(mode: str) -> str:
    label = {
        "confirm": dim("confirm"),
        "auto-edit": yellow("auto-edit"),
        "auto": _c("1;31", "auto"),
    }.get(mode, "")
    return label + " "


def _auto_approves(mode: str, name: str, reason: Optional[str]) -> bool:
    # The workspace boundary always asks, even in auto — that promise stands.
    if reason == "outside-workspace":
        return False
    # save_memory only writes to the memory file — never destructive
    if name == "save_memory":
        return True
    if mode == "auto-edit":
        return name in ("edit_file", "create_file")
    if mode == "auto":
        return True
    return False




def session_summary(s: Session) -> str:
    rate = s.cache_hit_rate * 100
    meta = (f"{s.turn_count()} turns · {s.usage['requests']} requests · "
            f"cache hit rate {rate:.0f}% · {_cache_age(s)} · updated {s.updated_at}")
    return (
        f"{cyan(s.id)}  {dim(s.model)}\n"
        f"   {s.title()}\n"
        f"   {dim(meta)}"
    )


# --- the agentic turn --------------------------------------------------
def _api_stats(model: str, usage: dict, elapsed: float) -> str:
    """Compact one-liner for a single API call."""
    p = pricing_for(model)
    hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    out = int(usage.get("completion_tokens", 0) or 0)
    cost = hit / 1e6 * p["cache_hit"] + miss / 1e6 * p["cache_miss"] + out / 1e6 * p["output"]
    return dim(f"  in {_human(hit + miss)} · out {_human(out)} · ${cost:.6f} · {elapsed:.0f}s")


def render_assistant(md: str) -> str:
    """Render a reply indented under a cyan bullet — visually subordinate
    to the magenta ▸ user prompt, creating a clear prompt/response divide."""
    body = format_markdown(md) if md else dim("(no text response)")
    lines = body.split("\n")
    return "\n".join([cyan("  ∙ ") + lines[0]] + ["    " + ln for ln in lines[1:]])


def _say(session: Session, kind: str, text: str, spinner: Optional[Spinner] = None) -> None:
    """Print a line AND record it in the session transcript, so resuming the
    session replays exactly what was on screen."""
    if spinner is not None:
        spinner.println(text)
    else:
        print(text)
    session.log(kind, text)


def _repair_dangling_tool_calls(session: Session) -> None:
    """After an interrupt, answer any tool_calls that never got results, so
    the append-only message log stays valid for the next request."""
    msgs = session.messages
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") == "user":
            return  # interrupted before the assistant replied — nothing dangles
        if m.get("role") == "assistant":
            answered = {t.get("tool_call_id") for t in msgs[i + 1:] if t.get("role") == "tool"}
            for tc in m.get("tool_calls") or []:
                if tc.get("id") not in answered:
                    session.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": "(interrupted by user before this tool ran)",
                    })
            return


# --- image attachments in the input -----------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
# tokens that may be a file path: a quoted string, or a run of non-space
# chars where '\ ' escapes embedded spaces (how terminals paste drag-drops).
_TOKEN_RE = re.compile(r"""'[^']*'|"[^"]*"|(?:\\.|[^\s'"])+""")


def _unquote_token(tok: str) -> str:
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "'\"":
        return tok[1:-1]
    return re.sub(r"\\(.)", r"\1", tok)  # un-escape '\ ', '\(' … from drag-drop


def _extract_images(text: str) -> tuple[str, list[Path]]:
    """Pull image-file references out of the input — drag-dropped or typed
    paths, quoted or backslash-escaped. Returns the prompt text with those
    tokens removed and the list of image files found. A token only counts when
    it resolves to an existing file with an image extension, so ordinary words
    are left untouched."""
    images: list[Path] = []

    def repl(m: "re.Match[str]") -> str:
        p = Path(_unquote_token(m.group(0))).expanduser()
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            images.append(p)
            return ""  # drop the path from the prompt text
        return m.group(0)

    cleaned = _TOKEN_RE.sub(repl, text)
    if not images:
        return text, []
    # tidy the gaps the removed tokens left, but keep author newlines intact
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = "\n".join(ln.strip() for ln in cleaned.splitlines()).strip()
    return cleaned, images


def _input_display(text: str) -> str:
    """How an input line is shown/recorded: image paths replaced by compact
    `[🖼 name]` markers so the scrollback isn't cluttered with long paths.
    Multi-line pastes are bolded per-line so ANSI codes don't bleed."""
    cleaned, images = _extract_images(text)
    if images:
        marks = " ".join(cyan(f"[🖼 {p.name}]") for p in images)
        text = (cleaned + " " + marks) if cleaned else marks
    return "\r\n".join(bold(ln) for ln in text.split("\n"))


def build_user_message(text: str) -> tuple[Any, int]:
    """Turn raw input into a chat `content` value, attaching referenced image
    files as base64 image_url blocks (OpenAI-style multimodal). Returns
    (content, n_images); content is a plain string when no image is attached,
    so the common path — and the cached prefix — is unchanged."""
    cleaned, images = _extract_images(text)
    if not images:
        return text, 0
    parts: list[dict[str, Any]] = []
    if cleaned:
        parts.append({"type": "text", "text": cleaned})
    attached = 0
    for p in images:
        try:
            raw = p.read_bytes()
        except OSError as e:
            print(yellow(f"  ⚠ couldn't read image {p.name}: {e}"))
            continue
        if len(raw) > MAX_IMAGE_BYTES:
            print(yellow(f"  ⚠ skipping {p.name} — exceeds the "
                         f"{MAX_IMAGE_BYTES // 1024 // 1024}MB image limit"))
            continue
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:{mime};base64,{b64}"}})
        attached += 1
    if attached == 0:
        return text, 0  # nothing usable attached — fall back to the raw text
    return parts, attached


def run_turn(state: State, user_input: str, typeahead: TypeAhead) -> None:
    content, _n_images = build_user_message(user_input)
    state.session.append({"role": "user", "content": content})
    cfg, session, client = state.cfg, state.session, state.client
    assert client is not None

    spinner = Spinner(state)
    typeahead.start(spinner)  # attach dock hooks first so no frame paints inline
    state.last_head = ""             # clear the idle ghost
    spinner.start("calling api")
    start_prompt_tokens = session.usage["prompt_tokens"]  # snapshot for accumulator
    start_hit = session.usage["prompt_cache_hit_tokens"]
    start_miss = session.usage["prompt_cache_miss_tokens"]
    p = pricing_for(session.model)
    final_text = ""
    reasoning_line_buf = ""       # partial line accumulator for streamed reasoning
    error: Optional[str] = None
    interrupted = False

    try:
        while True:
            buf: list[str] = []
            out_tok = 0

            def on_reasoning(t: str) -> None:
                nonlocal reasoning_line_buf
                spinner.set("reasoning")
                reasoning_line_buf += t
                while "\n" in reasoning_line_buf:
                    line, reasoning_line_buf = reasoning_line_buf.split("\n", 1)
                    ln = gray("  " + line.rstrip("\r"))
                    spinner.println(ln)
                    session.log("meta", ln)

            def on_text(t: str, _b: list = buf) -> None:
                nonlocal out_tok
                _b.append(t)
                out_tok += 1
                spinner.set(f"writing {out_tok} tok")

            try:
                result: ChatResult = client.stream_chat(
                    model=session.model,
                    messages=session.messages,
                    tools=TOOL_SCHEMAS,
                    temperature=cfg.temperature,
                    thinking={"type": "enabled"} if cfg.provider == "deepseek" else None,
                    on_text=on_text,
                    on_reasoning=on_reasoning,
                )
            except DeepSeekError as e:
                error = str(e)
                break

            session.append(result.message)
            session.record_usage(result.usage)
            spinner.set_prompt_tokens(session.usage["prompt_tokens"] - start_prompt_tokens)
            dh = session.usage["prompt_cache_hit_tokens"] - start_hit
            dm = session.usage["prompt_cache_miss_tokens"] - start_miss
            spinner.set_prompt_cost(dh / 1e6 * p["cache_hit"] + dm / 1e6 * p["cache_miss"])
            spinner.set_cache_stats(dh, dm)

            # Flush any partial reasoning line left in the buffer.
            if reasoning_line_buf:
                ln = gray("  " + reasoning_line_buf.rstrip("\r"))
                spinner.println(ln)
                session.log("meta", ln)
                reasoning_line_buf = ""

            tool_calls = result.message.get("tool_calls")
            if not tool_calls:
                final_text = result.message.get("content") or ""
                break

            for tc in tool_calls:
                fn = tc["function"]
                label = _tool_label(fn)
                spinner.set(label)
                # …and, for mutating tools, the diff/preview via the confirm gate
                out = execute(fn["name"], fn["arguments"],
                              confirm=_make_confirm(spinner, state, typeahead))
                if fn["name"] == "save_memory":
                    try:
                        mem_text = json.loads(fn.get("arguments", "{}")).get("text", "")[:80]
                    except (json.JSONDecodeError, TypeError):
                        mem_text = ""
                    if mem_text:
                        _say(session, "meta", green("  ✓ ") + mem_text, spinner)
                session.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            # loop so the model can read the tool results
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # If the user is mid-edit of a queued prompt, don't tear down and run
        # the next queued prompt underneath them — wait until they finish.
        # An interrupted turn (or Ctrl-C while waiting) cancels the edit.
        if typeahead.editing:
            if interrupted:
                typeahead.cancel_edit()
            else:
                spinner.set("paused — finish editing the queued prompt (Enter)")
                try:
                    typeahead.wait_until_not_editing()
                except KeyboardInterrupt:
                    typeahead.cancel_edit()
        state.last_head = spinner.last_head()  # freeze for the idle prompt
        spinner.stop()    # before the hooks detach, so no frame paints inline
        typeahead.stop()

    if interrupted:
        _repair_dangling_tool_calls(session)
        _say(session, "meta", yellow("✗ interrupted") + dim(" — partial turn saved"))
        dropped = typeahead.drop_all()
        if dropped:
            print(dim(f"  dropped {len(dropped)} queued prompt(s)"))
        session.save(cfg.sessions_dir)
        return

    if error:
        _say(session, "meta", red(f"deepseek error: {error}"))
        session.save(cfg.sessions_dir)
        return

    _say(session, "assistant", "\n" + render_assistant(final_text))
    session.save(cfg.sessions_dir)
    # refresh balance after spending
    if state.client:
        cfg.balance = state.client.get_balance()


def _short(s: str, n: int = 60) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def _tool_label(fn: dict) -> str:
    """Short human phrase shown on the spinner while a tool runs."""
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    name = fn.get("name", "tool")
    base = lambda key: Path(str(args.get(key, "?"))).name  # noqa: E731
    return {
        "read_file": f"read {base('path')}",
        "list_dir": f"list {args.get('path', '.')}",
        "glob": f"find {args.get('pattern', '?')}",
        "grep": f"search /{_short(str(args.get('pattern', '?')), 30)}/",
        "edit_file": f"edit {base('path')}",
        "create_file": f"create {base('path')}",
        "run_bash": f"run {_short(str(args.get('command', '')), 40)}",
        "save_memory": "saving a memory",
    }.get(name, name)


def _diff_preview(old: str, new: str, limit: int = 14) -> list[str]:
    """Colored +/- diff lines, limited to `limit` lines."""
    lines = []
    for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1):
        if ln.startswith(("---", "+++", "@@")):
            continue
        if ln.startswith("+"):
            lines.append(green(ln))
        elif ln.startswith("-"):
            lines.append(red(ln))
        else:
            lines.append(dim(ln))
    total = len(lines)
    if total > limit:
        lines = lines[:limit]
        lines.append(dim(f"… (+{total - limit} more lines)"))
    return lines


def _tilde(p: Path) -> str:
    """A path shown relative to $HOME (~/…) when possible, else absolute —
    far more readable than a long absolute path in a prompt."""
    home = Path.home()
    if p == home:
        return "~"
    try:
        return "~/" + str(p.relative_to(home))
    except ValueError:
        return str(p)


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except ValueError:
        return False


def _approval_dir(target: Path) -> Path:
    """The directory to offer remembering for an out-of-workspace target: the
    target itself when it's a directory, otherwise the directory holding it."""
    return target if target.is_dir() else target.parent


def _confirm_preview(name: str, args: dict, reason: Optional[str] = None) -> str:
    """Describe a gated action before asking to apply it."""
    if reason == "outside-workspace":
        target = resolve_target(str(args.get("path", ".")))
        verb = {
            "read_file": "read", "list_dir": "list", "glob": "search",
            "grep": "search", "edit_file": "edit", "create_file": "create",
        }.get(name, "access")
        return (yellow("  ⚠ ") + red(f"{name} wants to {verb} OUTSIDE the workspace:") + "\n"
                f"      {bold(_tilde(target))}\n"
                + dim(f"      workspace: {_tilde(workspace_root())}"))
    if name == "edit_file":
        path = str(args.get("path", "?"))
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        head = f"{magenta('edit')} {bold(path)}"
        extra = dim(" (replace_all)") if args.get("replace_all") else ""
        diff = "\n".join("    " + ln for ln in _diff_preview(old, new))
        return f"  {head}{extra}\n{diff}"
    if name == "create_file":
        content = args.get("content", "")
        lines = content.splitlines()
        preview = green("+ " + lines[0]) if lines else ""
        more = dim(f" +{len(lines)} lines") if len(lines) > 1 else ""
        return dim(f"  {green('create')} {bold(str(args.get('path', '?')))} {preview}{more}")
    if name == "run_bash":
        return dim(f"  {yellow('run')} {bold(str(args.get('command', '')))}")
    if name == "save_memory":
        return dim(f"  {magenta('memory')} {(args.get('text', '') or '')[:100]}")
    if name in ("read_file",):
        return dim(f"  {blue('read')} {bold(str(args.get('path', '?')))}")
    return dim(f"  {blue(name)} {dim(str(args))}")


def _make_confirm(spinner: Spinner, state: State, typeahead: "TypeAhead"):
    """Confirm gate; pauses the spinner to show a preview before applying.

    In an auto mode the action is shown (diff/preview) and auto-approved — except
    out-of-workspace access, which always asks unless its directory was already
    approved this session. Out-of-workspace prompts are multi-choice: allow once,
    allow that directory for the rest of the session, or deny. The answer is read
    through the type-ahead reader (it owns the keyboard while a turn runs); any
    half-typed draft is stashed and restored around the question.
    """
    def _ask(prompt: str, docked: bool) -> str:
        ans = typeahead.request_line(prompt).strip().lower()
        if docked:
            spinner.println(prompt + ans)  # echo into the flow, like replay
        state.session.log("meta", prompt + ans)
        return ans

    def confirm(name: str, args: dict, reason: Optional[str] = None) -> bool:
        session = state.session
        docked = spinner.input_state is not None
        sp = spinner if docked else None
        if not docked:
            spinner.pause()
        try:
            if reason == "outside-workspace":
                target = resolve_target(str(args.get("path", ".")))
                # already approved this area this session → allow silently.
                if any(_is_within(target, d) for d in state.approved_dirs):
                    return True
                _say(session, "meta", _confirm_preview(name, args, reason), sp)
                adir = _approval_dir(target)
                ans = _ask(yellow(f"  [y] allow once · [d] allow {_tilde(adir)} for "
                                  f"session · [N] deny "), docked)
                if ans in ("d", "a", "always"):
                    state.approved_dirs.add(adir)
                    _say(session, "meta",
                         green(f"  ✓ allowing {_tilde(adir)} for the rest of this session"), sp)
                    return True
                return ans in ("y", "yes")

            # in-workspace mutating tools (edit_file / create_file / run_bash / save_memory)
            if name in ("edit_file", "create_file"):
                _say(session, "meta", _confirm_preview(name, args, reason), sp)
            if _auto_approves(state.approval_mode, name, reason):
                return True
            return _ask(yellow("  apply? [y/N] "), docked) in ("y", "yes")
        finally:
            if not docked:
                spinner.resume()
    return confirm


# --- slash commands ----------------------------------------------------
def cmd_help(state: State, args: str) -> None:
    print(bold("Commands:"))
    for name, desc in [
        ("/sessions", "list saved sessions; /sessions <id|#> to resume one"),
        ("/new", "archive the current session and start a fresh one"),
        ("/connect", "configure & test the API connection (key, model, base url)"),
        ("/memory", "view memory; /memory add <text> | rm <id> | clear"),
        ("/model", "show or set the model (applies to new sessions)"),
        ("/mode", "cycle approval: confirm → auto-edit → auto (or Shift+Tab)"),
        ("/stats", "cumulative token + cache stats for this session"),
        ("/compact", "summarise history and replace it with a compact summary"),
        ("/help", "show this help"),
        ("/quit", "exit (sessions are saved automatically)"),
    ]:
        print(f"  {cyan(name):<22} {dim(desc)}")


def cmd_sessions(state: State, args: str) -> None:
    sessions = list_sessions(state.cfg.sessions_dir)
    if args.strip():
        # resume by id or 1-based index
        target = args.strip()
        chosen = None
        if target.isdigit() and 1 <= int(target) <= len(sessions):
            chosen = sessions[int(target) - 1]
        else:
            chosen = next((s for s in sessions if s.id == target), None)
        if not chosen:
            print(red(f"No session matching '{target}'."))
            return
        state.session = chosen
        state.cfg.model = chosen.model  # keep the line consistent
        _replay_conversation(state)
        print(green(f"\nResumed {chosen.id} ({chosen.turn_count()} turns). "
                    f"Its prefix will re-hit DeepSeek's cache."))
        return

    if not sessions:
        print(dim("No saved sessions yet."))
        return
    print(bold(f"{len(sessions)} session(s):"))
    for i, s in enumerate(sessions, 1):
        marker = green(" current") if s.id == state.session.id else ""
        print(f"{dim(f'{i:>2}.')} {session_summary(s)}{marker}")


def cmd_clear(state: State, args: str) -> None:
    state.session.save(state.cfg.sessions_dir)  # keep the old one
    state.session = Session.new(state.cfg.model, build_system_prompt(state.cfg))
    print(green(f"Started fresh session {state.session.id}. "
                f"(Previous one saved — find it with /sessions.)"))


def cmd_connect(state: State, args: str) -> None:
    cfg = state.cfg

    # Allow "/connect openrouter" or "/connect deepseek" to skip the picker
    if args.strip() in PROVIDER_LABELS:
        cfg.provider = args.strip()
    elif _raw_capable():
        labels = list(PROVIDER_LABELS.values())
        chosen = pick_from_list(labels, "Select provider")
        if chosen:
            # map label back to provider key
            for k, v in PROVIDER_LABELS.items():
                if v == chosen:
                    cfg.provider = k
                    break
        else:
            print(yellow("Selection cancelled."))
            return

    provider = cfg.provider
    label = PROVIDER_LABELS.get(provider, provider)
    print(bold(f"Connect to {label}"))

    # 1. api key
    have = "set" if cfg.api_key else "unset"
    key = getpass.getpass(f"  api key (currently {have}, blank = keep): ").strip()
    if key:
        cfg.api_key = key
    if not cfg.api_key:
        print(red("No API key available — cannot connect."))
        return

    # 2. connect and fetch models
    try:
        client = DeepSeekClient(cfg.api_key, cfg.base_url, cfg.provider)
        models = client.list_models()
    except DeepSeekError as e:
        print(red(f"Connection failed: {e}"))
        return

    # 3. pick model — interactive list if terminal supports it
    if models and _raw_capable():
        if cfg.model in models:
            models = [models.pop(models.index(cfg.model))] + models
        chosen = pick_from_list(models, f"Select model ({label})")
        if chosen:
            cfg.model = chosen
        else:
            print(yellow("Selection cancelled — keeping current model."))
    else:
        print(dim(f"  model [{cfg.model}]:"), end=" ")
        model = input().strip() or cfg.model
        cfg.model = model

    state.client = client
    cfg.save(include_key=True)
    ok = green("✓ connected")
    here = green("available") if cfg.model in models else yellow("not in /models list")
    print(f"{ok} · {len(models)} models · '{cfg.model}' {here}")
    print(dim(f"  saved to {cfg.config_path} — hrns will reconnect automatically next run"))
    print(dim(f"  to switch provider: /connect openrouter  or  export HRNS_PROVIDER=openrouter"))


def cmd_memory(state: State, args: str) -> None:
    path = state.cfg.memory_path
    parts = args.split(maxsplit=1)
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "add" and rest:
        n = memory.add(path, rest)
        print(green(f"Remembered ({n['id']}). Applies to new sessions."))
    elif sub == "rm" and rest:
        print(green("Removed.") if memory.remove(path, rest.strip()) else red("No such note id."))
    elif sub == "clear":
        memory.clear(path)
        print(green("Memory cleared."))
    else:
        notes = memory.list_notes(path)
        if not notes:
            print(dim("No memory yet. Add some with /memory add <text>."))
            return
        print(bold("Memory:"))
        for n in notes:
            print(f"  {cyan(n['id'])}  {n['text']}")


def cmd_model(state: State, args: str) -> None:
    from hrns.config import PRICING
    if not args.strip():
        print(f"Current model (new sessions): {bold(state.cfg.model)}")
        print(f"This session's model: {bold(state.session.model)}")
        print(dim("Known: " + ", ".join(PRICING.keys())))
        return
    state.cfg.model = args.strip()
    state.cfg.save()
    print(green(f"Default model set to {state.cfg.model}. Use /new to start a session on it."))


def cmd_mode(state: State, args: str) -> None:
    arg = args.strip()
    if arg:
        if arg not in APPROVAL_MODES:
            print(red(f"Unknown mode '{arg}'. Options: {', '.join(APPROVAL_MODES)}"))
            return
        state.approval_mode = arg
        state.cfg.approval_mode = arg
        state.cfg.save()
    else:
        _cycle_mode(state)
    print(f"approval mode: {mode_badge(state.approval_mode).strip()} "
          f"{dim('— ' + _MODE_DESC[state.approval_mode])}")


def cmd_stats(state: State, args: str) -> None:
    s = state.session
    u = s.usage
    print(bold(f"Session {s.id}"))
    print(f"  turns      {s.turn_count()}   requests {u['requests']}")
    print(f"  cache      {green(f'{u['prompt_cache_hit_tokens']:,} hit')} / "
          f"{u['prompt_cache_miss_tokens']:,} miss   "
          f"({s.cache_hit_rate*100:.1f}% hit rate)")
    print(f"  output     {u['completion_tokens']:,} tokens")
    win = context_window(s.model)
    fill = (s.context_tokens / win * 100) if win else 0.0
    print(f"  context    {s.context_tokens:,} / {win:,} tokens ({fill:.2f}% full)")
    p = pricing_for(s.model)
    saved = u["prompt_cache_hit_tokens"] / 1e6 * (p["cache_miss"] - p["cache_hit"])
    print(f"  cost       {_money(s.cost(p))}   {green(f'(cache saved {_money(saved)})')}")


def cmd_compact(state: State, args: str) -> None:
    """Summarise the conversation so far and replace history with the summary."""
    if state.client is None:
        print(red("Not connected. Run /connect first."))
        return

    session = state.session
    non_system = [m for m in session.messages if m.get("role") != "system"]
    if len(non_system) < 2:
        print(dim("Nothing to compact yet."))
        return

    summary_prompt = [session.messages[0]] + non_system + [
        {"role": "user", "content": "Summarise the conversation above concisely. "
         "Preserve all decisions, facts, and file paths. Use the same style the user would."}
    ]

    spinner = Spinner(state)
    spinner.start("compacting")
    t0 = time.monotonic()
    try:
        result = state.client.stream_chat(
            model=session.model,
            messages=summary_prompt,
            tools=None,
            temperature=0.2,
        )
    except DeepSeekError as e:
        spinner.stop()
        print(red(f"compact failed: {e}"))
        return
    finally:
        spinner.stop()

    summary = (result.message.get("content") or "").strip()
    elapsed = time.monotonic() - t0
    session.record_usage(result.usage)
    _say(session, "meta", dim("  compact ·") + _api_stats(session.model, result.usage, elapsed))

    system = session.messages[0]
    session.messages.clear()
    session.messages.append(system)
    session.messages.append({"role": "user", "content": summary})
    # approximate context — roughly the messages we now hold
    session.context_tokens = sum(len(str(m)) for m in session.messages) // 4
    _say(session, "meta", green(f"Compacted {len(non_system)} messages into a summary."))
    session.save(state.cfg.sessions_dir)

COMMANDS = {
    "/help": cmd_help, "/?": cmd_help,
    "/sessions": cmd_sessions,
    "/new": cmd_clear,
    "/connect": cmd_connect,
    "/memory": cmd_memory,
    "/model": cmd_model,
    "/mode": cmd_mode,
    "/stats": cmd_stats,
    "/compact": cmd_compact,
}


def _complete_command(ed: "LineEditor") -> Optional[list[str]]:
    """Tab-complete a slash command in the editor.

    A unique match completes fully (plus a trailing space for args); several
    matches extend to their longest common prefix. Returns the candidate list
    when no further progress is possible (caller shows it), else None.
    """
    text = ed.text()
    if not text.startswith("/") or " " in text:
        return None
    names = sorted((set(COMMANDS) | {"/quit"}) - {"/?"})
    matches = [c for c in names if c.startswith(text)]
    if not matches:
        return None
    if len(matches) == 1:
        ed.set_text(matches[0] + " ")
        return None
    common = os.path.commonprefix(matches)
    if len(common) > len(text):
        ed.set_text(common)
        return None
    return matches


def _command_ghost(text: str) -> str:
    """Inline suggestion: the unmatched remainder of the first command that
    `text` is a prefix of — rendered faded after the typed text. '' if none."""
    if not text.startswith("/") or " " in text:
        return ""
    names = sorted((set(COMMANDS) | {"/quit"}) - {"/?"})
    for c in names:
        if c.startswith(text) and len(c) > len(text):
            return c[len(text):]
    return ""


# --- the live status line shown above the prompt ----------------------
_MODEL_DISPLAY = {
    "deepseek-chat": "chat",
    "deepseek-reasoner": "reasoner",
    "deepseek-v4-flash": "v4 flash",
    "deepseek-v4-pro": "v4 pro",
}


def _model_name(model: str) -> str:
    return _MODEL_DISPLAY.get(model, model)


def _repo_info() -> tuple[str, str, int, int, int]:
    """(repo, branch, staged, unstaged, untracked) for the status bar, so each
    can be its own colored segment. `repo` is the git top-level's name (the
    cwd's name when not in a repo); the rest are empty/zero outside git.
    Cached for 5s to avoid spawning git on every keystroke."""
    _cache = getattr(_repo_info, "_cache", None)
    _ts = getattr(_repo_info, "_ts", 0.0)
    now = time.monotonic()
    if _cache is not None and now - _ts < 5.0:
        return _cache

    repo, branch, staged, unstaged, untracked = "", "", 0, 0, 0
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=1,
        ).stdout.strip()
        if top:
            repo = Path(top).name
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=1,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=1,
            ).stdout
            for l in status.splitlines():
                if len(l) < 2:
                    continue
                if l.startswith("??"):
                    untracked += 1
                elif l[0] != " ":
                    staged += 1
                elif l[1] != " ":
                    unstaged += 1
        else:
            repo = Path.cwd().name
    except Exception:
        repo = ""

    result = (repo, branch, staged, unstaged, untracked)
    _repo_info._cache = result
    _repo_info._ts = now
    return result


def statusline(state: State) -> str:
    """One colored line of session vitals, rendered above each input prompt.

    Order: model · cache-hit-rate · session-cost · balance · context/total
    tokens · repo · branch · git-stats. Each metric gets its own color so it's
    scannable at a glance:
      cyan=model  green=cache hit rate  yellow=cost/balance  magenta=tokens
      dim=repo  blue=branch  green/red/yellow=git +staged/-unstaged/?untracked
    """
    s = state.session
    u = s.usage
    hit, miss = u["prompt_cache_hit_tokens"], u["prompt_cache_miss_tokens"]
    cache_rate = (hit / (hit + miss) * 100) if (hit + miss) else None
    cost = s.cost(pricing_for(s.model))
    bal = state.cfg.balance
    cum_tok = u["prompt_tokens"] + u["completion_tokens"]
    repo, branch, staged, unstaged, untracked = _repo_info()

    gparts = []
    if staged:
        gparts.append(green(f"+{staged}"))
    if unstaged:
        gparts.append(red(f"-{unstaged}"))
    if untracked:
        gparts.append(yellow(f"?{untracked}"))

    segs = [
        dim(repo),                                                           # repo
        blue(branch),                                                        # branch
        " ".join(gparts),                                                    # git stats
        cyan(_model_name(s.model)),                                          # model
        green(f"{cache_rate:.1f}%" if cache_rate is not None else "--%"),    # cache hit rate
        _cache_age(s),                                                       # cache freshness
        yellow(_money(cost)),                                                # session cost
        yellow(f"${bal:.2f}" if bal is not None else "--"),                  # balance
        magenta(f"{_human(s.context_tokens)} ctx / {_human(cum_tok)} cum"),  # context / total
        dim(f"v{__version__}"),                                                  # version
    ]
    # context window usage %
    cw = context_window(s.model)
    if cw:
        ctx_pct = s.context_tokens / cw * 100
        segs.append(magenta(f"{ctx_pct:.1f}%"))
    return dim(" · ").join(seg for seg in segs if seg)


# --- input: line editing, type-ahead, and the prompt queue ------------
def _raw_capable() -> bool:
    return bool(_TTY and termios is not None and sys.stdin.isatty())


class _KeySource:
    """Keystrokes decoded straight from the stdin fd.

    Bypasses Python's buffered text layer so select() on the fd never lies
    about pending input (a paste would otherwise strand characters in the
    user-space buffer). One shared instance feeds both the interactive prompt
    and the type-ahead reader — they never run at the same time.
    """

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self._chars: deque[str] = deque()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def ready(self, timeout: float) -> bool:
        if self._chars:
            return True
        r, _, _ = select.select([self.fd], [], [], timeout)
        return bool(r)

    def getch(self, timeout: Optional[float] = None) -> str:
        """Next character; '' on EOF, or on timeout when one is given."""
        while not self._chars:
            if timeout is not None and not self.ready(timeout):
                return ""
            data = os.read(self.fd, 1024)
            if not data:
                return ""
            self._chars.extend(self._decoder.decode(data))
        return self._chars.popleft()


_KEY_SOURCE: Optional[_KeySource] = None


def _keys() -> _KeySource:
    global _KEY_SOURCE
    if _KEY_SOURCE is None:
        _KEY_SOURCE = _KeySource()
    return _KEY_SOURCE


def pick_from_list(items: list[str], title: str = "Select") -> str | None:
    """Interactive list picker — arrow keys, enter to select, esc to cancel.

    Renders directly into the terminal, restores the cursor position on exit.
    Returns the selected item or None on cancel. Requires a TTY."""
    n = len(items)
    if n == 0:
        return None
    if not _raw_capable():
        return None

    idx = 0
    query = ""         # filter buffer
    filtered = items   # items matching the query
    keys = _keys()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)

    # Drain any stale keystrokes buffered from the previous read_line
    while keys.getch(0):
        pass

    # Find terminal height so we know how many items to show
    size = shutil.get_terminal_size()
    max_visible = max(3, min(n, size.lines - 4))

    def _draw() -> None:
        nonlocal filtered, idx
        # If query changed, re-filter
        if query:
            q = query.lower().replace(" ", "")
            filtered = [it for it in items if q in it.lower().replace(" ", "")]
            if not filtered:
                filtered = items
            idx = max(0, min(idx, len(filtered) - 1))

        # Scroll window so idx stays visible
        half = max_visible // 2
        start = max(0, min(idx - half, len(filtered) - max_visible))
        end = start + max_visible

        lines: list[str] = []
        lines.append(f"\r\033[K{bold(cyan(title))}{dim(' · type to filter, ↑↓ to move, ↵ to select')}")
        for i in range(start, end):
            if i >= len(filtered):
                break
            prefix = " " + cyan("▸") if i == idx else "  "
            line = prefix + " " + filtered[i]
            if i == idx:
                line = _c("7", line)  # reverse video for the cursor line
            lines.append(f"\r\033[K{line}")
        if query:
            lines.append(f"\r\033[K{dim('filter: ' + query)}")
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()

    # Save position, draw, then loop
    sys.stdout.write("\0337")   # DECSC — save cursor
    _draw()
    try:
        tty.setraw(fd)
        while True:
            ch = keys.getch(10)
            if ch == "":
                break
            if ch == "\x1b":
                # Could be arrow keys or bare ESC
                nxt = keys.getch(0.05)
                if nxt == "[":
                    code = keys.getch(0.05)
                    if code == "A":      # up
                        idx = max(0, idx - 1)
                    elif code == "B":    # down
                        idx = min(len(filtered) - 1, idx + 1)
                elif nxt == "":
                    return None          # bare ESC = cancel
                else:
                    continue
            elif ch == "\r":
                if filtered and 0 <= idx < len(filtered):
                    return filtered[idx]
            elif ch in ("\x03", "\x04"):
                return None              # Ctrl+C / Ctrl+D = cancel
            elif ch == "\x7f":           # backspace
                query = query[:-1]
            elif ch.isprintable():
                query += ch
            else:
                continue
            # Reset filter on empty query
            if not query:
                filtered = items
            # Move cursor back up and redraw
            visible_lines = min(max_visible, len(filtered)) + (1 if query else 0)
            sys.stdout.write(f"\033[{visible_lines}A")
            _draw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        # Restore cursor and clear picker area
        sys.stdout.write("\0338")        # DECRC — restore cursor
        # Clear the lines we drew (they're below the restored cursor)
        drawn = 1 + min(max_visible, len(filtered)) + (1 if query else 0)
        sys.stdout.write("\n" * drawn + f"\033[{drawn}A")
        sys.stdout.flush()


class LineEditor:
    """The line-editing state machine (buffer + cursor), rendering-agnostic.

    handle() applies one keypress and returns an action string; any escape-
    continuation bytes are pulled via getch, which returns '' when none arrive
    soon — so a bare ESC never wedges the caller. The same editor drives the
    interactive prompt and the during-turn type-ahead.

    Keys: left/right/home/end, ctrl/alt+arrows word-jump, backspace/delete,
    ctrl+W & ctrl/shift+backspace del-word, ctrl+U/K kill to start/end,
    ctrl+A/E home/end, ctrl+L clear screen, shift+tab cycle approval mode,
    shift/alt+enter insert newline (plain enter submits), up/down recall
    queued prompts.
    """

    def __init__(self, text: str = "") -> None:
        self.buf: list[str] = list(text)
        self.cursor = len(self.buf)
        self._in_paste = False
        self._paste_buf: list[str] = []

    def text(self) -> str:
        return "".join(self.buf)

    def set_text(self, text: str) -> None:
        self.buf = list(text)
        self.cursor = len(self.buf)

    @staticmethod
    def _is_word(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    def _word_left(self) -> None:
        while self.cursor > 0 and self.buf[self.cursor - 1] == " ":
            self.cursor -= 1
        while self.cursor > 0 and self._is_word(self.buf[self.cursor - 1]):
            self.cursor -= 1

    def _word_right(self) -> None:
        while self.cursor < len(self.buf) and self.buf[self.cursor] == " ":
            self.cursor += 1
        while self.cursor < len(self.buf) and self._is_word(self.buf[self.cursor]):
            self.cursor += 1

    def _insert(self, s: str) -> None:
        for ch in s:
            self.buf.insert(self.cursor, ch)
            self.cursor += 1

    def _backspace(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1
            del self.buf[self.cursor]

    def _delete_word_left(self) -> None:
        end = self.cursor
        self._word_left()
        del self.buf[self.cursor:end]

    def handle(self, ch: str, getch: Callable[[], str]) -> str:
        """Apply one keypress. Returns: submit / interrupt / eof /
        clear-screen / shift-tab / tab / up / down / edit.

        Plain Enter (\\r) submits; Shift/Alt+Enter and pasted newlines insert a
        literal newline (multi-line prompts). Backspace deletes a char; Ctrl+W
        and Ctrl/Shift+Backspace (where the terminal encodes the modifier, plus
        Ctrl-H) delete the previous word."""
        if self._in_paste:                            # paste — batch everything except ESC
            if ch == "\x1b":                          # let \e[201~ through to _escape()
                return self._escape(getch)
            self._paste_buf.append(ch)                # raw — line-endings normalised at flush
            return "edit"
        if ch == "\r":                                # Enter — submit
            return "submit"
        if ch == "\n":                                # newline (paste / Ctrl+J) — insert
            self._insert("\n")
            return "edit"
        if ch == "\x03":
            return "interrupt"
        if ch == "\x04":
            return "eof"
        if ch == "\x09":                              # Tab
            return "tab"
        if ch == "\x0c":
            return "clear-screen"
        if ch == "\x01":                              # Ctrl+A — home
            self.cursor = 0
        elif ch == "\x05":                            # Ctrl+E — end
            self.cursor = len(self.buf)
        elif ch == "\x7f":                            # Backspace — delete char
            self._backspace()
        elif ch == "\x08":                            # Ctrl-H / Ctrl+Backspace — delete word
            self._delete_word_left()
        elif ch == "\x0b":                            # Ctrl+K — kill to end
            del self.buf[self.cursor:]
        elif ch == "\x15":                            # Ctrl+U — kill to start
            del self.buf[:self.cursor]
            self.cursor = 0
        elif ch == "\x17":                            # Ctrl+W — delete word back
            self._delete_word_left()
        elif ch == "\x1b":
            return self._escape(getch)
        elif ch.isprintable() and not self._in_paste:
            self._insert(ch)
        return "edit"

    def _escape(self, getch: Callable[[], str]) -> str:
        nxt = getch()
        if nxt == "":                                 # bare ESC — ignore
            return "edit"
        if nxt in ("\r", "\n"):                       # Alt/Shift+Enter (ESC+CR) — newline
            self._insert("\n")
        elif nxt in ("\x7f", "\x08"):                 # Alt+Backspace — delete word
            self._delete_word_left()
        elif nxt == "[":
            return self._csi(getch)
        elif nxt == "O":                              # SS3: ^[OH Home, ^[OF End
            code = getch()
            if code == "H":
                self.cursor = 0
            elif code == "F":
                self.cursor = len(self.buf)
        elif nxt == "b":                              # Alt+b / Alt+left — word left
            self._word_left()
        elif nxt == "f":                              # Alt+f / Alt+right — word right
            self._word_right()
        # other Alt+key sequences: ignore
        return "edit"

    def _csi(self, getch: Callable[[], str]) -> str:
        """Parse a CSI sequence (the bytes after ESC '['): collect the numeric
        parameters, read the final byte, then dispatch. Fully consuming the
        sequence keeps unknown ones from leaking as typed characters, and the
        parameters carry the modifier — so Shift+Enter (newline) and
        Shift/Ctrl+Backspace (delete word) work in terminals that encode them
        via modifyOtherKeys (CSI 27;m;k ~) or CSI-u (CSI cp;m u)."""
        params, final = "", ""
        while True:
            c = getch()
            if c == "":                               # truncated — give up cleanly
                return "edit"
            if c.isdigit() or c in ";:":
                params += c
                continue
            final = c
            break
        nums = [int(p) for p in params.split(";") if p.isdigit()]
        mods = nums[1] if len(nums) > 1 else 1        # 1 = no modifier

        if final == "Z":                              # Shift+Tab
            return "shift-tab"
        if final in ("A", "B"):                       # Up / Down
            return "up" if final == "A" else "down"
        if final == "u":                              # CSI-u: codepoint ; mods u
            key = nums[0] if nums else 0
            if key == 13:                             # Enter
                if mods > 1:
                    self._insert("\n")
                    return "edit"
                return "submit"
            if key in (127, 8):                       # Backspace
                self._delete_word_left() if mods > 1 else self._backspace()
            return "edit"
        if final == "~":                              # CSI n [; mods] ~
            n = nums[0] if nums else 0
            if n == 200:                              # bracketed-paste begin
                self._paste_buf.clear()
                self._in_paste = True
                return "edit"
            if n == 201:                              # bracketed-paste end
                self._in_paste = False
                if self._paste_buf:
                    # normalise \r\n and bare \r → \n, then splice once
                    text = "".join(self._paste_buf)
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    tail = list(text)
                    self.buf[self.cursor:self.cursor] = tail
                    self.cursor += len(tail)
                    self._paste_buf = []
                return "edit"
            if n in (1, 7):
                self.cursor = 0                       # Home
            elif n in (4, 8):
                self.cursor = len(self.buf)           # End
            elif n == 3 and self.cursor < len(self.buf):
                del self.buf[self.cursor]             # Delete
            elif n == 27 and len(nums) >= 3:          # modifyOtherKeys: 27 ; mods ; key ~
                key = nums[2]
                if key == 13:                         # Enter
                    if mods > 1:
                        self._insert("\n")
                    else:
                        return "submit"
                elif key in (127, 8):                 # Backspace
                    self._delete_word_left()
            return "edit"
        # cursor / nav letters, optionally modified (CSI [1 ; mods] <letter>)
        word = mods in (3, 4, 5, 6, 7, 8)             # Alt/Ctrl combos → move by word
        if final == "C":                              # right
            self._word_right() if word else self._move(+1)
        elif final == "D":                            # left
            self._word_left() if word else self._move(-1)
        elif final == "H":
            self.cursor = 0
        elif final == "F":
            self.cursor = len(self.buf)
        return "edit"

    def _move(self, delta: int) -> None:
        self.cursor = max(0, min(len(self.buf), self.cursor + delta))


def read_line(prompt_fn, state: State, initial: str = "") -> str:
    """Read one line with readline-like editing (see LineEditor for keys).

    On a capable terminal the prompt lives in the bottom dock — session status
    row, dim divider, then the input field — pinned to the screen's last rows
    while history scrolls above. On submit the line is echoed (bolded) into
    the flow, so user turns stand out in the scrollback.

    `initial` pre-fills the buffer — text typed ahead during the previous turn
    lands here, so nothing the user typed is ever thrown away.
    """
    if not _raw_capable():
        return input(prompt_fn())

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    keys = _keys()
    ed = LineEditor(initial)

    def flow_row() -> int:
        """1-based row of the flow's last (ready) line — just above the dock.
        The dock grows/shrinks, so this is recomputed rather than stashed via
        a stale cursor save."""
        return _Dock.rows - _Dock.height

    def redraw() -> None:
        parts: list[str] = []
        head = state.last_head
        foot = statusline(state) + dim(" · ") + mode_badge(state.approval_mode).strip()
        rows = _paint_input_box(parts, head, ed.text(), ed.cursor, block_cursor=False,
                                foot=foot)
        if rows == 0:                     # terminal too short — inline prompt
            ghost = _command_ghost(ed.text())
            flat = ed.text().replace("\n", " ")   # no room to expand; keep it linear
            parts.append("\r\033[K" + prompt_fn() + flat + dim(ghost))
            after = (len(ed.buf) - ed.cursor) + len(ghost)
            if after > 0:
                parts.append(f"\033[{after}D")
        sys.stdout.write("".join(parts))
        sys.stdout.flush()

    def to_flow() -> None:
        """Put the cursor back where the scrolling flow left off."""
        if _Dock.active:
            sys.stdout.write(f"\033[{flow_row()};1H")
            sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?2004h")   # enable bracketed paste
        sys.stdout.flush()
        redraw()                          # establishes/normalizes the dock
        while True:
            if not keys.ready(1.0):       # 1s tick — keeps the cache-age alive
                redraw()
                continue
            ch = keys.getch()
            if ch == "":                             # stdin closed — never recovers
                to_flow()
                raise EOFError
            act = ed.handle(ch, lambda: keys.getch(0.05))
            if act == "submit":
                text = ed.text()
                # clear the field's input rows, then echo the line into the flow.
                # image paths show as compact markers, but the raw text is
                # what we return (and what the turn turns into image attachments).
                shown = _input_display(text)
                if _Dock.active:
                    flow = flow_row()
                    # input rows sit two below the flow's last row (top border
                    # is at flow+1); blank them so the draft doesn't flash on
                    # while the spinner repaints a collapsed dock for the turn.
                    clear = "".join(f"\033[{flow + 2 + i};1H\033[K" + _box_mid("")
                                    for i in range(_Dock.height - 2))
                    sys.stdout.write(clear + f"\033[{flow};1H"
                                     + "\r\n" + prompt_fn() + shown + "\r\n")
                else:
                    sys.stdout.write("\r\033[K" + prompt_fn() + shown + "\r\n")
                sys.stdout.flush()
                return text
            if act == "interrupt":
                to_flow()
                raise KeyboardInterrupt
            if act == "eof":
                if ed.buf:
                    continue
                to_flow()
                raise EOFError
            if act == "shift-tab":
                _cycle_mode(state)
            elif act == "tab":
                matches = _complete_command(ed)
                if matches:                          # ambiguous — list candidates
                    listing = dim("  " + "  ".join(matches))
                    if _Dock.active:
                        # print into the flow above the box (the next redraw repaints)
                        sys.stdout.write(f"\033[{flow_row()};1H" + listing + "\r\n")
                    else:
                        sys.stdout.write("\r\n" + listing + "\r\n")
            elif act == "clear-screen":
                sys.stdout.write("\033[r\033[2J\033[H")   # reset region, wipe, home
                _Dock.active = False                       # rebuild the dock fresh
            # coalesce: while a paste still has bytes buffered, defer the repaint
            # so the field redraws (and the dock reflows) once, not per character
            if not keys.ready(0):
                redraw()
    finally:
        sys.stdout.write("\033[?2004l")   # disable bracketed paste
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


class TypeAhead:
    """Keyboard capture while a turn is running: type-ahead + a prompt queue.

    While the model works, a reader thread keeps the terminal in cbreak mode
    and feeds keystrokes into a LineEditor; the draft is echoed in the docked
    input row at the bottom of the spinner's live region (status above, dim
    divider between). Enter queues the line — queued prompts run in order as
    soon as the current turn finishes. Unsubmitted text survives the turn and
    pre-fills the next interactive prompt.

    Up/Down recall already-queued prompts back into the editor for editing or
    cancellation (Enter saves, an empty Enter drops that one). While such an
    edit is in progress the queue is "paused": wait_until_not_editing() blocks
    the turn from consuming the next prompt until the user finishes, so a
    half-edited message never starts running underneath them.

    A confirm prompt can borrow the keyboard via request_line(); the
    half-typed draft is stashed and restored around it.

    cbreak (not raw) keeps ISIG on, so Ctrl-C still raises KeyboardInterrupt
    in the main thread — that's what lets it interrupt the turn.
    """

    def __init__(self, state: State) -> None:
        self.state = state
        self.queue: deque[str] = deque()
        self._editor = LineEditor()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._saved: Optional[list] = None
        self._spinner: Optional[Spinner] = None
        # request_line plumbing (confirm prompts borrow the keyboard)
        self._req_prompt: Optional[str] = None
        self._req_text = ""
        self._req_done = threading.Event()
        # queue-editing state: which queued slot is being edited (None = the
        # live draft), the live draft stashed while editing, and a gate that's
        # SET whenever no edit is in progress (so consumption can proceed).
        self._edit_idx: Optional[int] = None
        self._live_stash = ""
        self._edit_hinted = False
        self._not_editing = threading.Event()
        self._not_editing.set()

    @property
    def active(self) -> bool:
        return self._thread is not None

    @property
    def editing(self) -> bool:
        """True while the user is editing an already-queued prompt."""
        return self._edit_idx is not None

    def wait_until_not_editing(self) -> None:
        """Block until no queued-message edit is in progress. Used to defer
        queue consumption — a turn must not start running the next queued
        prompt while the user is still editing one."""
        self._not_editing.wait()

    def cancel_edit(self) -> None:
        """Abandon any in-progress queue edit, restoring the live draft."""
        with self._lock:
            if self._edit_idx is not None:
                self._editor.set_text(self._live_stash)
                self._live_stash = ""
                self._edit_idx = None
                self._edit_hinted = False
        self._not_editing.set()

    def start(self, spinner: Spinner) -> None:
        if not _raw_capable() or self.active:
            return
        self._spinner = spinner
        spinner.extra = self._badge          # queued count, on the status row
        spinner.lead = lambda: mode_badge(self.state.approval_mode).strip()
        spinner.input_state = self._input_state   # (text, cursor, confirm prompt)
        fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.active:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
        if self._spinner is not None:
            self._spinner.extra = None
            self._spinner.lead = None
            self._spinner.input_state = None
            self._spinner = None

    def take_text(self) -> str:
        """Surrender the unsubmitted draft (pre-fills the next prompt)."""
        with self._lock:
            text = self._editor.text()
            self._editor.set_text("")
        return text

    def drop_all(self) -> list[str]:
        """Empty the queue (after Ctrl-C) and return what was dropped."""
        with self._lock:
            dropped = list(self.queue)
            self.queue.clear()
        return dropped

    # --- confirm prompts borrow the keyboard ----------------------------
    def request_line(self, prompt: str) -> str:
        """Synchronously read one line through the reader thread."""
        if not self.active:
            try:
                return input(prompt)
            except (EOFError, KeyboardInterrupt):
                return ""
        with self._lock:
            stash = self._editor.text()
            self._editor.set_text("")
            self._req_text = ""
            self._req_done.clear()
            self._req_prompt = prompt
        if self._spinner:
            self._spinner.render_now()    # question appears in the input row
        try:
            self._req_done.wait()
            with self._lock:
                return self._req_text
        except KeyboardInterrupt:
            return ""                                 # ^C at a confirm = decline
        finally:
            with self._lock:
                self._req_prompt = None
                self._editor.set_text(stash)

    # --- what the docked rows show ---------------------------------------
    def _badge(self) -> str:
        """Queued-prompt count (or the edit position), on the status row."""
        with self._lock:
            queued = len(self.queue)
            editing = self._edit_idx
        if editing is not None:
            return "  " + yellow(f"[editing {editing + 1}/{queued}]")
        return ("  " + yellow(f"[{queued} queued]")) if queued else ""

    def _input_state(self) -> "tuple[str, int, Optional[str]]":
        """The field's (text, cursor, confirm_prompt) for the spinner to paint.
        While a confirm question borrows the field, `confirm_prompt` is the
        question and `text` is the typed answer."""
        with self._lock:
            return self._editor.text(), self._editor.cursor, self._req_prompt

    # --- reader thread ----------------------------------------------------
    def _run(self) -> None:
        keys = _keys()
        try:
            while not self._stop.is_set():
                if not keys.ready(0.05):
                    continue
                ch = keys.getch()
                if ch == "":                          # stdin closed
                    return
                self._on_key(ch, lambda: keys.getch(0.05))
        finally:
            self._not_editing.set()                   # never strand a waiter

    def _on_key(self, ch: str, getch: Callable[[], str]) -> None:
        with self._lock:
            in_request = self._req_prompt is not None
            act = self._editor.handle(ch, getch)
            submit_text = self._editor.text().strip() if act == "submit" else ""

        if in_request:
            if act == "submit":
                self._req_text = submit_text
                self._req_done.set()
            elif act in ("interrupt", "eof"):
                self._req_done.set()                  # empty answer = decline
            elif act == "edit":
                # Single-char confirm (y/d/a/n) auto-submits without Enter
                # and clears the input field so the key feels instant.
                with self._lock:
                    txt = self._editor.text().strip()
                    if len(txt) == 1 and txt in "ydan":
                        self._editor.set_text("")
                        self._req_text = txt
                        self._req_done.set()
            if self._spinner:
                self._spinner.render_now()
            return

        # `note` is any one-time line to print into the flow; computed while
        # holding the lock but printed after releasing it (printing re-enters
        # the spinner's io lock, so we must not hold ours — see the deadlock
        # fixed in 5be1f5f).
        note: Optional[str] = None
        if act == "submit":
            note = self._commit_submit(submit_text)
        elif act == "up":
            note = self._nav_queue(-1)
        elif act == "down":
            note = self._nav_queue(+1)
        elif act == "shift-tab":
            _cycle_mode(self.state)
            note = dim("  approval mode → ") + mode_badge(self.state.approval_mode).strip()
        elif act == "tab":
            with self._lock:
                matches = _complete_command(self._editor)
            if matches:                               # ambiguous — list candidates
                note = dim("  " + "  ".join(matches))
        # eof / interrupt / clear-screen make no sense mid-turn — ignored

        if note and self._spinner:
            self._spinner.println(note)
        if self._spinner:
            self._spinner.render_now()

    # --- queue editing (driven from _on_key, runs in the reader thread) ---
    def _commit_submit(self, text: str) -> Optional[str]:
        """Enter pressed. While editing a queued slot this saves the edit back
        (empty text drops that slot); otherwise it queues `text` as a new
        prompt."""
        with self._lock:
            if self._edit_idx is not None:
                return self._finish_edit_locked(text)
            if text:
                self.queue.append(text)
            self._editor.set_text("")
            if not text:
                return None
            return "\n".join(dim("  + queued: " + ln) for ln in text.split("\n"))

    def _nav_queue(self, direction: int) -> Optional[str]:
        """Up (-1) / Down (+1) through queued prompts for editing. Up from the
        live draft enters edit mode at the newest queued prompt; Down past the
        newest leaves edit mode and restores the live draft. Edits to the
        current slot are persisted as you navigate."""
        with self._lock:
            if self._edit_idx is None:
                if direction > 0 or not self.queue:
                    return None                       # nothing to recall
                self._live_stash = self._editor.text()
                self._edit_idx = len(self.queue) - 1
                self._editor.set_text(self.queue[self._edit_idx])
                self._not_editing.clear()
                first, self._edit_hinted = not self._edit_hinted, True
                return (dim("  editing queued — ↑/↓ navigate · Enter saves · "
                            "empty Enter cancels") if first else None)
            # already editing: stash the current text into its slot, then move
            self.queue[self._edit_idx] = self._editor.text().strip()
            new_idx = self._edit_idx + direction
            if new_idx >= len(self.queue):            # past the newest → live draft
                self._exit_edit_locked()
                return None
            self._edit_idx = max(0, new_idx)
            self._editor.set_text(self.queue[self._edit_idx])
            return None

    def _finish_edit_locked(self, text: str) -> Optional[str]:
        """Save (or, when empty, drop) the slot being edited, then leave edit
        mode. Caller holds the lock."""
        idx = self._edit_idx
        note = None
        if idx is not None and 0 <= idx < len(self.queue):
            if text:
                self.queue[idx] = text
                note = dim("  ~ updated queued prompt")
            else:
                del self.queue[idx]
                note = dim("  ✗ cancelled a queued prompt")
        self._exit_edit_locked()
        return note

    def _exit_edit_locked(self) -> None:
        """Leave edit mode: prune any emptied slots, restore the live draft,
        and release the consumption gate. Caller holds the lock."""
        self.queue = deque(m for m in self.queue if m.strip())
        self._editor.set_text(self._live_stash)
        self._live_stash = ""
        self._edit_idx = None
        self._edit_hinted = False
        self._not_editing.set()


# --- replay previous sessions on resume --------------------------------
_ANSI = re.compile(r"\033\[[0-9;]*m")


def _replay_conversation(state: State) -> None:
    """Re-print the saved transcript verbatim — a resumed session looks
    exactly like it did when the user left it, meta info and all."""
    session = state.session
    if session.transcript:
        for entry in session.transcript:
            text = entry.get("text", "")
            print(text if _TTY else _ANSI.sub("", text))
        return

    # Legacy sessions (saved before transcripts existed): best-effort replay
    # from the message log — per-turn stats are gone, but speakers are clear.
    for msg in session.messages:
        role, content = msg.get("role", ""), msg.get("content", "")
        if role == "user" and content and isinstance(content, str):
            print("\n" + bold(magenta("▸ ")) + _input_display(content))
        elif role == "assistant" and content:
            print("\n" + render_assistant(content))


# --- main loop ---------------------------------------------------------
def banner(state: State) -> None:
    print(bold(cyan(f"hrns {__version__}")) + dim(" — coding harness"))
    conn = green("connected") if state.client else red("not connected (run /connect)")
    print(dim(f"  model {state.session.model} · session {state.session.id} · {conn}"))
    if state.session.turn_count() > 0:
        print(dim(f"  resumed — {state.session.turn_count()} previous turn(s)"))
    print(dim(f"  workspace {workspace_root()}"))
    cyc = "shift+tab" if _raw_capable() else "/mode"
    print(dim(f"  approval {state.approval_mode} ({cyc} to cycle) · /help · Ctrl-D to exit"))
    if _raw_capable():
        print(dim("  type while it works — Enter queues your next prompt · ↑ edits a queued one · Ctrl-C interrupts"))
        print(dim("  Shift+Enter for a newline · drag an image in to attach it"))


def main() -> None:
    set_workspace_root(Path.cwd())  # the folder hrns was opened in is the sandbox root
    cfg = Config.load()
    tools.MEMORY_PATH = cfg.memory_path  # so save_memory knows where to write

    # Auto-resume: load the most recent session so the terminal looks continuous.
    sessions = list_sessions(cfg.sessions_dir)
    if sessions:
        session = sessions[0]  # newest by updated_at
        cfg.model = session.model  # keep status line consistent
    else:
        session = Session.new(cfg.model, build_system_prompt(cfg))

    client = None
    if cfg.api_key:
        try:
            client = DeepSeekClient(cfg.api_key, cfg.base_url, cfg.provider)
        except DeepSeekError:
            client = None
    state = State(cfg=cfg, session=session, client=client, approval_mode=cfg.approval_mode)

    # fetch starting balance
    if state.client:
        cfg.balance = state.client.get_balance()

    banner(state)

    if sessions:
        _replay_conversation(state)
    typeahead = TypeAhead(state)
    prompt_fn = lambda: bold(magenta("▸ "))  # noqa: E731
    docked = _raw_capable()
    try:
        _main_loop(state, cfg, typeahead, prompt_fn, docked)
    finally:
        _undock()
    state.session.save(cfg.sessions_dir)
    print(dim(f"Saved session {state.session.id}. Bye."))


def _main_loop(state: State, cfg: Config, typeahead: TypeAhead,
               prompt_fn, docked: bool) -> None:
    while True:
        if not docked:
            # no dock (pipe / dumb terminal): print the vitals inline instead
            print("\n" + statusline(state))
            print(_divider())
        from_queue = bool(typeahead.queue)
        if from_queue:
            # a prompt queued while the previous turn ran — echo it like input
            line = typeahead.queue.popleft().strip()
            print(("\n" if docked else "") + prompt_fn() + _input_display(line) + dim("  (queued)"))
        else:
            try:
                line = read_line(prompt_fn, state, initial=typeahead.take_text()).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
        if not line:
            continue

        if line.startswith("/"):
            cmd = line.split(maxsplit=1)[0]
            rest = line[len(cmd):].strip()
            if cmd in ("/quit", "/exit", "/q"):
                return
            handler = COMMANDS.get(cmd)
            if handler:
                try:
                    handler(state, rest)
                except KeyboardInterrupt:
                    print(yellow("\ninterrupted"))
            else:
                print(red(f"Unknown command {cmd}. Try /help."))
            continue

        if state.client is None:
            print(yellow("Not connected. Run /connect to set your API key."))
            continue

        echo = prompt_fn() + _input_display(line) + (dim("  (queued)") if from_queue else "")
        state.session.log("user", "\n" + echo)
        try:
            run_turn(state, line, typeahead)
        except KeyboardInterrupt:
            # ^C that landed outside run_turn's own guard (e.g. while saving)
            print(yellow("\ninterrupted"))
            state.session.save(cfg.sessions_dir)


if __name__ == "__main__":
    main()
