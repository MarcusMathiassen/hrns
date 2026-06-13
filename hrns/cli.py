"""hrns REPL: chat loop, slash commands, and the cache-aware status line.

Input is never blocked: while a turn is running, a TypeAhead reader keeps
capturing keystrokes — Enter queues the line as the next prompt, and anything
left half-typed pre-fills the prompt once the turn finishes.
"""

from __future__ import annotations

import codecs
import difflib
import getpass
import itertools
import json
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:  # raw key reading for Shift+Tab; absent on non-Unix terminals
    import termios
    import tty
except ImportError:  # pragma: no cover
    termios = None  # type: ignore
    tty = None  # type: ignore

from hrns import __version__, memory
from hrns.client import ChatResult, DeepSeekClient, DeepSeekError
from hrns.config import Config, context_window, pricing_for
from hrns.session import Session, list_sessions
from hrns.tools import TOOL_SCHEMAS, execute, set_workspace_root, workspace_root

# --- ANSI styling (degrades to no-op if not a tty) --------------------
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def dim(t: str) -> str: return _c("2", t)
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


def _divider() -> str:
    """A dim rule spanning the full terminal width (re-measured per call,
    so it tracks live resizes)."""
    return dim("─" * shutil.get_terminal_size().columns)


# --- the input box (drawn inside the bottom dock) -----------------------
#   ┌─ auto · deepseek-chat · 95.0% · 3 turns · 12.3k · $0.01 · $10.00
#   │ the user types in here
#   └─
def _box_top(content: str) -> str:
    return dim("┌─ ") + content


def _box_mid(content: str) -> str:
    return dim("│ ") + content


def _box_bottom() -> str:
    return dim("└─")


# --- the bottom dock ----------------------------------------------------
class _Dock:
    """The 3-row UI pinned to the bottom of the terminal.

    A DECSTBM scroll region confines normal output to the rows above, so
    replies and meta scroll there while the status row, divider, and input
    field stay put on the screen's last three rows — even mid-reasoning.
    """
    active = False
    rows = 0


def _dock_ensure(parts: list[str]) -> int:
    """Append the escapes that (re)establish the dock. Returns the terminal
    row count, or 0 when the terminal is too short to dock."""
    size = shutil.get_terminal_size()
    if size.lines < 8:
        return 0
    if _Dock.active and _Dock.rows == size.lines:
        return size.lines
    if _Dock.active:                      # resized — release the old margins
        parts.append("\0337\033[r\0338")
    parts.append("\n\n\n\033[3A")         # free the bottom rows (scroll if needed)
    parts.append("\0337" + f"\033[1;{size.lines - 3}r" + "\0338")
    _Dock.active = True
    _Dock.rows = size.lines
    return size.lines


def _undock() -> None:
    """Clear the docked rows and release the scroll margins (on exit)."""
    if not _Dock.active:
        return
    sys.stdout.write(f"\0337\033[{_Dock.rows - 2};1H\033[J\033[r\0338\033[?25h")
    sys.stdout.flush()
    _Dock.active = False


# --- "working…" spinner with an elapsed timer -------------------------
class Spinner:
    """A self-erasing live region animated from a background thread.

    The main thread does the blocking work and calls .set(label) to describe
    what's happening ("thinking", "reading config.py"). The whole noisy
    intermediate stream (reasoning, tool I/O) is collapsed into the status row.

    When `input_line` is set (type-ahead active), it paints the input box
    into the bottom dock:

        ┌─ auto · ⠋ reasoning 3s [1 queued]   <- mode (via `lead`) + status
        │ the user's draft                     <- input row, block cursor
        └─

    Without `input_line` it degrades to the classic single status row.
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
        # when set, renders the input box's middle row
        self.input_line: Optional[Callable[[], str]] = None

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
        input_fn = self.input_line
        elapsed = time.monotonic() - self._t0
        spinner_part = f"{cyan(self._frame)} {label} {dim(f'{elapsed:.0f}s')}"
        with self._io:
            if self._paused.is_set() or self._stop.is_set():
                return
            if input_fn is None:
                sys.stdout.write(f"\r\033[K{spinner_part}")
                sys.stdout.flush()
                return
            parts: list[str] = []
            rows = _dock_ensure(parts)
            if rows == 0:                 # terminal too short to dock
                sys.stdout.write(f"\r\033[K{spinner_part}  {input_fn()}")
                sys.stdout.flush()
                return
            top = rows - 2
            lead_fn = self.lead
            head = (lead_fn() + dim(" · ") if lead_fn else "") + spinner_part + dim(" · ") + statusline(self.state) + extra
            # hw cursor stays (hidden) in the flow so prints land there;
            # the input row paints its own block cursor. Hide BEFORE saving:
            # some terminals (Terminal.app, iTerm2) include visibility in the
            # DECSC state, so save-then-hide gets undone by every restore —
            # leaving a second, visible cursor blinking in the flow.
            parts.append("\033[?25l\0337")
            parts.append(f"\033[{top};1H\033[K" + _box_top(head))
            parts.append(f"\033[{top + 1};1H\033[K" + _box_mid(input_fn()))
            parts.append(f"\033[{top + 2};1H\033[K" + _box_bottom())
            parts.append("\0338")
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
    if mode == "auto-edit":
        return name in ("edit_file", "create_file")
    if mode == "auto":
        return True
    return False




def session_summary(s: Session) -> str:
    rate = s.cache_hit_rate * 100
    meta = (f"{s.turn_count()} turns · {s.usage['requests']} requests · "
            f"cache hit rate {rate:.0f}% · updated {s.updated_at}")
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
    """Render a reply with a speaker marker so it's clear who said what:
    the first line gets a cyan bullet, the rest is indented under it."""
    body = format_markdown(md) if md else dim("(no text response)")
    lines = body.split("\n")
    return "\n".join([cyan("∙ ") + lines[0]] + ["  " + ln for ln in lines[1:]])


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


def run_turn(state: State, user_input: str, typeahead: TypeAhead) -> None:
    state.session.append({"role": "user", "content": user_input})
    cfg, session, client = state.cfg, state.session, state.client
    assert client is not None

    spinner = Spinner(state)
    typeahead.start(spinner)  # attach dock hooks first so no frame paints inline
    spinner.start("calling api")
    final_text = ""
    error: Optional[str] = None
    interrupted = False

    try:
        while True:
            buf: list[str] = []
            out_tok = 0

            def on_reasoning(_t: str) -> None:
                spinner.set("reasoning")

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
                    on_text=on_text,
                    on_reasoning=on_reasoning,
                )
            except DeepSeekError as e:
                error = str(e)
                break

            session.append(result.message)
            session.record_usage(result.usage)

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
                session.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            # loop so the model can read the tool results
    except KeyboardInterrupt:
        interrupted = True
    finally:
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


def _confirm_preview(name: str, args: dict, reason: Optional[str] = None) -> str:
    """Describe a gated action before asking to apply it."""
    if reason == "outside-workspace":
        path = args.get("path", ".")
        verb = {
            "read_file": "read", "list_dir": "list", "glob": "search",
            "grep": "search", "edit_file": "edit", "create_file": "create",
        }.get(name, "access")
        return (red(f"{name} wants to {verb} a path OUTSIDE the workspace:") + "\n"
                f"    {bold(str(path))}\n"
                + dim(f"    workspace: {workspace_root()}"))
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
    if name in ("read_file",):
        return dim(f"  {blue('read')} {bold(str(args.get('path', '?')))}")
    return dim(f"  {blue(name)} {dim(str(args))}")


def _make_confirm(spinner: Spinner, state: State, typeahead: "TypeAhead"):
    """Confirm gate; pauses the spinner to show a preview before applying.

    In an auto mode the action is shown (diff/preview) and auto-approved — except
    out-of-workspace access, which always asks. The answer is read through the
    type-ahead reader (it owns the keyboard while a turn runs); any half-typed
    draft is stashed and restored around the question.
    """
    def confirm(name: str, args: dict, reason: Optional[str] = None) -> bool:
        session = state.session
        docked = spinner.input_line is not None
        sp = spinner if docked else None
        if not docked:
            spinner.pause()
        if name in ("edit_file", "create_file"):
            _say(session, "meta", _confirm_preview(name, args, reason), sp)
        if _auto_approves(state.approval_mode, name, reason):
            if not docked:
                spinner.resume()
            return True
        prompt = yellow("  allow access outside the workspace? [y/N] "
                        if reason == "outside-workspace" else "  apply? [y/N] ")
        ans = typeahead.request_line(prompt).strip().lower()
        if docked:
            spinner.println(prompt + ans)  # echo into the flow, like replay
        session.log("meta", prompt + ans)
        if not docked:
            spinner.resume()
        return ans in ("y", "yes")
    return confirm


# --- slash commands ----------------------------------------------------
def cmd_help(state: State, args: str) -> None:
    print(bold("Commands:"))
    for name, desc in [
        ("/sessions", "list saved sessions; /sessions <id|#> to resume one"),
        ("/clear", "archive the current session and start a fresh one"),
        ("/connect", "configure & test the DeepSeek connection (API key, model)"),
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
    print(bold("Connect to DeepSeek"))
    print(dim(f"  base_url [{cfg.base_url}]:"), end=" ")
    base = input().strip() or cfg.base_url
    print(dim(f"  model [{cfg.model}]:"), end=" ")
    model = input().strip() or cfg.model
    have = "set" if cfg.api_key else "unset"
    key = getpass.getpass(f"  api key (currently {have}, blank = keep): ").strip()

    cfg.base_url, cfg.model = base, model
    if key:
        cfg.api_key = key
    if not cfg.api_key:
        print(red("No API key available — cannot connect."))
        return

    try:
        client = DeepSeekClient(cfg.api_key, cfg.base_url)
        models = client.list_models()
    except DeepSeekError as e:
        print(red(f"Connection failed: {e}"))
        return

    state.client = client
    cfg.save(include_key=True)
    ok = green("✓ connected")
    here = green("available") if model in models else yellow("not in /models list")
    print(f"{ok} · {len(models)} models · '{model}' {here}")
    print(dim(f"  saved to {cfg.config_path} — hrns will reconnect automatically next run"))


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
    print(green(f"Default model set to {state.cfg.model}. Use /clear to start a session on it."))


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
    "/clear": cmd_clear,
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


def _location_info() -> str:
    """Short cwd + git branch and dirty status, or empty string.
    Cached for 5s to avoid spawning git on every keystroke.
    """
    _cache = getattr(_location_info, "_cache", None)
    _ts = getattr(_location_info, "_ts", 0.0)
    now = time.monotonic()
    if _cache is not None and now - _ts < 5.0:
        return _cache
    try:
        cwd = Path.cwd()
        home = Path.home()
        try:
            cwd_short = "~" / cwd.relative_to(home)
        except ValueError:
            cwd_short = cwd

        # git info — fast; fails silently
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=1,
        ).stdout.strip()
        if not branch:
            result = dim(str(cwd_short))
            _location_info._cache = result
            _location_info._ts = now
            return result

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=1,
        ).stdout
        dirty = ""
        if status:
            staged = 0
            unstaged = 0
            untracked = 0
            for l in status.splitlines():
                if len(l) < 2:
                    continue
                if l.startswith("??"):
                    untracked += 1
                elif l[0] != " ":
                    staged += 1
                elif l[1] != " ":
                    unstaged += 1
            parts = []
            if staged:
                parts.append(f"+{staged}")
            if unstaged:
                parts.append(f"-{unstaged}")
            if untracked:
                parts.append(f"?{untracked}")
            dirty = " " + "/".join(parts) if parts else " *"

        result = dim(f"{cwd_short} · {branch}{dirty}")
        _location_info._cache = result
        _location_info._ts = now
        return result
    except Exception:
        result = ""
        _location_info._cache = result
        _location_info._ts = now
        return result


def statusline(state: State) -> str:
    """One colored line of session vitals, rendered above each input prompt.

    Each metric gets its own color so it's scannable at a glance:
      cyan=model  green=cache hit rate  blue=turns  magenta=context
      dim=cost  dim=balance
    """
    s = state.session
    u = s.usage
    hit, miss = u["prompt_cache_hit_tokens"], u["prompt_cache_miss_tokens"]
    cache_rate = (hit / (hit + miss) * 100) if (hit + miss) else None
    cost = s.cost(pricing_for(s.model))
    bal = state.cfg.balance
    cum_tok = u["prompt_tokens"] + u["completion_tokens"]

    segs = [
        _location_info(),
        cyan(_model_name(s.model)),
        green(f"{cache_rate:.1f}%" if cache_rate is not None else "--%"),
        blue(f"{s.turn_count()} turn{'' if s.turn_count() == 1 else 's'}"),
        magenta(f"{_human(s.context_tokens)} ctx / {_human(cum_tok)} cum"),
        yellow(_money(cost)),
        yellow(f"${bal:.2f}" if bal is not None else "--"),
    ]
    return dim(" · ").join(s for s in segs if s)


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


class LineEditor:
    """The line-editing state machine (buffer + cursor), rendering-agnostic.

    handle() applies one keypress and returns an action string; any escape-
    continuation bytes are pulled via getch, which returns '' when none arrive
    soon — so a bare ESC never wedges the caller. The same editor drives the
    interactive prompt and the during-turn type-ahead.

    Keys: left/right/home/end, ctrl/alt+arrows word-jump, backspace/delete,
    ctrl+W del-word, ctrl+U/K kill to start/end, ctrl+A/E home/end,
    ctrl+L clear screen, shift+tab cycle approval mode.
    """

    def __init__(self, text: str = "") -> None:
        self.buf: list[str] = list(text)
        self.cursor = len(self.buf)

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

    def handle(self, ch: str, getch: Callable[[], str]) -> str:
        """Apply one keypress. Returns: submit / interrupt / eof /
        clear-screen / shift-tab / tab / edit."""
        if ch == "\r":                                # Enter — submit
            return "submit"
        if ch == "\n":                                # literal newline from paste
            self.buf.insert(self.cursor, "\n")
            self.cursor += 1
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
        elif ch in ("\x08", "\x7f"):                  # backspace
            if self.cursor > 0:
                self.cursor -= 1
                del self.buf[self.cursor]
        elif ch == "\x0b":                            # Ctrl+K — kill to end
            del self.buf[self.cursor:]
        elif ch == "\x15":                            # Ctrl+U — kill to start
            del self.buf[:self.cursor]
            self.cursor = 0
        elif ch == "\x17":                            # Ctrl+W — delete word back
            end = self.cursor
            self._word_left()
            del self.buf[self.cursor:end]
        elif ch == "\x1b":
            return self._escape(getch)
        elif ch.isprintable():
            self.buf.insert(self.cursor, ch)
            self.cursor += 1
        return "edit"

    def _escape(self, getch: Callable[[], str]) -> str:
        nxt = getch()
        if nxt == "\r":                               # Alt+Enter — insert newline
            self.buf.insert(self.cursor, "\n")
            self.cursor += 1
        elif nxt == "[":
            code = getch()
            if code == "Z":                           # Shift+Tab
                return "shift-tab"
            if code == "C" and self.cursor < len(self.buf):   # right
                self.cursor += 1
            elif code == "D" and self.cursor > 0:             # left
                self.cursor -= 1
            elif code == "H":                         # Home
                self.cursor = 0
            elif code == "F":                         # End
                self.cursor = len(self.buf)
            elif code == "1":                         # ^[[1~ Home / ^[[1;5x Ctrl+arrow
                nxt2 = getch()
                if nxt2 == "~":
                    self.cursor = 0
                elif nxt2 == ";":
                    mod = getch()
                    dr = getch()                      # final byte of the sequence
                    if mod == "5":
                        if dr == "D":
                            self._word_left()
                        elif dr == "C":
                            self._word_right()
            elif code == "4":                         # ^[[4~ End
                if getch() == "~":
                    self.cursor = len(self.buf)
            elif code == "3":                         # ^[[3~ Delete
                if getch() == "~" and self.cursor < len(self.buf):
                    del self.buf[self.cursor]
            elif code and code in "0256789":          # any other CSI — consume
                while True:
                    c = getch()
                    if c in ("~", "") or c.isalpha():
                        break
            return "edit"
        if nxt == "b":                                # Alt+b / Alt+left — word left
            self._word_left()
        elif nxt == "f":                              # Alt+f / Alt+right — word right
            self._word_right()
        elif nxt == "O":                              # SS3: ^[OH Home, ^[OF End
            code = getch()
            if code == "H":
                self.cursor = 0
            elif code == "F":
                self.cursor = len(self.buf)
        # other Alt+key sequences (or a bare ESC): ignore
        return "edit"


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

    def redraw() -> None:
        parts: list[str] = []
        rows = _dock_ensure(parts)
        # ghost preview: faded remainder of the first matching /command
        ghost = _command_ghost(ed.text())
        if rows == 0:                     # terminal too short — inline prompt
            parts.append("\r\033[K" + prompt_fn() + ed.text() + dim(ghost))
            cursor_offset = len(ed.buf) - ed.cursor + len(ghost)
        else:
            top = rows - 2
            head = mode_badge(state.approval_mode).strip() + dim(" · ") + statusline(state)
            parts.append("\033[?25h")
            parts.append(f"\033[{top};1H\033[K" + _box_top(head))
            inp_row = top + 1
            bot_row = top + 2
            parts.append(f"\033[{bot_row};1H\033[K" + _box_bottom())
            # the input row is drawn last so the cursor parks inside the box
            text_display = ed.text()
            if "\n" in text_display:
                first = text_display.split("\n")[0]
                nl = text_display.count("\n")
                text_display = first + dim(f" +{nl} more line{'s' if nl > 1 else ''}")
                ghost = ""                # display text no longer maps to buf
                cursor_offset = 0
            else:
                cursor_offset = len(ed.buf) - ed.cursor + len(ghost)
            parts.append(f"\033[{inp_row};1H\033[K" + _box_mid(text_display + dim(ghost)))
        if cursor_offset > 0:
            parts.append(f"\033[{cursor_offset}D")
        sys.stdout.write("".join(parts))
        sys.stdout.flush()

    def to_flow() -> None:
        """Put the cursor back where the scrolling flow left off."""
        if _Dock.active:
            sys.stdout.write("\0338")
            sys.stdout.flush()

    try:
        tty.setraw(fd)
        parts: list[str] = []
        _dock_ensure(parts)               # may scroll — must precede the save
        parts.append("\0337")             # remember where the flow left off
        sys.stdout.write("".join(parts))
        sys.stdout.flush()
        redraw()
        while True:
            ch = keys.getch()
            if ch == "":                             # stdin closed — never recovers
                to_flow()
                raise EOFError
            act = ed.handle(ch, lambda: keys.getch(0.05))
            if act == "submit":
                text = ed.text()
                # clear the input row, then echo the line into the flow (bold)
                if _Dock.active:
                    sys.stdout.write(f"\033[{_Dock.rows - 1};1H\033[K" + _box_mid("")
                                     + "\0338\r\n" + prompt_fn() + bold(text) + "\r\n")
                else:
                    sys.stdout.write("\r\033[K" + prompt_fn() + bold(text) + "\r\n")
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
                        # print into the flow above the box, then re-save it
                        sys.stdout.write("\0338" + listing + "\r\n\0337")
                    else:
                        sys.stdout.write("\r\n" + listing + "\r\n")
            elif act == "clear-screen":
                sys.stdout.write("\033[2J\033[H\0337")  # re-save the flow spot
            redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


class TypeAhead:
    """Keyboard capture while a turn is running: type-ahead + a prompt queue.

    While the model works, a reader thread keeps the terminal in cbreak mode
    and feeds keystrokes into a LineEditor; the draft is echoed in the docked
    input row at the bottom of the spinner's live region (status above, dim
    divider between). Enter queues the line — queued prompts run in order as
    soon as the current turn finishes. Unsubmitted text survives the turn and
    pre-fills the next interactive prompt.

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

    @property
    def active(self) -> bool:
        return self._thread is not None

    def start(self, spinner: Spinner) -> None:
        if not _raw_capable() or self.active:
            return
        self._spinner = spinner
        spinner.extra = self._badge          # queued count, on the status row
        spinner.lead = lambda: mode_badge(self.state.approval_mode).strip()
        spinner.input_line = self._input_row  # the input box's middle row
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
            self._spinner.input_line = None
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
        """Queued-prompt count, appended to the status row."""
        with self._lock:
            queued = len(self.queue)
        return "  " + yellow(f"[{queued} queued]") if queued else ""

    def _input_row(self) -> str:
        """The input box's middle-row content: the draft (or a confirm
        question + answer), with an inverse-video block standing in for the
        hardware cursor (which stays hidden in the flow while a turn runs)."""
        with self._lock:
            req = self._req_prompt
            text = self._editor.text()
            cur = self._editor.cursor
        prompt = req if req is not None else ""
        plain_len = len(_ANSI.sub("", prompt))
        # Flatten newlines so windowing works on a single terminal row
        display_text = text.replace("\n", " ")
        # Adjust cursor: newlines before cursor aren't visible chars
        display_cur = cur - text[:cur].count("\n")
        # Cap at display length (cursor can't be past the flattened text)
        display_cur = min(display_cur, len(display_text))
        # window the draft so the cursor stays visible on one terminal row
        width = max(10, shutil.get_terminal_size().columns - plain_len - 5)
        start = 0
        if len(display_text) >= width:
            start = min(max(0, display_cur - width + 1), len(display_text) - width)
        visible = display_text[start:start + width]
        vcur = display_cur - start
        ghost = ""
        if req is None and cur == len(text):
            # faded preview of the first matching /command, kept on this row
            ghost = _command_ghost(text)[:max(0, width - len(visible))]
        if vcur < len(visible):
            body = visible[:vcur] + _c("7", visible[vcur]) + visible[vcur + 1:]
        elif ghost:
            # the block cursor sits on the first suggested char, rest faded
            body = visible + _c("7", ghost[0]) + dim(ghost[1:])
        else:
            body = visible + _c("7", " ")
        return prompt + ("…" if start else "") + body

    # --- reader thread ----------------------------------------------------
    def _run(self) -> None:
        keys = _keys()
        while not self._stop.is_set():
            if not keys.ready(0.05):
                continue
            ch = keys.getch()
            if ch == "":                              # stdin closed
                return
            self._on_key(ch, lambda: keys.getch(0.05))

    def _on_key(self, ch: str, getch: Callable[[], str]) -> None:
        with self._lock:
            in_request = self._req_prompt is not None
            act = self._editor.handle(ch, getch)
            text = self._editor.text().strip() if act == "submit" else ""

        if in_request:
            if act == "submit":
                self._req_text = text
                self._req_done.set()
            elif act in ("interrupt", "eof"):
                self._req_done.set()                  # empty answer = decline
            if self._spinner:
                self._spinner.render_now()
            return

        if act == "submit":
            with self._lock:
                if text:
                    self.queue.append(text)
                self._editor.set_text("")
            if text and self._spinner:
                self._spinner.println(dim("  + queued: ") + text)
        elif act == "shift-tab":
            _cycle_mode(self.state)
            if self._spinner:
                self._spinner.println(
                    dim("  approval mode → ") + mode_badge(self.state.approval_mode).strip())
        elif act == "tab":
            with self._lock:
                matches = _complete_command(self._editor)
            if matches and self._spinner:             # ambiguous — list candidates
                self._spinner.println(dim("  " + "  ".join(matches)))
        # eof / interrupt / clear-screen make no sense mid-turn — ignored
        if self._spinner:
            self._spinner.render_now()


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
            print("\n" + bold(green("› ")) + bold(content))
        elif role == "assistant" and content:
            print("\n" + render_assistant(content))


# --- main loop ---------------------------------------------------------
def banner(state: State) -> None:
    print(bold(cyan(f"hrns {__version__}")) + dim(" — DeepSeek coding harness"))
    conn = green("connected") if state.client else red("not connected (run /connect)")
    print(dim(f"  model {state.session.model} · session {state.session.id} · {conn}"))
    if state.session.turn_count() > 0:
        print(dim(f"  resumed — {state.session.turn_count()} previous turn(s)"))
    print(dim(f"  workspace {workspace_root()}"))
    cyc = "shift+tab" if _raw_capable() else "/mode"
    print(dim(f"  approval {state.approval_mode} ({cyc} to cycle) · /help · Ctrl-D to exit"))
    if _raw_capable():
        print(dim("  type while it works — Enter queues your next prompt · Ctrl-C interrupts"))


def main() -> None:
    set_workspace_root(Path.cwd())  # the folder hrns was opened in is the sandbox root
    cfg = Config.load()

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
            client = DeepSeekClient(cfg.api_key, cfg.base_url)
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
    prompt_fn = lambda: mode_badge(state.approval_mode) + bold(green("› "))  # noqa: E731
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
            print(("\n" if docked else "") + prompt_fn() + bold(line) + dim("  (queued)"))
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
            print(yellow("Not connected. Run /connect to set your DeepSeek API key."))
            continue

        echo = prompt_fn() + bold(line) + (dim("  (queued)") if from_queue else "")
        state.session.log("user", "\n" + echo)
        try:
            run_turn(state, line, typeahead)
        except KeyboardInterrupt:
            # ^C that landed outside run_turn's own guard (e.g. while saving)
            print(yellow("\ninterrupted"))
            state.session.save(cfg.sessions_dir)


if __name__ == "__main__":
    main()
