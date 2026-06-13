"""hrns REPL: chat loop, slash commands, and the cache-aware status line."""

from __future__ import annotations

import difflib
import getpass
import itertools
import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


# --- "working…" spinner with an elapsed timer -------------------------
class Spinner:
    """A single self-erasing status line animated from a background thread.

    The main thread does the blocking work and calls .set(label) to describe
    what's happening ("thinking", "reading config.py"). The whole noisy
    intermediate stream (reasoning, tool I/O) is collapsed into this one line.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self) -> None:
        self.enabled = _TTY
        self._label = "working"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self.elapsed = 0.0

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
            if not self._paused.is_set():
                with self._lock:
                    label = self._label
                elapsed = time.monotonic() - self._t0
                sys.stdout.write(f"\r\033[K{cyan(frame)} {label} {dim(f'{elapsed:.1f}s')}")
                sys.stdout.flush()
            time.sleep(0.09)

    def pause(self) -> None:
        """Hide the line so the main thread can prompt for input."""
        if not self.enabled:
            return
        self._paused.set()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def resume(self) -> None:
        self._paused.clear()

    def println(self, text: str) -> None:
        """Print a full line without colliding with the animated status line."""
        self.pause()
        print(text)
        self.resume()

    def stop(self) -> None:
        self.elapsed = time.monotonic() - self._t0
        if not self.enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


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
        "confirm": dim("[confirm]"),
        "auto-edit": yellow("[auto-edit]"),
        "auto": _c("1;31", "[AUTO]"),
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


# --- cache / cost reporting -------------------------------------------
def cache_line(model: str, usage: dict) -> str:
    p = pricing_for(model)
    hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    out = int(usage.get("completion_tokens", 0) or 0)
    total_in = hit + miss
    rate = (hit / total_in * 100) if total_in else 0.0
    cost = hit / 1e6 * p["cache_hit"] + miss / 1e6 * p["cache_miss"] + out / 1e6 * p["output"]
    saved = hit / 1e6 * (p["cache_miss"] - p["cache_hit"])
    return dim(
        f"cache {hit:,} hit / {miss:,} miss ({rate:.0f}%) · {out:,} out · "
        f"${cost:.6f} (saved ${saved:.6f})"
    )


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
    return dim(f"  in {_human(hit + miss)} · out {_human(out)} · ${cost:.6f} · {elapsed:.1f}s")


def run_turn(state: State, user_input: str) -> None:
    state.session.append({"role": "user", "content": user_input})
    cfg, session, client = state.cfg, state.session, state.client
    assert client is not None

    print(dim(f"  sending {_human(session.context_tokens)} tok · {session.turn_count()} turn(s)"))

    spinner = Spinner()
    spinner.start("calling api")
    agg = {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0, "completion_tokens": 0}
    final_text = ""
    error: Optional[str] = None

    try:
        for iteration in range(cfg.max_tool_iters):
            buf: list[str] = []
            out_tok = 0
            t0 = time.monotonic()

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

            api_elapsed = time.monotonic() - t0
            session.append(result.message)
            session.record_usage(result.usage)
            for k in agg:
                agg[k] += int(result.usage.get(k, 0) or 0)

            spinner.println(_api_stats(session.model, result.usage, api_elapsed))

            tool_calls = result.message.get("tool_calls")
            if not tool_calls:
                final_text = result.message.get("content") or ""
                break

            spinner.println(dim(f"  {len(tool_calls)} tool call(s) · iter {iteration + 1}/{cfg.max_tool_iters}"))

            for tc in tool_calls:
                fn = tc["function"]
                spinner.set(_tool_label(fn))
                out = execute(fn["name"], fn["arguments"], confirm=_make_confirm(spinner, state))
                if fn["name"] == "run_bash":
                    preview = (out or "").strip()[:80].replace("\n", " ")
                    if preview:
                        spinner.println(dim(f"    -> {preview}"))
                session.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            # loop so the model can read the tool results
        else:
            final_text = final_text or f"_Stopped after {cfg.max_tool_iters} tool steps._"
    finally:
        spinner.stop()

    if error:
        print(red(f"deepseek error: {error}"))
        session.save(cfg.sessions_dir)
        return

    print()
    print(format_markdown(final_text) if final_text else dim("(no text response)"))
    print(cache_line(session.model, agg) + dim(f" · {spinner.elapsed:.1f}s"))
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


def _diff_preview(old: str, new: str, limit: int = 14) -> str:
    """A small colored +/- preview of an edit, for the confirm prompt."""
    body = [
        ln for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1)
        if not ln.startswith(("---", "+++", "@@"))
    ]
    out = []
    for ln in body[:limit]:
        out.append(green(ln) if ln.startswith("+") else red(ln) if ln.startswith("-") else dim(ln))
    if len(body) > limit:
        out.append(dim(f"… (+{len(body) - limit} more diff lines)"))
    return "\n".join("    " + o for o in out)


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
        head = f"{magenta('edit')} {bold(str(args.get('path', '?')))}"
        if args.get("replace_all"):
            head += dim(" (replace_all)")
        return head + "\n" + _diff_preview(args.get("old_string", ""), args.get("new_string", ""))
    if name == "create_file":
        content = args.get("content", "")
        preview = "\n".join("    " + green("+ " + ln) for ln in content.splitlines()[:14])
        more = dim(f"\n    … (+{len(content.splitlines()) - 14} more lines)") if len(content.splitlines()) > 14 else ""
        return f"{green('create')} {bold(str(args.get('path', '?')))}\n{preview}{more}"
    if name == "run_bash":
        return f"{yellow('run')}\n    {bold(str(args.get('command', '')))}"
    return f"{blue(name)} {dim(str(args))}"


def _make_confirm(spinner: Spinner, state: State):
    """Confirm gate; pauses the spinner to show a preview before applying.

    In an auto mode the action is shown (diff/preview) and auto-approved — except
    out-of-workspace access, which always asks.
    """
    def confirm(name: str, args: dict, reason: Optional[str] = None) -> bool:
        spinner.pause()
        print(_confirm_preview(name, args, reason))
        if _auto_approves(state.approval_mode, name, reason):
            print(dim(f"  ✓ auto-approved · {state.approval_mode}"))
            spinner.resume()
            return True
        prompt = ("  allow access outside the workspace? [y/N] "
                  if reason == "outside-workspace" else "  apply? [y/N] ")
        try:
            ans = input(yellow(prompt)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
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
        print(green(f"Resumed {chosen.id} ({chosen.turn_count()} turns). "
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

    spinner = Spinner()
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
    print(dim("  compact ·") + _api_stats(session.model, result.usage, elapsed))

    system = session.messages[0]
    session.messages.clear()
    session.messages.append(system)
    session.messages.append({"role": "user", "content": summary})
    # approximate context — roughly the messages we now hold
    session.context_tokens = sum(len(str(m)) for m in session.messages) // 4
    session.save(state.cfg.sessions_dir)
    print(green(f"Compacted {len(non_system)} messages into a summary."))

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


# --- the live status line shown above the prompt ----------------------
def statusline(state: State) -> str:
    """One colored line of session vitals, rendered above each input prompt.

    Each metric gets its own color so it's scannable at a glance:
      blue=turns  magenta=tokens processed  yellow=context fill
      green=cache hit rate  cyan=cost
    """
    s = state.session
    u = s.usage
    win = context_window(s.model)
    cum_tok = u["prompt_tokens"] + u["completion_tokens"]
    hit, miss = u["prompt_cache_hit_tokens"], u["prompt_cache_miss_tokens"]
    cache_rate = (hit / (hit + miss) * 100) if (hit + miss) else None
    cost = s.cost(pricing_for(s.model))
    bal = state.cfg.balance

    turns = s.turn_count()
    remaining = win - s.context_tokens
    segs = [
        blue(f"{turns} turn{'' if turns == 1 else 's'}"),
        magenta(f"Σ {_human(s.context_tokens)} ctx / {_human(cum_tok)} cum"),
        yellow(f"{_human(remaining)} remain"),
        green(f"{cache_rate:.0f}% cache" if cache_rate is not None else "--% cache"),
        cyan(_money(cost)),
        cyan(f"${bal:.2f}") + dim(" bal") if bal is not None else dim("-- bal"),
        cyan(s.model.lower()),
    ]
    return dim(" · ").join(segs)


# --- input: a tiny raw-mode reader so Shift+Tab can cycle modes -------
def _raw_capable() -> bool:
    return bool(_TTY and termios is not None and sys.stdin.isatty())


def read_line(prompt_fn, state: State) -> str:
    """Read one line with readline-like editing (cursor, word-delete, etc.).

    Supports:
      left/right/home/end      — cursor movement
      ctrl+left/ctrl+right     — word-jump
      backspace / delete       — char delete
      ctrl+W / alt+backspace   — delete word backwards
      ctrl+U                   — delete to start of line
      ctrl+K                   — delete to end of line
      ctrl+A / ctrl+E          — home / end
      ctrl+L                   — clear screen
      shift+tab                — cycle approval mode
    """
    if not _raw_capable():
        return input(prompt_fn())

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    buf: list[str] = []
    cursor = 0

    def redraw() -> None:
        line = "".join(buf)
        sys.stdout.write("\r\033[K" + prompt_fn() + line)
        # move cursor back from end to actual position
        n = len(buf) - cursor
        if n > 0:
            sys.stdout.write(f"\033[{n}D")
        sys.stdout.flush()

    def _is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    def _del_word_back() -> None:
        nonlocal cursor
        if cursor == 0:
            return
        end = cursor
        # skip spaces
        while cursor > 0 and buf[cursor - 1] == " ":
            cursor -= 1
        # skip word chars
        while cursor > 0 and _is_word_char(buf[cursor - 1]):
            cursor -= 1
        del buf[cursor:end]
        redraw()

    try:
        tty.setraw(fd)
        redraw()
        while True:
            ch = sys.stdin.read(1)
            if ch == "":                             # stdin closed — never recovers
                raise EOFError
            if ch == "\x04":                         # Ctrl-D
                if buf:
                    continue
                raise EOFError
            if ch in ("\r", "\n"):
                sys.stdout.write("\r\n"); sys.stdout.flush()
                return "".join(buf)
            if ch == "\x03":                         # Ctrl-C
                raise KeyboardInterrupt

            # --- control characters ---------------------------------
            if ch == "\x01":                         # Ctrl+A — home
                cursor = 0; redraw(); continue
            if ch == "\x05":                         # Ctrl+E — end
                cursor = len(buf); redraw(); continue
            if ch == "\x08":                         # Ctrl+H / backspace (raw)
                if cursor > 0:
                    cursor -= 1; del buf[cursor]; redraw()
                continue
            if ch == "\x7f":                         # DEL / backspace
                if cursor > 0:
                    cursor -= 1; del buf[cursor]; redraw()
                continue
            if ch == "\x0b":                         # Ctrl+K — kill to end
                if cursor < len(buf):
                    del buf[cursor:]; redraw()
                continue
            if ch == "\x15":                         # Ctrl+U — kill to start
                if cursor > 0:
                    del buf[:cursor]; cursor = 0; redraw()
                continue
            if ch == "\x17":                         # Ctrl+W — delete word back
                _del_word_back(); continue
            if ch == "\x0c":                         # Ctrl+L — clear screen
                sys.stdout.write("\033[2J\033[H"); redraw(); continue

            # --- escape sequences ------------------------------------
            if ch == "\x1b":
                nxt = sys.stdin.read(1)
                if nxt == "[":
                    code = sys.stdin.read(1)
                    if code == "Z":                  # Shift+Tab
                        _cycle_mode(state); redraw(); continue
                    if code in "ABCD":               # arrows
                        if code == "C" and cursor < len(buf):   # right
                            cursor += 1
                        elif code == "D" and cursor > 0:         # left
                            cursor -= 1
                        # up/down (A/B) ignored
                        redraw(); continue
                    if code == "H":                  # Home
                        cursor = 0; redraw(); continue
                    if code == "F":                  # End
                        cursor = len(buf); redraw(); continue
                    if code == "1":                  # Home / Ctrl+arrow (grab next char)
                        nxt2 = sys.stdin.read(1)
                        if nxt2 == "~":             # Home (^[[1~)
                            cursor = 0; redraw(); continue
                        if nxt2 == ";":             # ^[[1;5D / ^[[1;5C — Ctrl+arrow
                            mod = sys.stdin.read(1)
                            dr = sys.stdin.read(1)   # final byte of the sequence
                            if mod == "5":           # Ctrl
                                if dr == "D":        # Ctrl+left — word left
                                    while cursor > 0 and buf[cursor - 1] == " ":
                                        cursor -= 1
                                    while cursor > 0 and _is_word_char(buf[cursor - 1]):
                                        cursor -= 1
                                elif dr == "C":      # Ctrl+right — word right
                                    while cursor < len(buf) and buf[cursor] == " ":
                                        cursor += 1
                                    while cursor < len(buf) and _is_word_char(buf[cursor]):
                                        cursor += 1
                                redraw()
                        continue
                    if code == "4" and sys.stdin.read(1) == "~":  # End (^[[4~)
                        cursor = len(buf); redraw(); continue
                    if code == "3":                  # Delete (^[[3~)
                        nxt2 = sys.stdin.read(1)
                        if nxt2 == "~" and cursor < len(buf):
                            del buf[cursor]; redraw()
                        continue
                    # consume remaining of any other CSI sequence
                    if code in "0123456789":
                        while True:
                            c = sys.stdin.read(1)
                            if c in ("~", "") or (c >= "A" and c <= "Z") or (c >= "a" and c <= "z"):
                                break
                    continue
                if nxt == "b":                       # Alt+left  — word left
                    while cursor > 0 and buf[cursor - 1] == " ":
                        cursor -= 1
                    while cursor > 0 and _is_word_char(buf[cursor - 1]):
                        cursor -= 1
                    redraw(); continue
                if nxt == "f":                       # Alt+right — word right
                    while cursor < len(buf) and buf[cursor] == " ":
                        cursor += 1
                    while cursor < len(buf) and _is_word_char(buf[cursor]):
                        cursor += 1
                    redraw(); continue
                if nxt == "O":                       # SS3: ^[OH Home, ^[OF End
                    code = sys.stdin.read(1)
                    if code == "H":
                        cursor = 0; redraw()
                    elif code == "F":
                        cursor = len(buf); redraw()
                    continue
                # any other Alt+key is a 2-byte sequence — ignore it
                continue

            # --- printable ------------------------------------------
            if ch.isprintable():
                buf.insert(cursor, ch); cursor += 1
                redraw()
                continue
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


# --- replay previous sessions on resume --------------------------------
def _replay_conversation(state: State) -> None:
    """Replay the session's messages 1:1, exactly as they appeared live."""
    session = state.session
    for msg in session.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue

        if role == "user" and content:
            print()
            print(content)
            continue

        if role == "assistant" and content:
            print()
            print(format_markdown(content))
            continue


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
    prompt_fn = lambda: mode_badge(state.approval_mode) + bold(green("› "))  # noqa: E731
    while True:
        print("\n" + statusline(state))
        try:
            line = read_line(prompt_fn, state).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd = line.split(maxsplit=1)[0]
            rest = line[len(cmd):].strip()
            if cmd in ("/quit", "/exit", "/q"):
                break
            handler = COMMANDS.get(cmd)
            if handler:
                handler(state, rest)
            else:
                print(red(f"Unknown command {cmd}. Try /help."))
            continue

        if state.client is None:
            print(yellow("Not connected. Run /connect to set your DeepSeek API key."))
            continue
        run_turn(state, line)

    state.session.save(cfg.sessions_dir)
    print(dim(f"Saved session {state.session.id}. Bye."))


if __name__ == "__main__":
    main()
