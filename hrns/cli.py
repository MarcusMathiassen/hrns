"""hrns REPL: chat loop, slash commands, and the cache-aware status line."""

from __future__ import annotations

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

from hrns import __version__, memory
from hrns.client import ChatResult, DeepSeekClient, DeepSeekError
from hrns.config import Config, pricing_for
from hrns.session import Session, list_sessions
from hrns.tools import TOOL_SCHEMAS, execute

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


BASE_SYSTEM_PROMPT = (
    "You are hrns, a command-line coding assistant powered by DeepSeek. "
    "You help with software engineering tasks in the user's current project. "
    "You can read files, list directories, write files, and run shell commands "
    "via tools. Prefer small, correct, well-explained changes. When you use a "
    "tool, briefly say why. Keep answers concise.\n\n"
    "Note: do not assume the current date or environment; ask or inspect via "
    "tools when it matters. (Volatile facts are deliberately kept out of this "
    "prompt so the cached prefix stays stable.)"
)


def build_system_prompt(cfg: Config) -> str:
    """Static base + a snapshot of persistent memory. Frozen per session."""
    return BASE_SYSTEM_PROMPT + memory.as_prompt_block(cfg.memory_path)


@dataclass
class State:
    cfg: Config
    session: Session
    client: Optional[DeepSeekClient]


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
        f"⚡ cache {hit:,} hit / {miss:,} miss ({rate:.0f}%) · {out:,} out · "
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
def run_turn(state: State, user_input: str) -> None:
    state.session.append({"role": "user", "content": user_input})
    cfg, session, client = state.cfg, state.session, state.client
    assert client is not None

    spinner = Spinner()
    spinner.start("thinking")
    agg = {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0, "completion_tokens": 0}
    final_text = ""
    error: Optional[str] = None

    try:
        for _ in range(cfg.max_tool_iters):
            buf: list[str] = []

            def on_reasoning(_t: str, _s: Spinner = spinner) -> None:
                _s.set("thinking")

            def on_text(t: str, _b: list = buf, _s: Spinner = spinner) -> None:
                _b.append(t)
                _s.set("writing")

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
            for k in agg:
                agg[k] += int(result.usage.get(k, 0) or 0)

            tool_calls = result.message.get("tool_calls")
            if not tool_calls:
                final_text = result.message.get("content") or ""
                break

            for tc in tool_calls:
                fn = tc["function"]
                spinner.set(_tool_label(fn))
                out = execute(fn["name"], fn["arguments"], confirm=_make_confirm(spinner))
                session.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            # loop so the model can read the tool results
        else:
            final_text = final_text or f"_Stopped after {cfg.max_tool_iters} tool steps._"
    finally:
        spinner.stop()

    if error:
        print(red(f"⚠ deepseek error: {error}"))
        session.save(cfg.sessions_dir)
        return

    print()
    print(format_markdown(final_text) if final_text else dim("(no text response)"))
    print(cache_line(session.model, agg) + dim(f" · {spinner.elapsed:.1f}s"))
    session.save(cfg.sessions_dir)


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
    if name == "read_file":
        return f"reading {Path(str(args.get('path', '?'))).name}"
    if name == "list_dir":
        return f"listing {args.get('path', '.')}"
    if name == "write_file":
        return f"writing {Path(str(args.get('path', '?'))).name}"
    if name == "run_bash":
        return f"running {_short(str(args.get('command', '')), 40)}"
    return name


def _make_confirm(spinner: Spinner):
    """Confirm gate for mutating tools; pauses the spinner to read input."""
    def confirm(name: str, args: dict) -> bool:
        spinner.pause()
        detail = args.get("command") or args.get("path") or ""
        try:
            ans = input(yellow(f"  run {name} [{_short(str(detail), 80)}]? [y/N] ")).strip().lower()
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
        ("/stats", "cumulative token + cache stats for this session"),
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
        marker = green(" ● current") if s.id == state.session.id else ""
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
    cfg.save()
    ok = green("✓ connected")
    here = green("available") if model in models else yellow("not in /models list")
    print(f"{ok} · {len(models)} models · '{model}' {here}")
    print(dim(f"  saved to {cfg.config_path}"))


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


def cmd_stats(state: State, args: str) -> None:
    s = state.session
    u = s.usage
    print(bold(f"Session {s.id}"))
    print(f"  turns      {s.turn_count()}   requests {u['requests']}")
    print(f"  cache      {green(f'{u['prompt_cache_hit_tokens']:,} hit')} / "
          f"{u['prompt_cache_miss_tokens']:,} miss   "
          f"({s.cache_hit_rate*100:.1f}% hit rate)")
    print(f"  output     {u['completion_tokens']:,} tokens")
    p = pricing_for(s.model)
    cost = (u["prompt_cache_hit_tokens"] / 1e6 * p["cache_hit"]
            + u["prompt_cache_miss_tokens"] / 1e6 * p["cache_miss"]
            + u["completion_tokens"] / 1e6 * p["output"])
    saved = u["prompt_cache_hit_tokens"] / 1e6 * (p["cache_miss"] - p["cache_hit"])
    print(f"  cost       ${cost:.6f}   {green(f'(cache saved ${saved:.6f})')}")


COMMANDS = {
    "/help": cmd_help, "/?": cmd_help,
    "/sessions": cmd_sessions,
    "/clear": cmd_clear,
    "/connect": cmd_connect,
    "/memory": cmd_memory,
    "/model": cmd_model,
    "/stats": cmd_stats,
}


# --- main loop ---------------------------------------------------------
def banner(state: State) -> None:
    print(bold(cyan(f"hrns {__version__}")) + dim(" — DeepSeek coding harness"))
    conn = green("connected") if state.client else red("not connected (run /connect)")
    print(dim(f"  model {state.session.model} · session {state.session.id} · {conn}"))
    print(dim("  /help for commands · Ctrl-D or /quit to exit"))


def main() -> None:
    cfg = Config.load()
    session = Session.new(cfg.model, build_system_prompt(cfg))
    client = None
    if cfg.api_key:
        try:
            client = DeepSeekClient(cfg.api_key, cfg.base_url)
        except DeepSeekError:
            client = None
    state = State(cfg=cfg, session=session, client=client)

    banner(state)
    while True:
        try:
            line = input(bold(green("\n› "))).strip()
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
