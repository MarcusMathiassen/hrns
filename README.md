# hrns

A minimal, **DeepSeek-focused** command-line coding harness, built around one
idea: **maximize prefix cache hits**. DeepSeek serves repeated request prefixes
from an on-disk KV cache at ~1/50th the price of a fresh "cache miss" token, so
`hrns` is designed so that almost every token you resend is one DeepSeek has
already cached.

Zero dependencies — pure Python standard library.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/MarcusMathiassen/hrns/main/install.sh | sh
```

Requires Python 3.12+ and either `curl` or `wget`. Installs to your user site-packages (`pip install --user`).

## The caching strategy

DeepSeek's [context caching](https://api-docs.deepseek.com/guides/kv_cache) is
automatic and keyed on the **prefix** of your `messages`. A cache hit happens
when the leading bytes of a request are byte-identical to one the platform has
already seen. `hrns` leans into that:

- **A static system prompt is the stable anchor** (`messages[0]`). Persistent
  memory is *snapshotted* into it once, at session creation, and never mutated.
- **The conversation log is append-only.** Past messages are never edited or
  reordered, so the prefix only ever grows — it never shifts.
- **Volatile data stays out of the prefix** (no "today's date" in the system
  prompt) so it can't silently invalidate the cache.
- **Sessions persist to disk byte-for-byte**, so resuming one days later replays
  an identical prefix and *still* hits DeepSeek's cache (within its TTL).
- **Tool loops are cache gold**: each step resends the whole growing prefix, so
  every round-trip after the first is mostly served from cache.

Every turn prints what actually happened:

```
cache 3,412 hit / 88 miss (97%) · 156 out · $0.000056 (saved $0.000477)
```

## Install / run

```bash
# run in place, no install:
python -m hrns

# or install the `hrns` command:
pip install -e .
hrns
```

First run:

```
› /connect          # enter your DeepSeek API key + pick a model, validates it
› read pyproject.toml and tell me the package name
```

The key is also picked up automatically from `DEEPSEEK_API_KEY` in the
environment or a project-local `.env`.

## Commands

Command names **Tab-complete** at the prompt: a unique prefix completes fully
(`/se⇥` → `/sessions `), an ambiguous one extends as far as possible and a
second ⇥ lists the candidates.

| Command | What it does |
|---|---|
| `/connect` | Configure & test the DeepSeek connection (key, base URL, model) — **remembered for next run** |
| `/sessions` | List saved sessions; `/sessions <id\|#>` resumes one (re-hits the cache) |
| `/clear` | Archive the current session and start a fresh one |
| `/memory` | `add <text>` / `rm <id>` / `clear` — durable facts baked into new sessions |
| `/model` | Show or set the model for new sessions — **remembered for next run** |
| `/mode` | Cycle approval mode: confirm → auto-edit → auto (or **Shift+Tab**) |
| `/stats` | Cumulative tokens, cache-hit rate, and cost (with cache savings) |
| `/help`, `/quit` | … |

### Type while it thinks — the prompt never blocks

The input box is **pinned to the bottom of the terminal** (a scroll region
keeps all output above it), idle or mid-reasoning alike. Its top border shows
the approval mode plus the session vitals — or the live spinner while a turn
runs — and you type inside it:

```
  …replies, diffs, and tool calls scroll here…

┌─ auto · deepseek-chat · 95.0% · 3 turns · 12.3k · $0.01 · $10.00
│ your next prompt, typed even while it works…
└─
```

Press **Enter** to queue the draft — queued prompts run in order as soon as
the current turn finishes, and anything left half-typed pre-fills the next
prompt. **Ctrl-C** interrupts the current turn:
partial progress is saved, and any tool calls that never ran are answered with
placeholders so the append-only message log stays valid.

The scrollback shows what matters: your words (bold, `› `-prefixed), replies
(cyan `⏺`), one dim `•` line per tool call, edit diffs/previews, and a single
cache/cost summary per turn — no per-call token chatter.

Every session also remembers its on-screen transcript: resuming one (auto on
start, or via `/sessions <id|#>`) replays the conversation **exactly** as you
left it — tool calls, diffs, and all. The transcript is display-only and
never sent to the API, so it cannot disturb the cached prefix.

## Tools

A modern coding-agent tool set:

| Tool | Purpose |
|---|---|
| `read_file` | Read with line numbers + `offset`/`limit` paging |
| `grep` | Regex content search (ripgrep-backed, stdlib fallback) |
| `glob` | Find files by pattern (`**/*.py`), newest-first |
| `list_dir` | Typed directory listing with sizes |
| `edit_file` | Exact, unique `str_replace`; returns a unified diff |
| `create_file` | New files only (won't clobber unless `overwrite`) |
| `run_bash` | Shell with exit status + timeout |

Read-only tools (`read_file`, `grep`, `glob`, `list_dir`) run automatically.
Mutating tools (`edit_file`, `create_file`, `run_bash`) show a colored diff/preview
and ask for confirmation first. Tool round-trips reuse the growing cached prefix,
so a multi-step edit is nearly free after the first call.

### Workspace containment

hrns is sandboxed to the directory it was launched in. Any tool whose target
path resolves **outside** that root — via an absolute path, `../`, or a symlink —
is blocked until you approve that specific access, *even for read-only tools*:

```
⚠ read_file wants to read a path OUTSIDE the workspace:
    /etc/passwd
    workspace: /Users/you/project
  allow access outside the workspace? [y/N]
```

`run_bash` is unconstrained (a shell can reach anywhere), so it always asks for
confirmation regardless.

### Approval modes (Shift+Tab)

Press **Shift+Tab** at the prompt to cycle how much hrns asks before acting
(or use `/mode`). The current mode shows as a badge on the prompt: `[confirm] ›`.

| Mode | Behavior |
|---|---|
| `confirm` (default) | ask before every edit, file write, and command |
| `auto-edit` | auto-approve `edit_file`/`create_file`; still ask for shell + outside-workspace |
| `auto` | auto-approve all in-workspace actions incl. `run_bash`; still ask for outside-workspace |

Auto-approved actions still print their diff/preview so you can see what happened.
**Out-of-workspace access always asks, in every mode** — the sandbox boundary
never auto-approves. The mode is remembered in `~/.hrns/config.json`, and the
prompt badge always shows which mode you're in.

## Where data lives

Everything persists under `~/.hrns/` (override with `HRNS_HOME`):

```
~/.hrns/
  config.json          # base_url, model, api key (chmod 600)
  sessions/<id>.json   # full append-only message log + usage per session
  memory/memory.json   # persistent cross-session memory
```

## Models & pricing

Defaults to `deepseek-chat`. Also knows `deepseek-reasoner` (thinking mode —
its `reasoning_content` is folded into the status spinner), and the `deepseek-v4-flash` /
`deepseek-v4-pro` names. Pricing is from DeepSeek's
[pricing page](https://api-docs.deepseek.com/quick_start/pricing) and lives in
`hrns/config.py`.

## Layout

```
hrns/
  config.py   # paths, model pricing, key resolution
  client.py   # streaming DeepSeek client (stdlib urllib), cache-aware usage
  session.py  # append-only, prefix-stable session log + persistence
  memory.py   # persistent cross-session memory
  tools.py    # read_file / grep / glob / list_dir / edit_file / create_file / run_bash
  cli.py      # REPL, slash commands, cache/cost status line
```

This is a starting point — deliberately small and easy to extend.
