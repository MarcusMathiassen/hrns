# hrns

A minimal, **DeepSeek-focused** command-line coding harness, built around one
idea: **maximize prefix cache hits**. DeepSeek serves repeated request prefixes
from an on-disk KV cache at ~1/50th the price of a fresh "cache miss" token, so
`hrns` is designed so that almost every token you resend is one DeepSeek has
already cached.

Zero dependencies — pure Python standard library.

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
⚡ cache 3,412 hit / 88 miss (97%) · 156 out · $0.000056 (saved $0.000477)
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

| Command | What it does |
|---|---|
| `/connect` | Configure & test the DeepSeek connection (API key, base URL, model) |
| `/sessions` | List saved sessions; `/sessions <id\|#>` resumes one (re-hits the cache) |
| `/clear` | Archive the current session and start a fresh one |
| `/memory` | `add <text>` / `rm <id>` / `clear` — durable facts baked into new sessions |
| `/model` | Show or set the model for new sessions |
| `/stats` | Cumulative tokens, cache-hit rate, and cost (with cache savings) |
| `/help`, `/quit` | … |

## Tools

The model can `read_file`, `list_dir`, `write_file`, and `run_bash`. Read-only
tools run automatically; writing and running shell commands ask for confirmation.

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
its `reasoning_content` streams dimmed), and the `deepseek-v4-flash` /
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
  tools.py    # read_file / list_dir / write_file / run_bash
  cli.py      # REPL, slash commands, cache/cost status line
```

This is a starting point — deliberately small and easy to extend.
