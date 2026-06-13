# hrns — Roadmap & Improvement Ideas

## 1. Agent Capabilities

### Multi-file refactoring
Currently the model makes one tool call at a time. For refactoring (rename a function across files, extract a module) the model has to plan and execute step by step, which is slow and fragile.
- **Idea:** add a `batch_edit` tool that accepts a list of `(path, old_string, new_string)` pairs and applies them atomically (all-or-nothing with a rollback on failure).
- **Idea:** track "pending changes" during a turn so the model can stage several edits and apply them in one confirm gate.

### Project-aware context
The model only sees what you explicitly `read_file`. It doesn't know the project's structure.
- **Idea:** on session start (or via a `/map` command), generate a compact tree of the project — directory names, file sizes, git history. Inject it into the system prompt or a dedicated tool result.
- **Idea:** `git diff --stat` summarised and fed as context before the first user turn.

### Terminal-aware output
When the model proposes a command, the user runs it, sees the output, and reports back. That's slow.
- **Idea:** for `run_bash` results, inject the exit code and first/last N lines of stdout/stderr into the next turn automatically so the model can self-correct.

## 2. UX & Editing

### Multiline editor (proper)
Currently multiline paste is flattened to `+N more lines` — you can't see or edit the full text.
- **Idea:** when Enter is pressed on an empty buffer, submit. When pressed on a non-empty buffer, insert a newline. Full multiline editing with up/down arrow navigation between lines, visible in the dock.

### Command history
Every REPL needs it.
- **Idea:** save every submitted line to `~/.hrns/history.jsonl`. Up/down arrows cycle through history. Per-project history (keyed on workspace root) so you get relevant suggestions.

### Syntax highlighting in the input field
- **Idea:** basic /command highlighting (cyan for `/compact`, yellow for args). Possibly also highlighting for filenames and paths.

### `/undo` for the last tool action
- **Idea:** keep a stack of previous file contents before each `edit_file`/`create_file`. `/undo` restores the most recent one. This is harder with a stateless API — would need local shadow copies.

## 3. Performance & Caching

### Lazy model name in status line
`_model_name()` is hardcoded in a dict. Could pull from the API's `/models` list for unknown models.

### Smarter context window tracking
The API returns `prompt_tokens` per request. hrns could estimate when the 1M context window is nearly full and suggest `/compact` proactively.

### Faster resume
On resume, the session transcript is replayed. For long sessions this could be slow (lots of ANSI rendering).
- **Idea:** play the transcript in a background thread above the dock so the user can start typing immediately.
- **Idea:** save a "last N lines" snapshot instead of the full transcript.

## 4. New Commands

### `/summarise`
Like `/compact` but doesn't replace history — just prints a summary and optionally saves it into memory.

### `/git` shorthand
`/git log --oneline -5` routes to `run_bash` without the confirm gate for read-only git commands. Possibly `/git commit -m "..."` with a cost warning.

### `/history`
List recent commands (in-session and across sessions).

### `/export`
Export the session as Markdown or HTML (with ANSI stripped) for sharing.

### `/plan`
Ask the model to generate a step-by-step plan (without executing any tools) and present it for approval.

## 5. Architecture

### Ditch the dock for a more standard approach
The 3-row bottom dock is clever but fragile — resize bugs, invisible cursor issues, weird interactions with tmux/scrollback.
- **Idea:** simpler prompt approach: print the status line, then `input()`, and handle Ctrl-R for history, etc. Lose the type-ahead but gain terminal compatibility.

### Pipe mode / CI
`echo "explain this" | hrns` should work non-interactively, printing just the reply.
- **Idea:** detect stdin not being a tty, skip the dock and raw mode entirely, run one turn, print output, exit.

### Config file as TOML or YAML
`~/.hrns/config.json` works but it's not user-friendly. A TOML file with comments would let users configure tools, aliases, and defaults by hand.

### Plugin system
Simple Python hooks: before/after each tool call, before/after each turn. Users could drop a `.py` file in `~/.hrns/hooks/`.

## 6. Testing

Zero tests currently. The codebase is small enough that adding them would be straightforward.
- **Idea:** `pytest` (it's stdlib-only, but pytest is nicer). Mock the `DeepSeekClient` for deterministic test fixtures. Test the confirm gate, the session round-trip (save + load gives identical bytes), the input editor, the status line rendering.
