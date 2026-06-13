"""A tiny, dependency-free DeepSeek client (OpenAI-compatible wire format).

Endpoints used:
    GET  /models           -> connectivity / auth check (used by `/connect`)
    GET  /user/balance     -> account balance (shown in status line)
    POST /chat/completions -> streamed chat, with usage.include for cache stats

We stream so the user sees tokens as they arrive, and we set
`stream_options.include_usage` so the final chunk carries the cache breakdown
(`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class DeepSeekError(Exception):
    pass


@dataclass
class ChatResult:
    message: dict[str, Any]            # the assembled assistant message
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, timeout: float = 300.0):
        if not api_key:
            raise DeepSeekError("No API key. Run /connect first.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def list_models(self) -> list[str]:
        req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [m.get("id", "?") for m in data.get("data", [])]
        except urllib.error.HTTPError as e:
            raise DeepSeekError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
        except urllib.error.URLError as e:
            raise DeepSeekError(f"Could not reach {self.base_url}: {e.reason}") from e

    def get_balance(self) -> float | None:
        """Return the account balance in USD, or None on failure."""
        req = urllib.request.Request(
            f"{self.base_url}/user/balance", headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            infos = data.get("balance_infos") or []
            for entry in infos:
                if entry.get("currency") == "USD":
                    return float(entry.get("total_balance", 0))
            return float(infos[0]["total_balance"]) if infos else None
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            return None

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.3,
        on_text: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
    ) -> ChatResult:
        """Stream a completion, assembling content + tool_calls from deltas."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "prefix": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        finish_reason: Optional[str] = None

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise DeepSeekError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from e
        except urllib.error.URLError as e:
            raise DeepSeekError(f"Could not reach {self.base_url}: {e.reason}") from e

        with resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                line = line[len("data:"):].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                # reasoning_content shows up on deepseek-reasoner
                reasoning = delta.get("reasoning_content")
                if reasoning and on_reasoning:
                    on_reasoning(reasoning)

                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    if on_text:
                        on_text(text)

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]

        message: dict[str, Any] = {"role": "assistant"}
        message["content"] = "".join(content_parts) or None
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["args"]},
                }
                for _, slot in sorted(tool_calls.items())
            ]
        elif message["content"] is None:
            # An empty stream (e.g. a reasoning-only response) must not produce
            # content=null with no tool_calls: the API rejects that message on
            # every later request ("content or tool_calls must be set"), which
            # would poison the append-only log for good.
            message["content"] = ""
        return ChatResult(message=message, usage=usage, finish_reason=finish_reason)
