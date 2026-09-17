"""LLM bridge / TextGenerator (DESIGN.md §10, pattern B).

Direction constraint (load-bearing): generator output never drives control
flow — it only fills strings. System One decides *what* to do, the generator
drafts the open-ended text, and System One validates the draft (Noul/Score)
before anything executes. See ``Runner``'s generate-then-validate path.
"""
from __future__ import annotations

import json as _json
import os as _os
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class TextGenerator(Protocol):
    """Minimal contract for an open-ended text producer."""
    def generate(self, prompt: str, context: dict[str, Any]) -> str: ...


@dataclass
class MockGenerator:
    """Deterministic template/echo generator for offline tests and demos.

    ``template`` may contain ``{prompt}`` (echoes the rendered prompt);
    otherwise the literal text is returned. Pass ``func`` for scripted
    per-prompt behavior.
    """
    template: str = "mock draft"
    func: Callable[[str, dict[str, Any]], str] | None = None

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        if self.func is not None:
            return self.func(prompt, context or {})
        if "{prompt}" in self.template:
            return self.template.format(prompt=prompt)
        return self.template


@dataclass
class HttpGenerator:
    """OpenAI-compatible ``/chat/completions`` generator over stdlib urllib.

    Covers OpenAI itself plus any OpenAI-standard endpoint (OpenRouter,
    local servers, …) via ``base_url``. Key from ``api_key`` or the
    ``OPENAI_API_KEY`` environment (mirrors the ``TypeSafeClient`` pattern);
    ``extra_headers`` carries provider-specific headers such as OpenRouter's
    ``HTTP-Referer``/``X-Title``. Stays in optional-land: only constructed
    when you opt in.

    OpenRouter example::

        HttpGenerator(base_url="https://openrouter.ai/api/v1",
                      model="openai/gpt-4o-mini")
    """
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    env_var: str = "OPENAI_API_KEY"
    timeout: float = 30.0
    extra_headers: dict[str, str] | None = None

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        import urllib.request

        key = self.api_key or _os.environ.get(self.env_var, "")
        if not key:
            raise RuntimeError(f"{self.env_var} is not set (env or explicit api_key)")
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}]}
        headers = {"Authorization": f"Bearer {key}",
                   "Content-Type": "application/json"}
        headers.update(self.extra_headers or {})
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=_json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected chat-completions response: {exc}") from exc


@dataclass
class AnthropicGenerator:
    """Anthropic Messages API generator over stdlib urllib.

    Key from ``api_key`` or the ``ANTHROPIC_API_KEY`` environment. ``model``
    has no default — pass an explicit model ID (e.g. a ``claude-*`` ID from
    https://docs.anthropic.com/en/docs/about-claude/models). Response text
    blocks are concatenated; non-text blocks are skipped.
    """
    model: str = ""
    base_url: str = "https://api.anthropic.com"
    api_key: str | None = None
    env_var: str = "ANTHROPIC_API_KEY"
    api_version: str = "2023-06-01"
    max_tokens: int = 1024
    timeout: float = 30.0

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        import urllib.request

        if not self.model:
            raise RuntimeError("AnthropicGenerator needs an explicit model ID")
        key = self.api_key or _os.environ.get(self.env_var, "")
        if not key:
            raise RuntimeError(f"{self.env_var} is not set (env or explicit api_key)")
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/v1/messages",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"x-api-key": key,
                     "anthropic-version": self.api_version,
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        try:
            blocks = data["content"]
            texts = [b["text"] for b in blocks
                     if isinstance(b, dict) and b.get("type") == "text" and "text" in b]
        except (KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"unexpected messages response: {exc}") from exc
        if not texts:
            raise RuntimeError(f"no text blocks in messages response: {data!r}"[:300])
        return "\n".join(texts)
