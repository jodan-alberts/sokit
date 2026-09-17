"""Tool execution with argument templating and idempotency.

Hardening (§9): ``execute()`` never lets a tool exception or hang kill the
loop. Timeouts are enforced with ``concurrent.futures`` (stdlib), infra
failures (timeout / unexpected exception) are retried with exponential
backoff, and both degrade to ``ok=False`` ToolResults the model can react
to next turn. Only ``ok=True`` results enter the idempotency cache — a
failed call may not have taken effect, so failures stay re-runnable.
``ok=False`` results returned by the tool itself are model-visible outcomes
(not infra failures) and are never retried.
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .decisions import Evaluation
from .policy import Action
from .state import State


@dataclass
class ToolResult:
    tool: str
    ok: bool
    output: str
    data: Any = None  # optional structured payload (e.g. for long-term memory)


class Tool(Protocol):
    name: str
    description: str

    def run(self, args: dict[str, Any], context: dict[str, Any]) -> ToolResult: ...


@dataclass
class FunctionTool:
    name: str
    fn: Callable[[dict[str, Any], dict[str, Any]], ToolResult]
    description: str = ""
    timeout: float | None = None   # per-tool override; None -> registry default
    retries: int | None = None     # per-tool override; None -> registry default

    def run(self, args: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        return self.fn(args, context)


def render(template: Any, evaluation: Evaluation, state: State) -> Any:
    """Substitute ``{{decision.<q>}}``, ``{{fields.<key>}}``, ``{{task}}`` in string args."""
    if not isinstance(template, str):
        return template

    def repl(match: re.Match) -> str:
        key = match.group(1).strip()
        if key == "task":
            return state.task
        if key.startswith("fields."):
            return str(state.fields.get(key.split(".", 1)[1], ""))
        if key.startswith("decision."):
            d = evaluation.get(key.split(".", 1)[1])
            return d.label if d else ""
        return match.group(0)

    return re.sub(r"\{\{([^{}]+)\}\}", repl, template)


class ToolRegistry:
    def __init__(self, default_timeout: float | None = 30.0, default_retries: int = 0) -> None:
        self._tools: dict[str, Tool] = {}
        self._executed: dict[tuple[str, str], ToolResult] = {}
        self.default_timeout = default_timeout
        self.default_retries = default_retries

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def has(self, name: str) -> bool:
        return name in self._tools

    def execute(self, action: Action, evaluation: Evaluation, state: State) -> ToolResult:
        if action.tool is None or not self.has(action.tool):
            return ToolResult(action.tool or "none", False, f"unknown tool: {action.tool}")

        args = {k: render(v, evaluation, state) for k, v in action.args.items()}
        key = (action.tool, json.dumps(args, sort_keys=True, default=str))

        # Idempotency guard: never re-run a side-effecting tool with the same
        # args (a stuck loop must not re-execute side effects).
        if key in self._executed and not action.allow_repeat:
            prev = self._executed[key]
            return ToolResult(action.tool, False, f"skipped (already ran): {prev.output}")

        result = self._run_guarded(action.tool, args, {"evaluation": evaluation, "state": state})
        # Only successes enter the idempotency cache: a failed call may not
        # have taken effect, so failures stay re-runnable.
        if result.ok:
            self._executed[key] = result
        return result

    def _run_guarded(self, name: str, args: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        tool = self._tools[name]
        timeout = getattr(tool, "timeout", None)
        if timeout is None:
            timeout = self.default_timeout
        retries = getattr(tool, "retries", None)
        if retries is None:
            retries = self.default_retries
        retries = max(0, int(retries))

        last_exc: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                return self._call_once(tool, args, context, timeout)
            except concurrent.futures.TimeoutError as exc:
                last_exc = exc
                if attempt >= retries:
                    return ToolResult(name, False, f"timeout after {timeout}s")
            except Exception as exc:  # noqa: BLE001 — tools are untrusted; degrade, don't crash
                last_exc = exc
                if attempt >= retries:
                    return ToolResult(name, False, f"ERROR: {exc}")
            if attempt < retries:
                time.sleep(0.05 * (2 ** attempt))
        # Unreachable, but keep a deterministic fallback.
        return ToolResult(name, False, f"ERROR: {last_exc}")

    @staticmethod
    def _call_once(tool: Tool, args: dict[str, Any], context: dict[str, Any],
                   timeout: float | None) -> ToolResult:
        if timeout is None:
            return tool.run(args, context)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(tool.run, args, context)
            # Cancel the future on timeout where possible; the worker thread
            # itself cannot be killed (stdlib limitation), but the loop moves on.
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise
