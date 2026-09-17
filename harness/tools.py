"""Tool execution with argument templating and idempotency."""
from __future__ import annotations

import json
import re
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
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._executed: dict[tuple[str, str], ToolResult] = {}

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

        result = self._tools[action.tool].run(args, {"evaluation": evaluation, "state": state})
        self._executed[key] = result
        return result
