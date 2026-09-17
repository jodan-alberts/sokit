"""Minimal ANSI styling for the SOKIT CLI (stdlib only, no dependencies).

Colors auto-disable when stdout is not a TTY or ``NO_COLOR`` is set, so piped
output and tests stay clean ASCII. Pass ``color=True`` explicitly to force
escapes (used by tests).
"""
from __future__ import annotations

import os
import sys
import threading
import time

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

BLACK = "\x1b[30m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
WHITE = "\x1b[37m"

GATE_COLORS = {"act": GREEN, "confirm": YELLOW, "escalate": RED}
OUTCOME_COLORS = {"completed": GREEN, "escalated": YELLOW,
                  "budget_exhausted": RED, "no_progress": RED}


def color_enabled(force: bool | None = None) -> bool:
    """True when escapes should be emitted."""
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def style(text: str, fg: str = "", bold: bool = False, dim: bool = False,
          force: bool | None = None) -> str:
    """Wrap text in ANSI codes (no-op when color is disabled)."""
    if not color_enabled(force):
        return text
    codes = ""
    if bold:
        codes += BOLD
    if dim:
        codes += DIM
    codes += fg
    return f"{codes}{text}{RESET}" if codes else text


def gate_chip(gate: str, force: bool | None = None) -> str:
    """Colored gate label, e.g. ``[act]`` in green."""
    color = GATE_COLORS.get(gate, WHITE)
    return style(f"[{gate}]", fg=color, bold=True, force=force)


def outcome_banner(outcome: str, force: bool | None = None) -> str:
    """Colored outcome headline."""
    color = OUTCOME_COLORS.get(outcome, WHITE)
    return style(f"● outcome: {outcome}", fg=color, bold=True, force=force)


def conf_bar(confidence: float, width: int = 10, force: bool | None = None) -> str:
    """Unicode confidence bar: ``██████░░░░ 0.62``."""
    clamped = max(0.0, min(1.0, confidence))
    filled = round(clamped * width)
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if clamped >= 0.8 else (YELLOW if clamped >= 0.5 else RED)
    return f"{style(bar, fg=color, force=force)} {clamped:.2f}"


def panel(title: str, lines: list[str], force: bool | None = None) -> str:
    """Rounded box panel. Content lines must already be styled (width is
    measured on visible characters, ignoring escape codes)."""
    import re as _re

    visible = lambda s: _re.sub(r"\x1b\[[0-9;]*m", "", s)  # noqa: E731
    inner = [title] + lines
    width = max((len(visible(s)) for s in inner), default=0)
    top = "╭─" + "─" * (width + 1) + "╮"
    bottom = "╰─" + "─" * (width + 1) + "╯"
    out = [style(top, fg=CYAN, dim=True, force=force)]
    for s in inner:
        pad = " " * (width - len(visible(s)))
        out.append(style("│ ", fg=CYAN, dim=True, force=force) + s + pad
                   + style(" │", fg=CYAN, dim=True, force=force))
    out.append(style(bottom, fg=CYAN, dim=True, force=force))
    return "\n".join(out)


def fmt_decision(name: str, value: object, confidence: float,
                 force: bool | None = None) -> str:
    name_s = style(f"{name:>14}", bold=True, force=force)
    return f"  {name_s}  {value!s:<16} {conf_bar(confidence, force=force)}"


def fmt_distribution(probs: dict | None, top: int = 3,
                     force: bool | None = None) -> str:
    """Compact sorted distribution, e.g. ``refund 0.90 · done 0.03 · …``.

    Shows the top-N options by probability (descending) so the runner-up
    — how close the model was to choosing differently — is visible.
    Returns "" when there is nothing to show.
    """
    if not probs:
        return ""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top]
    parts = [f"{k} {float(v):.2f}" for k, v in ranked]
    return style("    ↳ " + " · ".join(parts), dim=True, force=force)


def fmt_margin(probs: dict | None, force: bool | None = None) -> str:
    """Margin between the top-2 options (Choice/Noul decisiveness signal)."""
    if not probs or len(probs) < 2:
        return ""
    ranked = sorted(probs.values(), reverse=True)
    return style(f"margin +{ranked[0] - ranked[1]:.2f}", dim=True, force=force)


def fmt_question(name: str, question: object, force: bool | None = None) -> str:
    """One-line question definition: type, instructions, and option set."""
    qtype = str(getattr(question, "type", "?"))
    try:
        qtype = question.type.value  # QuestionType enum -> "choice"/"score"/"noul"
    except Exception:  # noqa: BLE001
        pass
    instructions = getattr(question, "instructions", "") or ""
    options = list(getattr(question, "options", None) or [])
    levels = list(getattr(question, "levels", None) or [])
    choices = options or levels
    suffix = f" [{', '.join(choices)}]" if choices else ""
    return style(f"    ? ({qtype}) {instructions}{suffix}", dim=True, force=force)


def fmt_gate_detail(gate: str, confidences: dict[str, float] | None,
                    auto: float | None = None, escalate: float | None = None,
                    gated_questions: list[str] | None = None,
                    force: bool | None = None) -> str:
    """Why the gate fired: min-confidence driver + thresholds + scope."""
    if not confidences:
        return ""
    driver = min(confidences, key=lambda k: confidences[k])
    detail = f"gate {gate} · min conf {confidences[driver]:.2f} on '{driver}'"
    if auto is not None and escalate is not None:
        detail += f" (auto≥{auto:g} esc<{escalate:g})"
    if gated_questions:
        detail += f" scoped to [{', '.join(gated_questions)}]"
    return style("    " + detail, dim=True, force=force)


def fmt_action(action: object, force: bool | None = None) -> str:
    tool = getattr(action, "tool", None)
    name = style(getattr(action, 'name', '?'), fg=MAGENTA, force=force)
    if tool:
        args = getattr(action, "args", {})
        return f"  → {name}  tool={tool} args={args}"
    flag = "escalate" if getattr(action, "escalate", False) else "terminal"
    return f"  → {name}  ({flag})"


class Spinner:
    """Animated `◐ thinking…` indicator while `runner.run()` blocks.

    Thread-based (stdlib `threading`); renders nothing when color/TTY is off.
    """

    FRAMES = "◐◓◑◒"

    def __init__(self, label: str = "thinking", enabled: bool | None = None) -> None:
        self.label = label
        self.enabled = color_enabled() if enabled is None else enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Spinner":
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        sys.stdout.write("\r\x1b[K")
        sys.stdout.flush()

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r{style(frame + ' ' + self.label + '…', fg=CYAN, dim=True)}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)
