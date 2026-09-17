"""SOKIT interactive CLI — demo, play with, and test the agents.

    python3 -m examples.cli list
    python3 -m examples.cli inspect support --mock
    python3 -m examples.cli run support "I want a refund" --fields '{"account_id": "acct_123"}'
    python3 -m examples.cli repl draft

Client selection: --mock forces the deterministic mock, --real forces the
live TypeSafe API (needs TYPESAFE_API_KEY). Default: live when a key is set,
mock otherwise (with a notice). Stdlib only (argparse + ANSI).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import draft_reply, incident_agent, support_agent
from examples.tui import (
    Spinner,
    fmt_action,
    fmt_decision,
    fmt_distribution,
    fmt_gate_detail,
    fmt_margin,
    fmt_question,
    gate_chip,
    outcome_banner,
    panel,
    style,
    CYAN,
    GREEN,
    MAGENTA,
    RED,
    YELLOW,
)
from harness import AnthropicGenerator, ConfidenceGate, HttpGenerator, MockClient, MockGenerator, TypeSafeClient

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE = 2

CANNED_DRAFT = "Hello! Your refund is on its way. Kind regards, Support"
DRAFT_OK_RULE = {"draft_ok": {"yes": ["kind regards"], "no": []}}


def build_support(client, gen_opts):
    """-> (runner, memory)."""
    return support_agent.make_runner(client)


def build_incident(client, gen_opts):
    """-> (runner, memory)."""
    return incident_agent.make_runner(client)


def build_draft(client, gen_opts):
    """-> runner. Generator is selectable: the canned mock (default, offline),
    any OpenAI-standard endpoint (OpenAI, OpenRouter, local, …), or Anthropic."""
    gen_opts = gen_opts or {}
    kind = gen_opts.get("kind", "mock")
    if kind == "openai":
        generator = HttpGenerator(model=gen_opts.get("model") or "gpt-4o-mini",
                                  base_url=gen_opts.get("base_url") or
                                  "https://api.openai.com/v1")
    elif kind == "anthropic":
        generator = AnthropicGenerator(model=gen_opts.get("model") or "",
                                       base_url=gen_opts.get("base_url") or
                                       "https://api.anthropic.com")
    else:
        generator = MockGenerator(template=CANNED_DRAFT)
    return draft_reply.make_runner(client, generator)


AGENTS = {
    "support": {
        "description": "Support-ticket triage + refunds (SQLite accounts, KB provider)",
        "guide": [
            "Classifies the ticket, looks up the SQLite account DB, refunds if eligible.",
            "Try: 'I want a refund' + {\"account_id\": \"acct_123\"} (pro → refunded),",
            "     'Please close my account' + {\"account_id\": \"acct_999\"} (basic → escalated),",
            "     any task + {\"account_id\": \"acct_xyz\"} (unknown → failure path).",
            "Play: :gates 0.95 0.5 to force confirm prompts on every turn.",
        ],
        "build": build_support,
        "mock_client": lambda: MockClient(rules=support_agent.mock_rules()),
        "example_task": "I want a refund for my subscription, please.",
        "example_fields": {"account_id": "acct_123"},
    },
    "incident": {
        "description": "2am pager: checkout 500s — investigate, rollback, verify",
        "guide": [
            "Investigates (health, deploys), rolls back the suspect deploy, verifies;",
            "pages on-call with evidence when the fix doesn't hold.",
            "Try: the default pager task, then :gates 0.95 0.5 and watch rollback",
            "ask for confirmation first. Each task starts a fresh world.",
        ],
        "build": build_incident,
        "mock_client": lambda: MockClient(rules=incident_agent.MOCK_RULES),
        "example_task": incident_agent.TASK,
        "example_fields": {},
    },
    "draft": {
        "description": "Reply drafting: decide → draft (mock, OpenAI-standard, or Anthropic) → validate → send",
        "guide": [
            "System One routes to draft_reply; the generator fills ONLY args[\"message\"];",
            "a validator Noul gates execution. Generator failure or rejection escalates.",
            "Try: --generator openai|anthropic for real drafts (needs the LLM key).",
            "Break it: ask for something the validator should refuse and watch it escalate.",
        ],
        "build": build_draft,
        "mock_client": lambda: MockClient(rules={
            "next_action": draft_reply.next_action_rule, **DRAFT_OK_RULE}),
        "example_task": "Customer says: 'I want a refund for my subscription, please.'",
        "example_fields": {"account_id": "acct_123"},
    },
}


def model_label(client, mode: str) -> str:
    """Human-readable model identity for headers (mock vs live TypeSafe)."""
    if mode == "mock":
        return "mock (deterministic, offline)"
    model = getattr(client, "model", "jev-latest")
    endpoint = getattr(client, "endpoint", "")
    return f"live TypeSafe · {model} · {endpoint}" if endpoint else f"live TypeSafe · {model}"


def _tool_names(runner) -> list[str]:
    registry = getattr(runner, "tools", None)
    tools = getattr(registry, "_tools", None)
    if isinstance(tools, dict):
        return sorted(tools)
    return []


def agent_card_lines(agent: str, runner=None, mode: str = "mock") -> list[str]:
    """Inspect lines for an agent: questions, tools, gate, policy version."""
    info = AGENTS[agent]
    runner = runner  # built by caller when available; else introspect lazily
    lines = [info["description"], ""]
    if runner is not None:
        policy = runner.policy
        gate = runner.confidence_gate
        lines.append(f"policy v{getattr(policy, 'version', '?')} · "
                     f"model: {model_label(getattr(runner, 'client', None), mode)}")
        scoped = f"scoped to [{', '.join(gate.questions)}]" if gate.questions else "all questions"
        lines.append(f"gate: auto={gate.auto:g} escalate={gate.escalate:g} ({scoped})")
        lines.append(f"tools: {', '.join(_tool_names(runner)) or '—'}")
        lines.append(f"max_turns={runner.max_turns} · "
                     f"providers={len(getattr(runner.state_builder, 'providers', []))}")
        lines.append("")
        lines.append("questions:")
        for name, q in policy.questions.items():
            qtype = getattr(getattr(q, 'type', '?'), 'value', str(getattr(q, 'type', '?')))
            instr = getattr(q, 'instructions', '')
            opts = list(getattr(q, 'options', None) or getattr(q, 'levels', None) or [])
            lines.append(f"  · {name} ({qtype}): {instr}")
            if opts:
                lines.append(f"    options: {', '.join(opts)}")
    return lines


def select_client(agent: str, mock: bool, real: bool) -> tuple[object, str]:
    """Return (client, mode) where mode is 'mock' or 'real'."""
    from harness.client import _read_env_file

    _read_env_file()
    has_key = bool(os.environ.get("TYPESAFE_API_KEY"))
    if real and not has_key:
        raise SystemExit("error: --real needs TYPESAFE_API_KEY (env or .env)")
    if mock or not has_key:
        if not mock and not has_key:
            print(style("notice: no TYPESAFE_API_KEY — using deterministic mock",
                        dim=True), file=sys.stderr)
        return AGENTS[agent]["mock_client"](), "mock"
    return TypeSafeClient(), "real"


def gen_opts_from(args) -> dict:
    return {"kind": args.generator, "model": args.generator_model,
            "base_url": args.base_url}


GENERATOR_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def check_generator(args) -> int | None:
    """Fail fast on missing keys/models (else they would surface mid-run as
    generator-failure escalations). None means OK."""
    kind = args.generator
    if kind == "mock":
        return None
    env_var = GENERATOR_KEYS[kind]
    if not os.environ.get(env_var):
        print(f"error: --generator {kind} needs {env_var} "
              "(env or .env; never passed as a flag)", file=sys.stderr)
        return EXIT_USAGE
    if kind == "anthropic" and not args.generator_model:
        print("error: --generator anthropic needs --generator-model "
              "(e.g. a claude-* model ID)", file=sys.stderr)
        return EXIT_USAGE
    return None


def build_runner(agent: str, client, gates: tuple[float, float], max_turns: int,
                 on_turn=None, on_confirm=None, gen_opts: dict | None = None):
    """Build a fresh runner (fresh per task: no idempotency/memory bleed)."""
    runner_or_pair = AGENTS[agent]["build"](client, gen_opts)
    runner = runner_or_pair[0] if isinstance(runner_or_pair, tuple) else runner_or_pair
    auto, escalate = gates
    prev_questions = runner.confidence_gate.questions
    runner.confidence_gate = ConfidenceGate(auto=auto, escalate=escalate,
                                            questions=prev_questions)
    runner.max_turns = max_turns
    runner.on_turn = on_turn
    runner.on_confirm = on_confirm
    return runner


def print_guide(agent: str) -> None:
    info = AGENTS[agent]
    print(panel(f"{agent} — guide", [info["description"], "", *info["guide"]]))


def print_turn(turn: int, evaluation, gate, actions, trace: bool = True,
               questions: dict | None = None, gate_obj=None,
               verbose: bool = False) -> None:
    """Render one turn: decision + full distribution + gate reasoning.

    ``trace=False`` stays silent (for --quiet). The default trace shows the
    top-3 probability mass per question so the runner-up is visible; ``verbose``
    adds question definitions (type + instructions + options).
    """
    if not trace:
        return
    print(f"\n{style(f'── turn {turn}', bold=True)} {gate_chip(gate.value)}")
    for name, d in evaluation.decisions.items():
        if verbose and questions is not None and name in questions:
            print(fmt_question(name, questions[name]))
        print(fmt_decision(name, d.value, d.confidence), end="")
        margin = fmt_margin(getattr(d, "probabilities", None))
        print(f"  {margin}" if margin else "")
        dist = fmt_distribution(getattr(d, "probabilities", None))
        if dist:
            print(dist)
    if gate_obj is not None:
        scoped = getattr(gate_obj, "questions", None)
        confs = {n: dd.confidence for n, dd in evaluation.decisions.items()
                 if not scoped or n in scoped}
        print(fmt_gate_detail(gate.value, confs, auto=gate_obj.auto,
                              escalate=gate_obj.escalate,
                              gated_questions=gate_obj.questions))
    for action in actions:
        print(fmt_action(action))


def print_events(result, trace: bool = True) -> None:
    if not trace or not result.state.events:
        return
    print(style("\n── tool events", bold=True))
    for e in result.state.events:
        color = GREEN if e.content.startswith("OK") else (
            YELLOW if e.content.startswith("ERROR: skipped") else RED)
        print(f"  {style(f't{e.turn}', dim=True)} "
              f"{style(f'[{e.kind}:{e.source}]', fg=MAGENTA)} "
              f"{style(e.content, fg=color)}")


def print_summary(agent: str, result, runner, mode: str, trace: bool = True) -> None:
    """End-of-run model summary: outcome, turns, tool calls, final read."""
    if not trace:
        return
    records = result.telemetry.records
    tool_calls = sum(1 for e in result.state.events if e.kind == "tool")
    tool_errors = sum(1 for e in result.state.events
                      if e.kind == "tool" and not e.content.startswith("OK"))
    drafts = sum(1 for e in result.state.events if e.kind == "draft")
    lines = [
        f"model: {model_label(runner.client, mode)}",
        f"policy v{getattr(runner.policy, 'version', '?')} · "
        f"turns {len(records)}/{runner.max_turns} · "
        f"tools {tool_calls} ({tool_errors} errors)"
        + (f" · drafts {drafts}" if drafts else ""),
    ]
    final = result.evaluation
    if final is not None:
        for name, d in final.decisions.items():
            probs = getattr(d, "probabilities", None) or {}
            if probs:
                top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:2]
                alt = f" (runner-up: {top[1][0]} {top[1][1]:.2f})" if len(top) > 1 else ""
            else:
                alt = ""
            lines.append(f"  final {name} = {d.value} (conf {d.confidence:.2f}){alt}")
    if result.state.events:
        last = result.state.events[-1]
        lines.append(f"  last event t{last.turn} [{last.kind}:{last.source}] {last.content[:120]}")
    print(panel(f"{agent} · summary", lines))


def run_once(agent: str, task: str, fields: dict, gates: tuple[float, float],
             max_turns: int, trace: bool, client=None, mode: str = "mock",
             auto_yes: bool = False, telemetry_out: str | None = None,
             gen_opts: dict | None = None, verbose: bool = False) -> tuple[object, int]:
    """Run one task; return (result, exit_code)."""
    spinner = Spinner(f"running {agent} on {mode}")

    runner_box: dict = {}

    def on_turn(turn, evaluation, gate, actions):
        spinner.__exit__()  # first turn flowing: hand over from spinner to stream
        r = runner_box.get("runner")
        print_turn(turn, evaluation, gate, actions, trace,
                   questions=getattr(getattr(r, "policy", None), "questions", None),
                   gate_obj=getattr(r, "confidence_gate", None),
                   verbose=verbose)

    def on_confirm(evaluation, state):
        if auto_yes:
            return True
        try:
            answer = input(style("  [confirm] proceed? [y/N] ", fg=YELLOW, bold=True))
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes")

    runner = build_runner(agent, client, gates, max_turns,
                          on_turn=on_turn, on_confirm=on_confirm,
                          gen_opts=gen_opts)
    runner_box["runner"] = runner
    if verbose:
        try:
            state_preview = runner.state_builder.assemble(
                __import__("harness").State(task=task, fields=fields))[:600]
            print(panel("model input · state preview (truncated)",
                        [state_preview.replace("\n", " ⏎ ")]))
        except Exception:  # noqa: BLE001 — preview is best-effort
            pass
    spinner = Spinner(f"running {agent} on {mode}")
    with spinner:
        result = runner.run(task, fields)
    print_events(result, trace)
    print("\n" + outcome_banner(result.outcome))
    print_summary(agent, result, runner, mode, trace)
    if telemetry_out:
        result.telemetry.to_jsonl(telemetry_out)
        print(style(f"telemetry → {telemetry_out}", dim=True))
    code = EXIT_OK if result.outcome == "completed" else EXIT_RUN_FAILED
    return result, code


def cmd_list(_args) -> int:
    rows = []
    for name, info in AGENTS.items():
        try:
            runner_or_pair = info["build"](info["mock_client"](), {})
            runner = runner_or_pair[0] if isinstance(runner_or_pair, tuple) else runner_or_pair
            n_q = len(runner.policy.questions)
            tools = ", ".join(_tool_names(runner)) or "—"
            rows.append(f"  {style(name.ljust(18), fg=CYAN, bold=True)} {info['description']}")
            rows.append(style(f"    {'':<18} {n_q} questions · tools: {tools}", dim=True))
        except Exception:  # noqa: BLE001 — list must work even if a build fails
            rows.append(f"  {style(name.ljust(18), fg=CYAN, bold=True)} {info['description']}")
    rows.append("")
    rows.append(style("  tip: `inspect <agent>` shows its questions, tools, and gate", dim=True))
    print(panel("agents", rows))
    return EXIT_OK


def cmd_inspect(args) -> int:
    """Show the full decision space of one agent (understand the model)."""
    agent = args.agent
    if agent not in AGENTS:
        print(f"error: unknown agent '{agent}' (see `list`)", file=sys.stderr)
        return EXIT_USAGE
    try:
        client, mode = select_client(agent, args.mock, args.real)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE
    gen_opts = gen_opts_from(args) if hasattr(args, "generator") else {}
    runner_or_pair = AGENTS[agent]["build"](client, gen_opts)
    runner = runner_or_pair[0] if isinstance(runner_or_pair, tuple) else runner_or_pair
    print(panel(f"{agent} · inspect", agent_card_lines(agent, runner, mode)))
    print_guide(agent)
    return EXIT_OK


def cmd_doctor(_args) -> int:
    """Environment diagnostics: keys, stdlib deps, endpoint reachability."""
    import urllib.request

    rows = []

    def row(label: str, ok: bool, detail: str = ""):
        mark = style("ok  ", fg=GREEN, bold=True) if ok else style("MISS", fg=RED, bold=True)
        rows.append(f"  {mark} {label.ljust(22)} {detail}")

    row("python", sys.version_info >= (3, 10), sys.version.split()[0])
    from harness.client import _read_env_file
    _read_env_file()
    row("TYPESAFE_API_KEY", bool(os.environ.get("TYPESAFE_API_KEY")),
        "live Jev decisions" if os.environ.get("TYPESAFE_API_KEY") else "mock mode only")
    row("OPENAI_API_KEY", bool(os.environ.get("OPENAI_API_KEY")),
        "for --generator openai" if os.environ.get("OPENAI_API_KEY") else "mock drafts only")
    row("ANTHROPIC_API_KEY", bool(os.environ.get("ANTHROPIC_API_KEY")),
        "for --generator anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock drafts only")
    try:
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a TEXT)")
        conn.close()
        row("sqlite3", True, "SqlProvider ready")
    except Exception as exc:  # noqa: BLE001
        row("sqlite3", False, str(exc)[:60])
    row("readline", _readline is not None,
        "history + tab-completion" if _readline is not None else "plain input() fallback")
    try:
        req = urllib.request.Request("https://api.typesafe.ai/v1/systemone",
                                     data=b"{}", method="GET")
        try:
            urllib.request.urlopen(req, timeout=4)
            row("typesafe endpoint", True, "reachable")
        except Exception as exc:  # noqa: BLE001 — any HTTP reply proves connectivity
            import urllib.error
            if isinstance(exc, urllib.error.HTTPError):
                row("typesafe endpoint", True, f"reachable (HTTP {exc.code})")
            else:
                row("typesafe endpoint", False, f"{type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001
        row("typesafe endpoint", False, str(exc)[:60])
    print(panel("doctor", rows))
    return EXIT_OK


def cmd_eval(args) -> int:
    """Run the labeled-eval regression gate (offline, mock-backed)."""
    from examples import eval_demo

    return eval_demo.main(cases_path=args.cases, error_budget=args.error_budget)


def cmd_show(args) -> int:
    """Re-render a saved telemetry JSONL file (--telemetry-out output)."""
    try:
        with open(args.path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {args.path}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not records:
        print(f"error: no records in {args.path}", file=sys.stderr)
        return EXIT_USAGE
    for rec in records:
        turn, gate = rec.get("turn"), rec.get("gate", "?")
        print(f"\n{style(f'── turn {turn}', bold=True)} {gate_chip(str(gate))}", end="")
        outcome = rec.get("outcome")
        if outcome:
            print(f"  {style(f'outcome={outcome}', dim=True)}")
        else:
            print()
        for name, d in rec.get("decisions", {}).items():
            print(fmt_decision(name, d.get("value"), float(d.get("confidence", 0.0))), end="")
            margin = fmt_margin(d.get("probabilities"))
            print(f"  {margin}" if margin else "")
            dist = fmt_distribution(d.get("probabilities"))
            if dist:
                print(dist)
        for a in rec.get("actions", []):
            tool = a.get("tool")
            if tool:
                print(f"  → {style(a.get('name', '?'), fg=MAGENTA)}  "
                      f"tool={tool} args={a.get('args', {})}")
            else:
                print(f"  → {style(a.get('name', '?'), fg=MAGENTA)}  (terminal)")
        note = rec.get("note")
        if note:
            print(f"  {style(f'note: {note}', dim=True)}")
    return EXIT_OK


class Transcript:
    """Write-through tee of stdout into a file (full session capture)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")
        self._real = sys.stdout

    def __getattr__(self, name: str):
        # isatty/encoding/fileno/… delegate so color detection etc. keeps working.
        return getattr(self.__dict__["_real"], name)

    def write(self, text: str):
        self._real.write(text)
        self._fh.write(text)
        return len(text)

    def flush(self):
        self._real.flush()
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        finally:
            if sys.stdout is self:
                sys.stdout = self._real


from contextlib import contextmanager


@contextmanager
def transcript(path: str | None):
    """Capture the full rendered session (prompts included) to a file."""
    if not path:
        yield
        return
    tee = Transcript(path)
    prev = sys.stdout
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = prev
        tee.close()
        print(style(f"transcript → {path}", dim=True))


def cmd_run(args) -> int:
    if args.agent not in AGENTS:
        print(f"error: unknown agent '{args.agent}' (see `list`)", file=sys.stderr)
        return EXIT_USAGE
    try:
        fields = json.loads(args.fields) if args.fields else {}
        if not isinstance(fields, dict):
            raise ValueError("fields must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: --fields: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        client, mode = select_client(args.agent, args.mock, args.real)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE
    if (code := check_generator(args)) is not None:
        return code
    print(panel(f"{args.agent} · {mode}",
                [f"task: {args.task}", f"fields: {json.dumps(fields)}",
                 f"model: {model_label(client, mode)}",
                 f"gates: auto={args.auto} escalate={args.escalate}"]))
    print_guide(args.agent)
    _, code = run_once(args.agent, args.task, fields, (args.auto, args.escalate),
                       args.max_turns, trace=not args.quiet, client=client, mode=mode,
                       auto_yes=args.yes, telemetry_out=args.telemetry_out,
                       gen_opts=gen_opts_from(args), verbose=args.verbose)
    return code


REPL_HELP = (":fields {...}  set fields  ·  :agent <name>  switch agent  ·  "
             ":gates <auto> <esc>  retune  ·  :trace on|off  ·  :guide  ·  "
             ":inspect [agent]  show decision space  ·  :questions  list questions  ·  "
             ":verbose on|off  ·  "
             ":retry  rerun last task  ·  :task  multi-line task  ·  :quit")


REPL_COMMANDS = ("fields", "agent", "gates", "trace", "guide", "inspect", "questions",
                 "verbose", "retry", "task", "help", "quit", "q", "exit")

try:
    import readline as _readline
except ImportError:
    _readline = None


def _repl_completer(text: str, state: int) -> str | None:
    """Tab-complete :commands and agent names after :agent."""
    buf = _readline.get_line_buffer() if _readline is not None else ""
    if buf.startswith(":agent "):
        options = [a for a in AGENTS if a.startswith(text)]
    elif buf.startswith(":") and " " not in buf:
        options = [":" + c for c in REPL_COMMANDS if (":" + c).startswith(text)]
    else:
        return None
    return options[state] if state < len(options) else None


def _history_path() -> str:
    return os.path.expanduser(os.environ.get("SOKIT_HISTORY", "~/.sokit_history"))


def cmd_repl(args) -> int:
    if _readline is not None:
        _readline.set_completer(_repl_completer)
        _readline.parse_and_bind("tab: complete")
        try:
            _readline.read_history_file(_history_path())
        except OSError:
            pass
    agent = args.agent or "incident"
    if agent not in AGENTS:
        print(f"error: unknown agent '{agent}' (see `list`)", file=sys.stderr)
        return EXIT_USAGE
    try:
        client, mode = select_client(agent, args.mock, args.real)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE
    if (code := check_generator(args)) is not None:
        return code
    gates: tuple[float, float] = (args.auto, args.escalate)
    fields: dict = dict(AGENTS[agent]["example_fields"])
    trace = True
    verbose = bool(getattr(args, "verbose", False))
    print(panel(f"sokit repl · {agent} · {mode}",
                [AGENTS[agent]["description"],
                 f"model: {model_label(client, mode)}",
                 f"gates: auto={gates[0]} escalate={gates[1]}",
                 REPL_HELP]))
    print_guide(agent)
    last: tuple[str, dict] | None = None

    def do_run(task_text: str) -> None:
        nonlocal last
        last = (task_text, dict(fields))
        run_once(agent, task_text, dict(fields), gates, args.max_turns, trace,
                 client=client, mode=mode, auto_yes=args.yes,
                 gen_opts=gen_opts_from(args), verbose=verbose)

    def read_multiline() -> str | None:
        print(style("(multi-line task — end with a single . on its own line)", dim=True))
        lines = []
        while True:
            try:
                chunk = input("... ")
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            if chunk.strip() == ".":
                return "\n".join(lines).strip()
            lines.append(chunk)

    try:
        while True:
            try:
                prompt = (f"{style(agent, fg=CYAN, bold=True)} "
                          f"{style(f'⚡{gates[0]:g}/{gates[1]:g}', dim=True)}> ")
                line = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return EXIT_OK
            if not line:
                continue
            if line.startswith(":"):
                cmd, _, rest = line[1:].partition(" ")
                if cmd in ("quit", "q", "exit"):
                    return EXIT_OK
                if cmd == "help":
                    print(REPL_HELP)
                elif cmd == "guide":
                    print_guide(agent)
                elif cmd == "retry":
                    if last is None:
                        print("nothing to retry yet")
                    else:
                        task_text, _ = last
                        do_run(task_text)
                elif cmd == "task":
                    task_text = read_multiline()
                    if task_text:
                        do_run(task_text)
                elif cmd == "agent":
                    if rest.strip() in AGENTS:
                        agent = rest.strip()
                        fields = dict(AGENTS[agent]["example_fields"])
                        try:
                            client, mode = select_client(agent, args.mock, args.real)
                        except SystemExit as exc:
                            print(exc, file=sys.stderr)
                            return EXIT_USAGE
                        print(style(f"switched to {agent} · {mode}", fg=GREEN))
                        print_guide(agent)
                    else:
                        print(f"unknown agent '{rest.strip()}' — {', '.join(AGENTS)}")
                elif cmd == "fields":
                    try:
                        parsed = json.loads(rest) if rest.strip() else {}
                        if not isinstance(parsed, dict):
                            raise ValueError("must be a JSON object")
                        fields = parsed
                        print(style(f"fields = {json.dumps(fields)}", dim=True))
                    except (json.JSONDecodeError, ValueError) as exc:
                        print(f"error: {exc}")
                elif cmd == "gates":
                    try:
                        auto_s, esc_s = rest.split()
                        gates = (float(auto_s), float(esc_s))
                        ConfidenceGate(auto=gates[0], escalate=gates[1])  # validate
                        print(style(f"gates: auto={gates[0]} escalate={gates[1]}", dim=True))
                    except (ValueError, TypeError) as exc:
                        print(f"error: :gates <auto> <esc> — {exc}")
                elif cmd == "trace":
                    trace = rest.strip().lower() not in ("off", "0", "no")
                    print(style(f"trace {'on' if trace else 'off'}", dim=True))
                elif cmd == "verbose":
                    arg = rest.strip().lower()
                    if not arg:
                        verbose = not verbose
                    else:
                        verbose = arg not in ("off", "0", "no")
                    print(style(f"verbose {'on' if verbose else 'off'}"
                                " (question definitions + state preview)", dim=True))
                elif cmd == "inspect":
                    target = rest.strip() or agent
                    if target not in AGENTS:
                        print(f"unknown agent '{target}' — {', '.join(AGENTS)}")
                    else:
                        runner_or_pair = AGENTS[target]["build"](client, gen_opts_from(args))
                        r = runner_or_pair[0] if isinstance(runner_or_pair, tuple) else runner_or_pair
                        print(panel(f"{target} · inspect",
                                    agent_card_lines(target, r, mode)))
                elif cmd == "questions":
                    runner_or_pair = AGENTS[agent]["build"](client, gen_opts_from(args))
                    r = runner_or_pair[0] if isinstance(runner_or_pair, tuple) else runner_or_pair
                    for name, q in r.policy.questions.items():
                        print(fmt_question(name, q))
                        opts = list(getattr(q, "options", None) or getattr(q, "levels", None) or [])
                        if opts:
                            print(style(f"      options: {', '.join(opts)}", dim=True))
                else:
                    print(f"unknown command ':{cmd}' — {REPL_HELP}")
                continue
            do_run(line)
    finally:
        if _readline is not None:
            try:
                _readline.write_history_file(_history_path())
            except OSError:
                pass
    return EXIT_OK


def _client_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mock", action="store_true", help="force the deterministic mock")
    p.add_argument("--real", action="store_true", help="force the live TypeSafe API")


def _generator_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--generator", choices=["mock", "openai", "anthropic"], default="mock",
                   help="draft source for the draft agent (openai = any OpenAI-standard "
                        "endpoint: OpenAI, OpenRouter, local, …)")
    p.add_argument("--generator-model", default=None,
                   help="LLM model ID (default: gpt-4o-mini for openai; required for anthropic)")
    p.add_argument("--base-url", default=None,
                   help="override endpoint (e.g. https://openrouter.ai/api/v1, http://localhost:11434/v1)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli", description="SOKIT interactive CLI")
    _client_flags(p)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available agents")
    sub.add_parser("doctor", help="environment diagnostics (keys, deps, endpoints)")

    r = sub.add_parser("run", help="run one task one-shot")
    _client_flags(r)
    _generator_flags(r)
    r.add_argument("agent", help="agent name (see list)")
    r.add_argument("task", help="task text")
    r.add_argument("--fields", default="", help="JSON object of structured fields")
    r.add_argument("--auto", type=float, default=0.8)
    r.add_argument("--escalate", type=float, default=0.5)
    r.add_argument("--max-turns", type=int, default=6)
    r.add_argument("--quiet", action="store_true", help="outcome banner only")
    r.add_argument("--yes", action="store_true", help="auto-approve confirm gates")
    r.add_argument("--verbose", action="store_true",
                   help="show question definitions + state preview + full distributions")
    r.add_argument("--telemetry-out", default=None, help="dump telemetry to JSONL")
    r.add_argument("--transcript-out", default=None, help="capture full session output to a file")

    repl = sub.add_parser("repl", help="interactive shell")
    _client_flags(repl)
    _generator_flags(repl)
    repl.add_argument("agent", nargs="?", default="incident")
    repl.add_argument("--auto", type=float, default=0.8)
    repl.add_argument("--escalate", type=float, default=0.5)
    repl.add_argument("--max-turns", type=int, default=6)
    repl.add_argument("--yes", action="store_true", help="auto-approve confirm gates")
    repl.add_argument("--verbose", action="store_true",
                      help="show question definitions + state preview + full distributions")
    repl.add_argument("--transcript-out", default=None, help="capture full session output to a file")

    insp = sub.add_parser("inspect", help="show an agent's decision space (questions, tools, gate)")
    _client_flags(insp)
    _generator_flags(insp)
    insp.add_argument("agent", nargs="?", default="support",
                      help="agent name (see list)")

    e = sub.add_parser("eval", help="labeled-eval regression gate (offline)")
    e.add_argument("--cases", default=None, help="JSONL cases file (default: bundled eval_cases)")
    e.add_argument("--error-budget", type=float, default=0.15, help="tuner error budget")
    e.add_argument("--transcript-out", default=None, help="capture full session output to a file")

    s = sub.add_parser("show", help="re-render saved telemetry JSONL")
    s.add_argument("path", help="telemetry file from --telemetry-out")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with transcript(getattr(args, "transcript_out", None)):
        if args.command == "list":
            return cmd_list(args)
        if args.command == "inspect":
            return cmd_inspect(args)
        if args.command == "run":
            return cmd_run(args)
        if args.command == "repl":
            return cmd_repl(args)
        if args.command == "doctor":
            return cmd_doctor(args)
        if args.command == "eval":
            return cmd_eval(args)
        if args.command == "show":
            return cmd_show(args)
    return EXIT_USAGE  # unreachable (required subcommand)


if __name__ == "__main__":
    sys.exit(main())
