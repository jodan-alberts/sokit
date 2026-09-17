"""SOKIT interactive CLI — demo, play with, and test the agents.

    python3 -m examples.cli list
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
        "build": build_support,
        "mock_client": lambda: MockClient(rules=support_agent.mock_rules()),
        "example_task": "I want a refund for my subscription, please.",
        "example_fields": {"account_id": "acct_123"},
    },
    "incident": {
        "description": "2am pager: checkout 500s — investigate, rollback, verify",
        "build": build_incident,
        "mock_client": lambda: MockClient(rules=incident_agent.MOCK_RULES),
        "example_task": incident_agent.TASK,
        "example_fields": {},
    },
    "draft": {
        "description": "Reply drafting: decide → draft (mock, OpenAI-standard, or Anthropic) → validate → send",
        "build": build_draft,
        "mock_client": lambda: MockClient(rules={
            "next_action": draft_reply.next_action_rule, **DRAFT_OK_RULE}),
        "example_task": "Customer says: 'I want a refund for my subscription, please.'",
        "example_fields": {"account_id": "acct_123"},
    },
}


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


def print_turn(turn: int, evaluation, gate, actions, trace: bool = True) -> None:
    if not trace:
        return
    print(f"\n{style(f'── turn {turn}', bold=True)} {gate_chip(gate.value)}")
    for name, d in evaluation.decisions.items():
        print(fmt_decision(name, d.value, d.confidence))
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


def run_once(agent: str, task: str, fields: dict, gates: tuple[float, float],
             max_turns: int, trace: bool, client=None, mode: str = "mock",
             auto_yes: bool = False, telemetry_out: str | None = None,
             gen_opts: dict | None = None) -> tuple[object, int]:
    """Run one task; return (result, exit_code)."""
    spinner = Spinner(f"running {agent} on {mode}")

    def on_turn(turn, evaluation, gate, actions):
        spinner.__exit__()  # first turn flowing: hand over from spinner to stream
        print_turn(turn, evaluation, gate, actions, trace)

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
    spinner = Spinner(f"running {agent} on {mode}")
    with spinner:
        result = runner.run(task, fields)
    print_events(result, trace)
    print("\n" + outcome_banner(result.outcome))
    if telemetry_out:
        result.telemetry.to_jsonl(telemetry_out)
        print(style(f"telemetry → {telemetry_out}", dim=True))
    code = EXIT_OK if result.outcome == "completed" else EXIT_RUN_FAILED
    return result, code


def cmd_list(_args) -> int:
    rows = [f"  {style(name.ljust(18), fg=CYAN, bold=True)} {info['description']}"
            for name, info in AGENTS.items()]
    print(panel("agents", rows))
    return EXIT_OK


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
                 f"gates: auto={args.auto} escalate={args.escalate}"]))
    _, code = run_once(args.agent, args.task, fields, (args.auto, args.escalate),
                       args.max_turns, trace=not args.quiet, client=client, mode=mode,
                       auto_yes=args.yes, telemetry_out=args.telemetry_out,
                       gen_opts=gen_opts_from(args))
    return code


REPL_HELP = (":fields {...}  set fields  ·  :agent <name>  switch agent  ·  "
             ":gates <auto> <esc>  retune  ·  :trace on|off  ·  :quit")


def cmd_repl(args) -> int:
    try:
        import readline  # noqa: F401 — history for free where available
    except ImportError:
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
    print(panel(f"sokit repl · {agent} · {mode}",
                [AGENTS[agent]["description"],
                 f"gates: auto={gates[0]} escalate={gates[1]}",
                 REPL_HELP]))
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
            else:
                print(f"unknown command ':{cmd}' — {REPL_HELP}")
            continue
        run_once(agent, line, dict(fields), gates, args.max_turns, trace,
                 client=client, mode=mode, auto_yes=args.yes,
                 gen_opts=gen_opts_from(args))
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
    r.add_argument("--telemetry-out", default=None, help="dump telemetry to JSONL")

    repl = sub.add_parser("repl", help="interactive shell")
    _client_flags(repl)
    _generator_flags(repl)
    repl.add_argument("agent", nargs="?", default="incident")
    repl.add_argument("--auto", type=float, default=0.8)
    repl.add_argument("--escalate", type=float, default=0.5)
    repl.add_argument("--max-turns", type=int, default=6)
    repl.add_argument("--yes", action="store_true", help="auto-approve confirm gates")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "repl":
        return cmd_repl(args)
    return EXIT_USAGE  # unreachable (required subcommand)


if __name__ == "__main__":
    sys.exit(main())
