"""System One Harness — a decision-driven agent loop for System One models.

A System One model (TypeSafe's Jev-class) makes fast, calibrated, typed
decisions but cannot generate text, call tools, or fetch data. This package is
the *body* that adds iteration, tool-use, and datasource access around it.
"""
from .calibration import expected_calibration_error, reliability_table
from .client import MockClient, SystemOneClient, TypeSafeClient
from .confidence import ConfidenceGate, Gate
from .context import Document, StateBuilder
from .decisions import (
    Decision,
    Evaluation,
    Question,
    QuestionType,
    choice,
    noul,
    score,
)
from .memory import InMemoryStore, LongTermMemory
from .policy import Action, Policy, choice_route, noul_route
from .providers import (
    ClockProvider,
    FilesProvider,
    FunctionProvider,
    HttpProvider,
    MemoryProvider,
    SqlProvider,
    WebSearchProvider,
)
from .runner import RunResult, Runner
from .state import Event, State
from .telemetry import Telemetry, TurnRecord
from .tools import FunctionTool, ToolRegistry, ToolResult

__all__ = [
    "Action",
    "ClockProvider",
    "ConfidenceGate",
    "Decision",
    "Document",
    "Evaluation",
    "Event",
    "FilesProvider",
    "FunctionProvider",
    "FunctionTool",
    "Gate",
    "HttpProvider",
    "InMemoryStore",
    "LongTermMemory",
    "MemoryProvider",
    "MockClient",
    "Policy",
    "Question",
    "QuestionType",
    "RunResult",
    "Runner",
    "SqlProvider",
    "State",
    "StateBuilder",
    "SystemOneClient",
    "Telemetry",
    "ToolRegistry",
    "ToolResult",
    "TurnRecord",
    "TypeSafeClient",
    "WebSearchProvider",
    "choice",
    "choice_route",
    "expected_calibration_error",
    "noul",
    "noul_route",
    "reliability_table",
    "score",
]
