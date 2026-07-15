"""
events.py
=========
Structured event log for the agent layer.

Replaces free-text run-log strings with machine-readable records:

    {"stage": "intent_detection", "action": "prediction_selected",
     "reason": "user requested prediction"}

This is deliberately dumb and dependency-free. It records; it decides nothing.
The deterministic ML core is NOT touched — its own `run_log` keeps working exactly as
before, and can be attached to an event as payload.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Event:
    stage: str                      # e.g. "intent_detection", "tool_routing", "tool_execution"
    action: str                     # e.g. "prediction_selected", "forecast_tool_invoked"
    reason: str = ""                # why this happened, in plain words
    data: dict = field(default_factory=dict)   # payload (metrics, errors, engine run_log...)
    status: str = "ok"              # "ok" | "error" | "skipped" | "not_implemented"
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class EventLog:
    """Ordered, append-only list of Events."""

    def __init__(self):
        self._events: list[Event] = []

    def add(self, stage: str, action: str, reason: str = "",
            data: dict | None = None, status: str = "ok") -> Event:
        ev = Event(stage=stage, action=action, reason=reason,
                   data=data or {}, status=status)
        self._events.append(ev)
        return ev

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._events]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_list(), indent=indent, ensure_ascii=False, default=_fallback)

    def pretty(self) -> list[str]:
        """Human-readable lines — for the existing Run Log tab."""
        out = []
        for e in self._events:
            mark = "" if e.status == "ok" else f" [{e.status}]"
            line = f"[{e.stage}] {e.action}{mark}"
            if e.reason:
                line += f" — {e.reason}"
            out.append(line)
        return out

    def __len__(self) -> int:
        return len(self._events)


def _fallback(o: Any):
    return str(o)
