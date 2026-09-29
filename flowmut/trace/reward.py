"""Monitoring-trace memory and the LinUCB reward of FlowMuT.

Let ``E_t`` and ``Q_t`` be the event labels and adjacent transitions of the
trace observed at iteration ``t``, and ``E_{<t}``/``Q_{<t}`` those seen in
earlier executions.  The reward

    r_t = 1/2 * |E_t \\ E_{<t}| / max(1, |E_t|)
        + 1/2 * |Q_t \\ Q_{<t}| / max(1, |Q_t|)

equally weights the fractions of newly observed events and transitions.  Traces
with no new behaviour or no complete execution receive zero, and failure labels
are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from flowmut.adapters.base import ExecutionResult, TraceEvent
from flowmut.trace.normalize import label_set, transition_set, truncate


@dataclass
class TraceMemory:
    """``E_{<t}`` and ``Q_{<t}``: the accumulated event/transition histories."""

    events: Set[str] = field(default_factory=set)
    transitions: Set[Tuple[str, str]] = field(default_factory=set)
    #: How many executions contributed to the memory (used for reporting).
    executions: int = 0
    #: Per-execution reward history.
    rewards: List[float] = field(default_factory=list)
    #: Bounded history of the traces themselves, for diversity reporting.
    traces: List[List[TraceEvent]] = field(default_factory=list)
    max_traces: int = 512

    # -- observation -------------------------------------------------------
    def observe(self, result: ExecutionResult, max_events: int = 4096) -> float:
        """Compute ``r_t`` for ``result`` and fold it into the history."""
        events = truncate([e for e in result.events if e.category != "failure"], max_events)
        labels = label_set(events)
        transitions = transition_set(events)

        if result.status != "ok":
            # "Traces with no complete execution receive zero."
            reward = 0.0
        else:
            new_events = labels - self.events
            new_transitions = transitions - self.transitions
            e_term = len(new_events) / max(1, len(labels))
            q_term = len(new_transitions) / max(1, len(transitions))
            reward = 0.5 * e_term + 0.5 * q_term

        self.events |= labels
        self.transitions |= transitions
        self.executions += 1
        self.rewards.append(reward)
        if events:
            self.traces.append(list(events))
            if len(self.traces) > self.max_traces:
                self.traces.pop(0)
        return reward

    # -- reporting ---------------------------------------------------------
    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    def summary(self) -> Dict[str, Any]:
        n = len(self.rewards)
        return {
            "executions": self.executions,
            "distinct_events": len(self.events),
            "distinct_transitions": len(self.transitions),
            "mean_reward": (sum(self.rewards) / n) if n else 0.0,
            "cumulative_reward": sum(self.rewards),
            "zero_reward_fraction": (sum(1 for r in self.rewards if r <= 0.0) / n) if n else 0.0,
        }

    def reset(self) -> None:
        self.events.clear()
        self.transitions.clear()
        self.executions = 0
        self.rewards.clear()
        self.traces.clear()


def behavior_novelty(events: Sequence[TraceEvent], memory: TraceMemory) -> float:
    """Reward of a trace against a *copy* of the memory (no state update)."""
    labels = label_set(events)
    transitions = transition_set(events)
    new_events = labels - memory.events
    new_transitions = transitions - memory.transitions
    return (0.5 * len(new_events) / max(1, len(labels))
            + 0.5 * len(new_transitions) / max(1, len(transitions)))


def trace_signature(events: Sequence[TraceEvent]) -> Tuple[str, ...]:
    """A stable signature for the *execution behaviour* of one mutant."""
    return tuple(f"{e.category}:{e.label}" for e in events)
