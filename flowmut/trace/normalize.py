"""Normalisation of framework internals into control-flow trace labels.

The reward of Section "Selection with LinUCB" is computed over *event labels*
and *adjacent transitions* of a structured runtime trace.  Adapters observe raw
framework activity (dispatcher entries, autograd actions, graph-capture and
optimisation decisions, tensor adaptations, memory/alias handling,
synchronisation).  This module maps that activity onto stable labels and
deliberately drops tensor values and properties, producer/consumer identities
and kernel names, so that the trace stays distinct from the tensor-level
signatures used by DFSD.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from flowmut.adapters.base import TraceEvent

# ---------------------------------------------------------------------------
# Symbol scrubbing
# ---------------------------------------------------------------------------

#: Version-like or address-like suffixes that would make labels unstable.
_NOISE_PATTERNS = (
    re.compile(r"0x[0-9a-fA-F]+"),
    re.compile(r"\b\d{6,}\b"),
    re.compile(r"\.(cuda|npu|kernel|cu)\b"),
    re.compile(r"_\d+$"),
    re.compile(r"@\w+"),
    re.compile(r"\b(kernel|sm\d+|block|grid|thread)\w*", re.IGNORECASE),
)


def scrub(symbol: str) -> str:
    """Remove values, addresses and kernel identifiers from a raw symbol."""
    s = str(symbol)
    for pattern in _NOISE_PATTERNS:
        s = pattern.sub("", s)
    s = s.replace("::", ".").replace("__", ".")
    s = re.sub(r"[^\w.\->]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "unknown"


#: Substrings that mark an event as a failure rather than behaviour.
FAILURE_MARKERS = (
    "error", "exception", "fail", "abort", "invalid", "assert", "notimplemented",
    "unsupported", "unavailable", "timeout", "fallback_unsupported",
)


def is_failure_label(label: str) -> bool:
    low = label.lower()
    return any(m in low for m in FAILURE_MARKERS)


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

_DISPATCHER_HINTS = ("aten.", "dispatch", "op.", "primitive", "overload", "backend",
                     "delegate", "aten_")
_AUTOGRAD_HINTS = ("autograd", "grad", "backward", "forward_ad", "saved", "accumulate")
_CAPTURE_HINTS = ("capture", "trace", "jit", "compile", "dynamo", "symbolic",
                  "graph_break", "graphbreak", "export", "script")
_OPT_HINTS = ("fusion", "fuse", "optim", "simplif", "canonical", "rewrite", "pass",
              "constant_fold", "layout_opt", "scheduler", "lowering")
_ADAPT_HINTS = ("adapt", "promote", "cast", "convert", "copy", "to_", "contiguous",
                "resolve", "coerce", "materialize")
_MEMORY_HINTS = ("alloc", "memory", "alias", "storage", "buffer", "reuse", "pool",
                 "view", "inplace", "in_place")
_SYNC_HINTS = ("sync", "stream", "event", "barrier", "wait", "notify", "launch")


def categorize(label: str) -> str:
    """Assign a normalised label to one of the eight control-flow categories."""
    low = label.lower()
    if is_failure_label(low):
        return "failure"
    for hints, category in (
        (_CAPTURE_HINTS, "graph_capture"),
        (_OPT_HINTS, "optimization"),
        (_AUTOGRAD_HINTS, "autograd"),
        (_MEMORY_HINTS, "memory"),
        (_SYNC_HINTS, "sync"),
        (_ADAPT_HINTS, "adaptation"),
        (_DISPATCHER_HINTS, "dispatcher"),
    ):
        if any(h in low for h in hints):
            return category
    return "dispatcher"


# ---------------------------------------------------------------------------
# Trace recorder
# ---------------------------------------------------------------------------

@dataclass
class TraceRecorder:
    """Accumulates raw framework activity and yields normalised events.

    ``max_events`` bounds the trace length so that a pathological mutant cannot
    dominate the reward computation.  Repeated identical consecutive events are
    collapsed into a repeat count, which keeps transitions meaningful without
    losing the *number* of dispatches.
    """

    max_events: int = 4096
    drop_failures: bool = True
    _events: List[TraceEvent] = field(default_factory=list)
    _seen: Set[str] = field(default_factory=set)
    _overflow: bool = False

    def record(self, raw_label: str, category: Optional[str] = None,
               dedup_consecutive: bool = True) -> None:
        if len(self._events) >= self.max_events:
            self._overflow = True
            return
        label = scrub(raw_label)
        cat = category or categorize(label)
        if self.drop_failures and cat == "failure":
            return
        event = TraceEvent(label=label, category=cat)
        key = f"{cat}:{label}"
        if dedup_consecutive and self._events and \
                f"{self._events[-1].category}:{self._events[-1].label}" == key:
            return
        if key in self._seen and dedup_consecutive:
            # Keep the first occurrence per trace; repetition is not new behaviour.
            pass
        self._seen.add(key)
        self._events.append(event)

    def extend(self, raw_labels: Iterable[str]) -> None:
        for label in raw_labels:
            self.record(label)

    @property
    def events(self) -> List[TraceEvent]:
        return list(self._events)

    @property
    def overflowed(self) -> bool:
        return self._overflow

    def __len__(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Trace length control
# ---------------------------------------------------------------------------

def truncate(events: Sequence[TraceEvent], limit: int) -> List[TraceEvent]:
    if limit <= 0 or len(events) <= limit:
        return list(events)
    head = limit // 2
    tail = limit - head
    return list(events[:head]) + list(events[len(events) - tail:])


def category_histogram(events: Sequence[TraceEvent]) -> Dict[str, int]:
    return dict(Counter(e.category for e in events))


def label_set(events: Sequence[TraceEvent]) -> Set[str]:
    return {f"{e.category}:{e.label}" for e in events}


def transition_set(events: Sequence[TraceEvent]) -> Set[Tuple[str, str]]:
    labels = [f"{e.category}:{e.label}" for e in events]
    return set(zip(labels, labels[1:]))
