"""Framework/mode adapter interface.

A FlowMuT adapter turns a framework-neutral :class:`~flowmut.ir.Graph` into a
runnable program on one framework in one execution mode, profiles the tensors on
every edge, and reports the framework's *control-flow trace* for each execution.

Concrete adapters live in :mod:`flowmut.adapters.torch_adapter` (``eager``,
``compiled``) and :mod:`flowmut.adapters.mindspore_adapter` (``pynative``,
``graph``).  They are looked up through :mod:`flowmut.adapters.registry`.

Adapters are deliberately *out-of-process safe*: everything they need is a
Graph plus a :data:`Sample` (numpy arrays keyed by input edge id), so a campaign
for MindSpore can run from a MindSpore interpreter while the driver stays on the
PyTorch interpreter.
"""

from __future__ import annotations

import hashlib
import json
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from flowmut.ir.graph import Graph
from flowmut.ir.specs import TensorSpec, ValueProfile

#: A profiling input: numpy arrays keyed by graph input edge id.
Sample = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------

#: The control-flow categories the paper asks adapters to map framework
#: internals onto.  Tensor values, producer/consumer identities and kernel
#: names are deliberately *not* representable here.
TRACE_CATEGORIES: Tuple[str, ...] = (
    "dispatcher",     # dispatcher choices / backend selection
    "autograd",       # autograd graph actions
    "graph_capture",  # graph capture / tracing / compilation
    "optimization",   # graph optimisation and fusion decisions
    "adaptation",     # tensor adaptation (dtype/layout/device promotion)
    "memory",         # memory allocation and alias handling
    "sync",           # synchronisation and stream events
    "control",        # control-flow / fallback decisions
)

#: Categories excluded from the reward because they signal failures.
FAILURE_CATEGORIES: Tuple[str, ...] = ("failure",)


@dataclass(frozen=True)
class TraceEvent:
    """One normalised framework control-flow event."""

    label: str
    category: str = "dispatcher"

    def __str__(self) -> str:
        return f"{self.category}:{self.label}"

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "category": self.category}


def event_labels(events: Sequence[TraceEvent]) -> List[str]:
    return [f"{e.category}:{e.label}" for e in events]


def event_transitions(events: Sequence[TraceEvent]) -> List[str]:
    labels = event_labels(events)
    return [f"{a}->{b}" for a, b in zip(labels, labels[1:])]


# ---------------------------------------------------------------------------
# Execution outcomes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """What one execution of one mutant produced."""

    status: str = "ok"                 # "ok" | "error" | "timeout" | "unavailable"
    framework: str = ""
    mode: str = ""
    duration_s: float = 0.0
    error_type: str = ""
    error_message: str = ""
    error_traceback: str = ""
    events: List[TraceEvent] = field(default_factory=list)
    #: ``edge id -> tensor state`` observed in this execution (profile runs only).
    edge_specs: Dict[str, List[TensorSpec]] = field(default_factory=dict)
    #: Graph outputs as numpy arrays (used by the oracle for divergence checks).
    outputs: List[Optional[np.ndarray]] = field(default_factory=list)
    #: Mode-specific extras (compile counters, graph-break reasons, ...).
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    @property
    def crashed(self) -> bool:
        return self.status in ("error", "timeout")

    @property
    def failed(self) -> bool:
        return self.status != "ok"

    def labels(self) -> List[str]:
        return event_labels(self.events)

    def transitions(self) -> List[str]:
        return event_transitions(self.events)

    def to_dict(self, with_outputs: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status,
            "framework": self.framework,
            "mode": self.mode,
            "duration_s": self.duration_s,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "events": [e.to_dict() for e in self.events],
            "extras": {k: v for k, v in self.extras.items() if _jsonable_scalar(v)},
        }
        if with_outputs:
            d["outputs"] = [
                None if o is None else {"shape": list(o.shape), "dtype": str(o.dtype)}
                for o in self.outputs
            ]
        return d


def _jsonable_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


# ---------------------------------------------------------------------------
# Timeout guard
# ---------------------------------------------------------------------------

class TimeoutExceeded(Exception):
    pass


class TimeoutGuard:
    """Best-effort wall-clock guard around a single execution.

    Uses ``SIGALRM`` when running on the main thread of a Unix process; when the
    signal path is unavailable it degrades to a post-hoc duration check so the
    driver never crashes because of the guard itself.
    """

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds
        self._armed = False
        self._prev = None

    def __enter__(self) -> "TimeoutGuard":
        if self.seconds and self.seconds > 0:
            try:
                self._prev = signal.signal(signal.SIGALRM, self._raise)
                signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
                self._armed = True
            except (ValueError, AttributeError, OSError):
                self._armed = False
        return self

    @staticmethod
    def _raise(signum, frame):  # pragma: no cover - signal path
        raise TimeoutExceeded(f"execution exceeded the wall-clock budget")

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._armed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                if self._prev is not None:
                    signal.signal(signal.SIGALRM, self._prev)
            except (ValueError, OSError):
                pass
        return False


# ---------------------------------------------------------------------------
# Graph fingerprinting (parameter reuse across mutants)
# ---------------------------------------------------------------------------

def graph_fingerprint(graph: Graph) -> str:
    payload = json.dumps(
        [
            [n.id, n.op, n.inputs, {k: repr(v) for k, v in sorted(n.attrs.items())},
             n.scope, n.num_outputs]
            for n in graph.nodes.values()
        ] + [list(graph.inputs), list(graph.outputs)],
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def node_signature(node) -> str:
    """Signature used to decide whether a parameter may be reused after mutation.

    The signature is invariant under graph rewriting -- so a mutant keeps the
    weights of every node it did not touch -- but it must identify the *tensor*,
    not just the layer: the declared shape is part of it, otherwise two
    same-named parameters of different width (in two blocks, or in two seed
    models sharing an adapter) would map onto one buffer.
    """
    shape = getattr(node, "out_shape", None)
    payload = json.dumps([node.op, {k: repr(v) for k, v in sorted(node.attrs.items())},
                          node.scope, list(shape) if shape else None], sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

class FrameworkAdapter(ABC):
    """Base class for all framework/mode adapters."""

    framework: str = ""
    mode: str = ""

    def __init__(self, mode: Optional[str] = None, device: str = "cpu",
                 seed: int = 20240929, timeout_s: Optional[float] = None,
                 capture_values: bool = True):
        if mode:
            self.mode = mode
        self.device = device
        self.seed = int(seed)
        self.timeout_s = timeout_s
        self.capture_values = capture_values
        self._module_cache: Dict[str, Any] = {}

    # -- capabilities ------------------------------------------------------
    @classmethod
    @abstractmethod
    def is_available(cls) -> Tuple[bool, str]:
        """Return ``(available, reason)`` for this framework/mode."""

    @property
    def key(self) -> str:
        return f"{self.framework}:{self.mode}"

    def describe(self) -> Dict[str, Any]:
        ok, reason = self.is_available()
        return {"framework": self.framework, "mode": self.mode, "available": ok,
                "reason": reason, "device": self.device}

    # -- tensors -----------------------------------------------------------
    @abstractmethod
    def to_tensor(self, array: np.ndarray, dtype: str, device: Optional[str] = None,
                  requires_grad: bool = False) -> Any:
        """Convert a numpy array to a framework tensor."""

    @abstractmethod
    def to_numpy(self, tensor: Any) -> Optional[np.ndarray]:
        """Convert a framework tensor to numpy (``None`` when not materialisable)."""

    @abstractmethod
    def tensor_spec(self, tensor: Any) -> TensorSpec:
        """Describe a live tensor as ``F``-style annotation."""

    # -- model materialisation --------------------------------------------
    @abstractmethod
    def materialize(self, graph: Graph, reuse: Any = None) -> Any:
        """Build a runnable module for ``graph``, reusing parameters from ``reuse``."""

    @abstractmethod
    def execute(self, graph: Graph, sample: Sample, module: Any = None,
                capture_specs: bool = False) -> ExecutionResult:
        """Run ``graph`` on ``sample`` in this adapter's mode."""

    # -- convenience -------------------------------------------------------
    def profile(self, graph: Graph, samples: Sequence[Sample],
                module: Any = None) -> Dict[str, List[TensorSpec]]:
        """``ProfileExecution(f, X_p)``: collect ``F(e)`` for every edge."""
        merged: Dict[str, List[TensorSpec]] = {}
        for sample in samples:
            result = self.execute(graph, sample, module=module, capture_specs=True)
            if result.failed:
                continue
            for eid, specs in result.edge_specs.items():
                merged.setdefault(eid, []).extend(specs)
        return merged

    def close(self) -> None:
        self._module_cache.clear()


# ---------------------------------------------------------------------------
# Shared tensor-state helpers
# ---------------------------------------------------------------------------

def value_profile(array: np.ndarray) -> ValueProfile:
    """Binned numerical summary; never retains the tensor values themselves."""
    if array is None or array.size == 0:
        return ValueProfile()
    if array.dtype == bool:
        a = array.astype(np.float32)
    elif np.issubdtype(array.dtype, np.complexfloating):
        a = np.abs(array).astype(np.float32)
    else:
        a = array.astype(np.float64, copy=False)
    finite = np.isfinite(a)
    nan_count = int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    inf_count = int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    good = a[finite] if finite.any() else np.zeros(1, dtype=np.float64)
    abs_mean = float(np.abs(good).mean()) if good.size else 0.0
    magnitude = int(np.floor(np.log2(abs_mean))) if abs_mean > 0 else 0
    return ValueProfile(
        min=float(good.min()) if good.size else 0.0,
        max=float(good.max()) if good.size else 0.0,
        mean=float(good.mean()) if good.size else 0.0,
        std=float(good.std()) if good.size else 0.0,
        abs_mean=abs_mean,
        l2=float(np.sqrt(np.square(good).sum())) if good.size else 0.0,
        nan_count=nan_count,
        inf_count=inf_count,
        zero_fraction=float((a == 0).mean()) if a.size else 0.0,
        magnitude_bin=max(-30, min(30, magnitude)),
    )


#: Canonical dtype names -> numpy dtypes, shared by adapters.
NUMPY_DTYPES: Dict[str, Any] = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
    "complex64": np.complex64,
}


def numpy_dtype(dtype: str):
    return NUMPY_DTYPES.get(dtype, np.float32)


def stopwatch():
    return time.perf_counter()
