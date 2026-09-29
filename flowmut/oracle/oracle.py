"""Test-case legality checking and bug oracles for FlowMuT.

A test case is *legal* when its mutation satisfies the mutation specification and
the documented preconditions of the affected operators, independently of its
execution outcome.  An execution failure caused by a violation of those
conditions therefore indicates an *illegal* test case rather than a framework
bug.

For a legal test case the oracle flags a potential framework bug when execution

* crashes or raises an uncaught exception (``crash`` / ``uncaught_exception``),
* violates documented operator semantics (``semantics``),
* produces a difference beyond the expected numerical tolerance across two
  semantically equivalent execution modes (``divergence``),
* yields non-finite values (``nonfinite``), or
* hangs the framework (``timeout``).

A failure is only counted as a bug when it can be reproduced, which is what
:meth:`BugOracle.reproduce` decides.

The module is deliberately framework-agnostic: it only consumes the
:class:`~flowmut.adapters.base.ExecutionResult` produced by an adapter and the
constraint outcome produced by the operator pool.  It never imports torch or
mindspore.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from flowmut.ir.ops import canonical_op

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from flowmut.adapters.base import ExecutionResult
    from flowmut.ir.graph import Graph
    from flowmut.operators.constraints import ConstraintReport

__all__ = [
    "SYMPTOMS",
    "BOUNDED_OPS",
    "Observation",
    "BugReport",
    "BugOracle",
    "check_semantics",
    "check_finite",
    "first_nonfinite",
    "classify_error",
    "compare_outputs",
]

# ---------------------------------------------------------------------------
# Symptom taxonomy
# ---------------------------------------------------------------------------

#: Every symptom the oracle can report.  ``"none"`` means "no anomaly observed".
SYMPTOMS: Tuple[str, ...] = (
    "none",
    "crash",
    "uncaught_exception",
    "semantics",
    "divergence",
    "nonfinite",
    "timeout",
    "illegal",
    "unavailable",
)

#: Ops whose outputs must stay within a documented range.
BOUNDED_OPS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "sigmoid": (0.0, 1.0),
    "tanh": (-1.0, 1.0),
    "softmax": (0.0, 1.0),
    "log_softmax": (None, 0.0),
    "hardsigmoid": (0.0, 1.0),
}

#: Exception type names that indicate a hard failure rather than an ordinary
#: recoverable Python exception.
_HARD_ERROR_TYPES: Tuple[str, ...] = (
    "runtimeerror",
    "systemerror",
    "memoryerror",
    "assertionerror",
    "internalerror",
    "fatalerror",
    "hardwareerror",
    "cudaerror",
    "cudnn",
    "segmentationfault",
)

#: Substrings in ``error_message`` that betray a native/hard failure.
_HARD_ERROR_MARKERS: Tuple[str, ...] = (
    "segmentation fault",
    "segfault",
    "sigsegv",
    "sigabrt",
    "core dumped",
    "abort",
    "terminate called",
    "fatal",
    "check failed",
    "out of memory",
    "std::bad_alloc",
    "cuda error",
    "cuda runtime",
    "device-side assert",
    "illegal memory access",
    "bus error",
    "kernel died",
    "internal error",
    "hardware error",
)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _coerce_float_array(value: Any) -> Optional[np.ndarray]:
    """Best-effort conversion of ``value`` to a float array.

    Returns ``None`` when the object cannot be interpreted as numeric.  Never
    raises: ``result.outputs`` may legally contain arbitrary objects.
    """
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    try:
        if arr.dtype == object:
            arr = arr.astype(np.float64)
        elif arr.dtype == bool:
            arr = arr.astype(np.float64)
        elif np.issubdtype(arr.dtype, np.complexfloating):
            arr = np.abs(arr)
        arr = arr.astype(np.float64, copy=False)
    except Exception:
        return None
    return arr


def first_nonfinite(outputs: Sequence[Any]) -> Tuple[int, str]:
    """Return ``(index, message)`` for the first output holding NaN/Inf.

    Returns ``(-1, "")`` when every output is finite (or cannot be inspected).
    """
    try:
        items = list(outputs or [])
    except Exception:
        return -1, ""
    for i, item in enumerate(items):
        if item is None:
            continue
        arr = _coerce_float_array(item)
        if arr is None or arr.size == 0:
            continue
        try:
            nan = int(np.count_nonzero(np.isnan(arr)))
            inf = int(np.count_nonzero(np.isinf(arr)))
        except Exception:
            continue
        if nan or inf:
            return i, (f"output[{i}] contains {nan} NaN and {inf} Inf value(s)")
    return -1, ""


def check_finite(outputs: Sequence[Any]) -> Optional[str]:
    """Return a message when any output holds NaN/Inf, else ``None``."""
    _index, message = first_nonfinite(outputs)
    return message or None


def classify_error(error_type: Any, error_message: Any) -> str:
    """Classify a failing execution as a hard ``crash`` or a soft exception."""
    etype = str(error_type or "").lower()
    emsg = str(error_message or "").lower()
    if any(token in etype for token in _HARD_ERROR_TYPES):
        return "crash"
    if any(marker in emsg for marker in _HARD_ERROR_MARKERS):
        return "crash"
    return "uncaught_exception"


def _compare_pair(x: Any, y: Any, tolerance: float, relative_tolerance: float,
                  index: int) -> Tuple[float, str]:
    """Compare two individual outputs; return ``(max_abs_diff, message)``."""
    ax = _coerce_float_array(x)
    ay = _coerce_float_array(y)
    if ax is None or ay is None:
        return float("inf"), (f"output[{index}]: value could not be interpreted "
                              f"as a numeric tensor")
    if ax.shape != ay.shape:
        return float("inf"), (f"output[{index}]: shape mismatch "
                              f"{tuple(ax.shape)} != {tuple(ay.shape)}")

    nan_x, nan_y = np.isnan(ax), np.isnan(ay)
    inf_x, inf_y = np.isinf(ax), np.isinf(ay)
    # non-finite on one side but not the other is a divergence
    issues = (nan_x ^ nan_y) | (inf_x ^ inf_y)
    # +inf vs -inf is a divergence too
    sign_mismatch = (inf_x & inf_y) & (np.sign(ax) != np.sign(ay))
    bad = issues | sign_mismatch
    if bool(np.any(bad)):
        count = int(np.count_nonzero(bad))
        return float("inf"), (f"output[{index}]: {count} non-finite element(s) "
                              f"differ between the two executions")

    finite = ~(nan_x | inf_x)
    if not bool(np.any(finite)):
        # both sides are identical non-finite patterns
        return 0.0, ""
    d = np.abs(ax[finite] - ay[finite])
    max_diff = float(np.max(d)) if d.size else 0.0
    try:
        close = bool(np.allclose(ax[finite], ay[finite],
                                 rtol=float(relative_tolerance),
                                 atol=float(tolerance)))
    except Exception:
        close = False
    if close:
        return max_diff, ""
    return max_diff, (f"output[{index}]: max|a-b|={max_diff:.6g} exceeds "
                      f"atol={tolerance:g} (rtol={relative_tolerance:g})")


def compare_outputs(a: Sequence[Optional[np.ndarray]],
                    b: Sequence[Optional[np.ndarray]],
                    tolerance: float, relative_tolerance: float
                    ) -> Tuple[int, float, str]:
    """Compare two output lists.

    Returns ``(worst_index, max_abs_diff, message)`` where ``message`` is empty
    exactly when the two lists agree within tolerance.  ``None`` entries,
    differing list lengths, shape mismatches and NaN/Inf are tolerated (a
    mismatch is itself reported as a divergence).  This function never raises.
    """
    try:
        left = list(a) if a is not None else []
    except TypeError:
        left = [a]
    except Exception:
        return -1, float("inf"), "left outputs could not be enumerated"
    try:
        right = list(b) if b is not None else []
    except TypeError:
        right = [b]
    except Exception:
        return -1, float("inf"), "right outputs could not be enumerated"

    worst_index = -1
    worst_diff = 0.0
    worst_message = ""
    count = max(len(left), len(right))
    for i in range(count):
        x = left[i] if i < len(left) else None
        y = right[i] if i < len(right) else None
        if x is None and y is None:
            continue
        if x is None or y is None:
            diff, message = float("inf"), (
                f"output[{i}]: present in only one of the two executions")
        else:
            try:
                diff, message = _compare_pair(x, y, tolerance,
                                              relative_tolerance, i)
            except Exception as exc:  # pragma: no cover - defensive
                diff, message = float("inf"), (
                    f"output[{i}]: comparison failed "
                    f"({type(exc).__name__}: {exc})")
        if message and (worst_index < 0 or diff > worst_diff):
            worst_index, worst_diff, worst_message = i, diff, message
    return worst_index, worst_diff, worst_message


# ---------------------------------------------------------------------------
# Semantic checks
# ---------------------------------------------------------------------------

def _bounded_output_targets(mutant_graph: "Graph"
                            ) -> List[Tuple[int, str]]:
    """Graph output positions whose producer is a bounded op, in order."""
    targets: List[Tuple[int, str]] = []
    try:
        graph_outputs = list(getattr(mutant_graph, "outputs", []) or [])
        nodes = getattr(mutant_graph, "nodes", {}) or {}
        edges = getattr(mutant_graph, "edges", {}) or {}
    except Exception:
        return targets
    for pos, edge_id in enumerate(graph_outputs):
        edge = edges.get(edge_id)
        if edge is None:
            continue
        producer = nodes.get(getattr(edge, "producer", ""))
        if producer is None:
            continue
        op = canonical_op(getattr(producer, "op", ""))
        if op in BOUNDED_OPS:
            targets.append((pos, op))
    return targets


def check_semantics(mutant_graph: "Graph", result: "ExecutionResult",
                    tolerance: float = 1e-3) -> Optional[str]:
    """Check documented output ranges for the graph's bounded operators.

    Returns a human-readable message for the first semantic violation, or
    ``None`` when every inspectable bounded output respects its documented
    range.  Outputs are matched to the graph's output edges positionally; when
    the adapter materialised only the bounded outputs, they are matched to the
    bounded graph outputs in order.
    """
    if mutant_graph is None or result is None:
        return None
    try:
        graph_outputs = list(getattr(mutant_graph, "outputs", []) or [])
        targets = _bounded_output_targets(mutant_graph)
        try:
            outputs = list(getattr(result, "outputs", []) or [])
        except Exception:
            outputs = []
    except Exception:
        return None

    if not targets or not outputs:
        return None
    if len(outputs) == len(graph_outputs):
        pairs = [(outputs[pos], pos, op) for (pos, op) in targets]
    elif len(outputs) == len(targets):
        pairs = [(outputs[k], pos, op)
                 for k, (pos, op) in enumerate(targets)]
    else:
        pairs = [(outputs[pos], pos, op) for (pos, op) in targets
                 if pos < len(outputs)]

    for value, pos, op in pairs:
        if value is None:
            continue
        arr = _coerce_float_array(value)
        if arr is None or arr.size == 0:
            continue
        lower, upper = BOUNDED_OPS[op]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        try:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        except Exception:
            continue
        if lower is not None and lo < lower - float(tolerance):
            return (f"output[{pos}] ({op}): min={lo:.6g} below the documented "
                    f"lower bound {lower!r}")
        if upper is not None and hi > upper + float(tolerance):
            return (f"output[{pos}] ({op}): max={hi:.6g} above the documented "
                    f"upper bound {upper!r}")
    return None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _violation_to_dict(violation: Any) -> Dict[str, Any]:
    to_dict = getattr(violation, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            pass
    if isinstance(violation, dict):
        return dict(violation)
    return {"repr": repr(violation)}


def _truncate(text: Any, limit: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if limit and limit > 0 and len(text) > limit:
        return text[:limit] + f"... [truncated, {len(text) - limit} chars omitted]"
    return text


@dataclass
class Observation:
    """What the oracle concluded about one mutant execution."""

    status: str = "ok"                 # the ExecutionResult status
    legal: bool = True                 # constraint report accepted
    is_bug: bool = False
    symptom: str = "none"              # one of SYMPTOMS
    detail: str = ""
    reproducible: bool = False
    #: constraint violations that made the mutant illegal (first N)
    violations: List[Any] = field(default_factory=list)
    #: index of the output that diverged / went non-finite, when applicable
    output_index: int = -1
    max_abs_diff: float = 0.0
    #: which constraint category the first violation belongs to (RQ3)
    violation_kind: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "legal": bool(self.legal),
            "is_bug": bool(self.is_bug),
            "symptom": self.symptom,
            "detail": self.detail,
            "reproducible": bool(self.reproducible),
            "violations": [_violation_to_dict(v) for v in self.violations],
            "output_index": int(self.output_index),
            "max_abs_diff": float(self.max_abs_diff),
            "violation_kind": self.violation_kind,
        }


@dataclass
class BugReport:
    """A reproducible potential framework bug."""

    symptom: str
    framework: str
    mode: str
    seed_model: str = ""
    operator: str = ""
    anchor: str = ""
    message: str = ""
    detail: str = ""
    traceback_text: str = ""
    duration_s: float = 0.0
    mutant_summary: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        """Stable dedup key: ``(symptom, framework, mode, seed_model, operator)``."""
        payload = "\x1f".join([
            str(self.symptom or ""),
            str(self.framework or ""),
            str(self.mode or ""),
            str(self.seed_model or ""),
            str(self.operator or ""),
        ])
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return f"bug:{digest}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key(),
            "symptom": self.symptom,
            "framework": self.framework,
            "mode": self.mode,
            "seed_model": self.seed_model,
            "operator": self.operator,
            "anchor": self.anchor,
            "message": self.message,
            "detail": self.detail,
            "traceback_text": self.traceback_text,
            "duration_s": float(self.duration_s),
            "mutant_summary": dict(self.mutant_summary),
        }


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class BugOracle:
    """Decide whether one mutant execution indicates a potential framework bug.

    :param adapter: the adapter under test (used only for bookkeeping / hints).
    :param reference_adapter: the semantically equivalent adapter whose outputs
        form the divergence reference (e.g. eager vs. compiled).
    :param tolerance: absolute numerical tolerance for divergence.
    :param relative_tolerance: relative numerical tolerance for divergence.
    :param check_divergence: enable the cross-mode divergence check.
    :param check_finite: enable the NaN/Inf check.
    :param check_semantics: enable the documented-semantics check.
    :param reproduce_attempts: default number of reproduction attempts.
    :param max_detail_chars: cap for free-text observation details.
    """

    def __init__(self, adapter: Any = None, reference_adapter: Any = None,
                 tolerance: float = 1e-3, relative_tolerance: float = 1e-4,
                 check_divergence: bool = True, check_finite: bool = True,
                 check_semantics: bool = True, reproduce_attempts: int = 1,
                 max_detail_chars: int = 4000) -> None:
        self.adapter = adapter
        self.reference_adapter = reference_adapter
        self.tolerance = float(tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.check_divergence = bool(check_divergence)
        self.check_finite = bool(check_finite)
        self.check_semantics = bool(check_semantics)
        self.reproduce_attempts = int(reproduce_attempts)
        self.max_detail_chars = int(max_detail_chars)
        self._stats: Dict[str, Any] = {
            "total": 0,
            "legal": 0,
            "illegal": 0,
            "bugs": 0,
            "symptoms": {s: 0 for s in SYMPTOMS},
            "violation_kinds": {},
        }

    # -- judging -----------------------------------------------------------
    def judge(self, report: "ConstraintReport", result: "ExecutionResult",
              reference: Optional["ExecutionResult"] = None,
              mutant_graph: Optional["Graph"] = None,
              seed_model: str = "", operator: str = "", anchor: str = ""
              ) -> Observation:
        """Classify one execution.  Never raises."""
        try:
            observation = self._judge(report, result, reference, mutant_graph,
                                      seed_model, operator, anchor)
        except Exception as exc:  # pragma: no cover - defensive
            status = "unavailable"
            try:
                status = str(getattr(result, "status", "unavailable") or "unavailable")
            except Exception:
                pass
            observation = Observation(
                status=status,
                legal=bool(getattr(report, "accepted", False)),
                is_bug=False,
                symptom="unavailable",
                detail=(f"oracle internal error: {type(exc).__name__}: {exc}"),
            )
        observation.detail = _truncate(observation.detail, self.max_detail_chars)
        self._record(observation)
        return observation

    def _judge(self, report: "ConstraintReport", result: "ExecutionResult",
               reference: Optional["ExecutionResult"],
               mutant_graph: Optional["Graph"],
               seed_model: str, operator: str, anchor: str) -> Observation:
        status = str(getattr(result, "status", "unavailable") or "unavailable")
        legal = bool(getattr(report, "accepted", True)) if report is not None else True
        observation = Observation(status=status, legal=legal)

        # 1. legality first: an illegal mutant is never a bug, even if it fails.
        if not legal:
            violations = list(getattr(report, "violations", []) or [])
            observation.violations = violations[:5]
            observation.violation_kind = str(
                getattr(report, "first_violation_kind", "") or "")
            observation.symptom = "illegal"
            observation.is_bug = False
            kind = observation.violation_kind or "unknown"
            parts = [
                f"illegal mutant: the mutation violates the specification "
                f"({len(violations)} recorded violation(s), first kind={kind})"
            ]
            if status != "ok":
                parts.append(
                    f"execution status={status} is explained by the violated "
                    f"preconditions and does not count as a framework bug")
            else:
                parts.append("execution succeeded but the mutant remains illegal")
            observation.detail = "; ".join(parts)
            return observation

        # 2. timeout: a legal mutant that hangs the framework is a bug.
        if status == "timeout":
            observation.symptom = "timeout"
            observation.is_bug = True
            observation.detail = (
                f"legal mutant exceeded the wall-clock budget "
                f"({getattr(result, 'duration_s', 0.0)}s)")
            return observation

        # 3. unavailable: the framework/mode could not even be exercised.
        if status == "unavailable":
            observation.symptom = "unavailable"
            observation.is_bug = False
            observation.detail = "framework/mode unavailable; no verdict possible"
            return observation

        # 4. error: crash (hard failure) or uncaught exception.
        if status == "error":
            symptom = classify_error(getattr(result, "error_type", ""),
                                     getattr(result, "error_message", ""))
            observation.symptom = symptom
            observation.is_bug = True
            error_type = getattr(result, "error_type", "") or "exception"
            error_message = getattr(result, "error_message", "") or ""
            observation.detail = (f"legal mutant raised {error_type}: "
                                  f"{error_message}").strip()
            return observation

        # 5. execution succeeded: finite -> semantics -> divergence.
        if self.check_finite:
            index, message = first_nonfinite(getattr(result, "outputs", []) or [])
            if index >= 0:
                observation.symptom = "nonfinite"
                observation.is_bug = True
                observation.output_index = index
                observation.detail = message
                return observation

        if self.check_semantics and mutant_graph is not None:
            message = check_semantics(mutant_graph, result,
                                      tolerance=self.tolerance)
            if message:
                observation.symptom = "semantics"
                observation.is_bug = True
                observation.detail = message
                return observation

        if reference is not None and self.check_divergence:
            index, max_diff, message = compare_outputs(
                getattr(result, "outputs", []) or [],
                getattr(reference, "outputs", []) or [],
                self.tolerance, self.relative_tolerance)
            if message:
                observation.symptom = "divergence"
                observation.is_bug = True
                observation.output_index = index
                observation.max_abs_diff = max_diff
                observation.detail = message
                return observation

        # 6. no anomaly.
        observation.symptom = "none"
        observation.is_bug = False
        observation.detail = ("legal execution matched the reference within "
                              "tolerance" if reference is not None
                              else "legal execution completed without anomalies")
        return observation

    # -- reproduction ------------------------------------------------------
    def reproduce(self, run_fn: Callable[[], "ExecutionResult"],
                  attempts: int = 2) -> bool:
        """Try to reproduce an already-observed failure.

        The in-campaign observation that motivated the call already counts as
        the *first* reproduction, so ``attempts <= 1`` is conservatively treated
        as reproducible without running anything.  Otherwise ``run_fn`` is
        invoked up to ``attempts`` times; a failing status, a non-finite
        ``ok`` result or a divergence marker in ``extras`` on any attempt makes
        the failure reproducible.  A Python exception raised by ``run_fn`` is
        only treated as a reproduction when it occurs on *every* attempt (it may
        otherwise be an infrastructure artefact rather than the framework bug).
        """
        try:
            total = int(attempts)
        except Exception:
            total = 2
        if total <= 1:
            # The first observation counts as the first reproduction.
            return True

        saw_failure = False
        raised = 0
        for _ in range(total):
            try:
                result = run_fn()
            except Exception:
                raised += 1
                continue
            if self._failure_symptom(result):
                saw_failure = True
        if saw_failure:
            return True
        # Exceptions only count when they happen on every attempt.
        return raised == total

    @staticmethod
    def _failure_symptom(result: Any) -> str:
        if result is None:
            return ""
        status = str(getattr(result, "status", "ok") or "ok")
        if status == "timeout":
            return "timeout"
        if status == "unavailable":
            return ""
        if status == "error":
            return classify_error(getattr(result, "error_type", ""),
                                  getattr(result, "error_message", ""))
        if status != "ok":
            return status
        extras = getattr(result, "extras", None) or {}
        try:
            if extras.get("divergence") is True:
                return "divergence"
        except Exception:
            pass
        index, _message = first_nonfinite(getattr(result, "outputs", []) or [])
        if index >= 0:
            return "nonfinite"
        return ""

    # -- reporting ---------------------------------------------------------
    def to_report(self, observation: Observation,
                  result: "ExecutionResult",
                  **meta: Any) -> Optional[BugReport]:
        """Build a :class:`BugReport` for a bug observation, else ``None``."""
        if observation is None or not observation.is_bug:
            return None
        try:
            framework = str(meta.get("framework")
                            or getattr(result, "framework", "") or "")
            mode = str(meta.get("mode") or getattr(result, "mode", "") or "")
            seed_model = str(meta.get("seed_model", "") or "")
            operator = str(meta.get("operator", meta.get("operator_name", "")) or "")
            anchor = str(meta.get("anchor", "") or "")
            mutant_summary = meta.get("mutant_summary") or {}
            if not isinstance(mutant_summary, dict):
                summary_fn = getattr(mutant_summary, "summary", None)
                mutant_summary = dict(summary_fn()) if callable(summary_fn) else {}
            message = meta.get("message") or self._auto_message(observation, result)
            detail = meta.get("detail") or observation.detail
            traceback_text = str(meta.get("traceback_text")
                                 or getattr(result, "error_traceback", "") or "")
            try:
                duration_s = float(getattr(result, "duration_s", 0.0) or 0.0)
            except Exception:
                duration_s = 0.0
            return BugReport(
                symptom=observation.symptom,
                framework=framework,
                mode=mode,
                seed_model=seed_model,
                operator=operator,
                anchor=anchor,
                message=_truncate(message, self.max_detail_chars),
                detail=_truncate(detail, self.max_detail_chars),
                traceback_text=_truncate(traceback_text, self.max_detail_chars),
                duration_s=duration_s,
                mutant_summary=dict(mutant_summary),
            )
        except Exception:  # pragma: no cover - defensive
            return None

    @staticmethod
    def _auto_message(observation: Observation, result: Any) -> str:
        error_type = str(getattr(result, "error_type", "") or "")
        error_message = str(getattr(result, "error_message", "") or "")
        if observation.symptom in ("crash", "uncaught_exception"):
            return f"{error_type or 'exception'}: {error_message}".strip()
        if observation.symptom == "timeout":
            return "execution timed out without producing a result"
        if observation.symptom == "nonfinite":
            return "execution produced NaN/Inf values"
        if observation.symptom == "semantics":
            return "execution violated documented operator semantics"
        if observation.symptom == "divergence":
            return (f"outputs diverged across semantically equivalent modes "
                    f"(max|a-b|={observation.max_abs_diff:.6g})")
        return observation.detail

    # -- bookkeeping -------------------------------------------------------
    def _record(self, observation: Observation) -> None:
        try:
            self._stats["total"] += 1
            if observation.legal:
                self._stats["legal"] += 1
            else:
                self._stats["illegal"] += 1
            if observation.is_bug:
                self._stats["bugs"] += 1
            symptom = observation.symptom if observation.symptom in SYMPTOMS else "none"
            self._stats["symptoms"][symptom] = self._stats["symptoms"].get(symptom, 0) + 1
            if observation.violation_kind:
                kinds = self._stats["violation_kinds"]
                kinds[observation.violation_kind] = \
                    kinds.get(observation.violation_kind, 0) + 1
        except Exception:  # pragma: no cover - defensive
            pass

    @property
    def statistics(self) -> Dict[str, Any]:
        """Counts of every symptom/legality decision seen so far."""
        return {
            "total": self._stats["total"],
            "judged": self._stats["total"],
            "legal": self._stats["legal"],
            "illegal": self._stats["illegal"],
            "bugs": self._stats["bugs"],
            "by_symptom": dict(self._stats["symptoms"]),
            "symptoms": dict(self._stats["symptoms"]),
            "violation_kinds": dict(self._stats["violation_kinds"]),
        }
