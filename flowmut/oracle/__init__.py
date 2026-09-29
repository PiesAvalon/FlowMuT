"""FlowMuT oracle package: legality checking and framework bug detection."""

from flowmut.oracle.oracle import (  # noqa: F401
    BOUNDED_OPS,
    SYMPTOMS,
    BugOracle,
    BugReport,
    Observation,
    check_finite,
    check_semantics,
    classify_error,
    compare_outputs,
    first_nonfinite,
)

__all__ = [
    "SYMPTOMS", "BOUNDED_OPS",
    "Observation", "BugReport", "BugOracle",
    "check_finite", "check_semantics", "first_nonfinite",
    "classify_error", "compare_outputs",
]
