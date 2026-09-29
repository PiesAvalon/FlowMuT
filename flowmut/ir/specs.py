"""Tensor-level descriptions shared by the IR, the TFG and every adapter.

The IR never holds framework tensors.  It records *specifications* (shape,
dtype, layout, device, gradient state, aliasing) that a framework adapter
materialises into real tensors.  ``F(e)`` (the TFG edge annotation) is the
profiled counterpart of :class:`TensorSpec`: the same fields, filled with
observed values instead of declared ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Canonical enumerations (kept as strings so they can be logged/deduplicated)
# ---------------------------------------------------------------------------

DTYPES: Tuple[str, ...] = (
    "float64",
    "float32",
    "float16",
    "bfloat16",
    "complex64",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint8",
    "bool",
)

FLOAT_DTYPES: Tuple[str, ...] = ("float64", "float32", "float16", "bfloat16")
INT_DTYPES: Tuple[str, ...] = ("int64", "int32", "int16", "int8", "uint8")
COMPLEX_DTYPES: Tuple[str, ...] = ("complex64",)

DTYPE_ITEMSIZE: Dict[str, int] = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "complex64": 8,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}

LAYOUTS: Tuple[str, ...] = ("contiguous", "strided", "channels_last", "sparse")

DEVICES: Tuple[str, ...] = ("cpu", "cuda", "npu", "meta")

FRAMEWORKS: Tuple[str, ...] = ("pytorch", "mindspore")

#: The four evaluated execution modes: ``(framework, mode)``.
MODES: Tuple[Tuple[str, str], ...] = (
    ("pytorch", "eager"),
    ("pytorch", "compiled"),
    ("mindspore", "pynative"),
    ("mindspore", "graph"),
)

MODE_ALIASES: Dict[str, Tuple[str, str]] = {
    "torch": ("pytorch", "eager"),
    "torch-eager": ("pytorch", "eager"),
    "pytorch": ("pytorch", "eager"),
    "torch-compiled": ("pytorch", "compiled"),
    "torch.compile": ("pytorch", "compiled"),
    "compiled": ("pytorch", "compiled"),
    "ms": ("mindspore", "pynative"),
    "mindspore": ("mindspore", "pynative"),
    "ms-pynative": ("mindspore", "pynative"),
    "ms-graph": ("mindspore", "graph"),
    "graph": ("mindspore", "graph"),
}


def parse_mode(text: str) -> Tuple[str, str]:
    """Resolve ``"torch-compiled"``/``"ms-graph"`` etc. to ``(framework, mode)``."""
    key = text.strip().lower().replace("_", "-")
    if ":" in key:
        head, _, tail = key.partition(":")
        key = f"{head}-{tail}" if head and tail else key
    if key in MODE_ALIASES:
        return MODE_ALIASES[key]
    for fw, mode in MODES:
        if key == f"{fw}-{mode}":
            return (fw, mode)
    raise ValueError(f"unknown framework/mode {text!r}; expected one of {sorted(MODE_ALIASES)}")


# ---------------------------------------------------------------------------
# Numeric summary of a profiled tensor (part of F(e))
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValueProfile:
    """Binned numerical summary of a tensor; values are never stored verbatim."""

    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    abs_mean: float = 0.0
    l2: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    zero_fraction: float = 0.0
    #: log2 bucket of the dynamic range, used by the DFSD signature / coverage.
    magnitude_bin: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "abs_mean": self.abs_mean,
            "l2": self.l2,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "zero_fraction": self.zero_fraction,
            "magnitude_bin": self.magnitude_bin,
        }

    @property
    def is_finite(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0


# ---------------------------------------------------------------------------
# TensorSpec: declared / profiled tensor description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorSpec:
    """Shape + property description of one tensor edge."""

    shape: Tuple[int, ...] = ()
    dtype: str = "float32"
    layout: str = "contiguous"
    device: str = "cpu"
    requires_grad: bool = False
    #: Identity of a shared storage buffer, or ``None`` when the tensor owns its
    #: storage.  Two edges with the same ``alias_of`` share memory.
    alias_of: Optional[str] = None
    #: True when the tensor is a non-contiguous view of another tensor.
    is_view: bool = False
    #: Whether the tensor is a leaf autograd node.
    is_leaf: bool = True
    strides: Optional[Tuple[int, ...]] = None
    value: Optional[ValueProfile] = None
    numel: int = 0

    # -- derived helpers ---------------------------------------------------
    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def itemsize(self) -> int:
        return DTYPE_ITEMSIZE.get(self.dtype, 4)

    @property
    def nbytes(self) -> int:
        n = self.numel or 1
        for d in self.shape:
            n *= int(d)
        return n * self.itemsize

    @property
    def is_floating(self) -> bool:
        return self.dtype in FLOAT_DTYPES

    @property
    def is_integral(self) -> bool:
        return self.dtype in INT_DTYPES

    @property
    def is_complex(self) -> bool:
        return self.dtype in COMPLEX_DTYPES

    def with_(self, **changes: Any) -> "TensorSpec":
        return replace(self, **changes)

    def non_batch_shape(self) -> Tuple[int, ...]:
        return tuple(self.shape[1:]) if self.shape else ()

    def key(self) -> Tuple[Any, ...]:
        """The *tensor state* coverage element (rank, dtype, device, layout,
        grad state and numerical properties), per Section "Coverage features"."""
        vp = self.value
        return (
            self.rank,
            self.dtype,
            self.device,
            self.layout,
            bool(self.requires_grad),
            bool(self.alias_of),
            bool(self.is_view),
            None if vp is None else vp.magnitude_bin,
        )

    def signature_tuple(self) -> Tuple[Any, ...]:
        """Part of the Data Flow Signature used by DFSD."""
        vp = self.value
        return (
            self.rank,
            self.dtype,
            self.layout,
            self.device,
            bool(self.alias_of),
            bool(self.requires_grad),
            None if vp is None else vp.magnitude_bin,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "device": self.device,
            "requires_grad": self.requires_grad,
            "alias_of": self.alias_of,
            "is_view": self.is_view,
            "is_leaf": self.is_leaf,
            "strides": None if self.strides is None else list(self.strides),
            "numel": self.numel,
            "value": None if self.value is None else self.value.to_dict(),
        }


# ---------------------------------------------------------------------------
# Lossless helpers
# ---------------------------------------------------------------------------

def shape_numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def broadcast_shape(a: Sequence[int], b: Sequence[int]) -> Optional[Tuple[int, ...]]:
    """NumPy-style broadcast shape, ``None`` when incompatible."""
    ra, rb = list(a), list(b)
    out: list = []
    for i in range(1, max(len(ra), len(rb)) + 1):
        da = ra[-i] if i <= len(ra) else 1
        db = rb[-i] if i <= len(rb) else 1
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            return None
    return tuple(reversed(out))


def is_broadcastable_to(src: Sequence[int], dst: Sequence[int]) -> bool:
    """True when ``src`` can broadcast to exactly ``dst``."""
    if len(src) > len(dst):
        return False
    offset = len(dst) - len(src)
    for i, d in enumerate(src):
        if d != dst[offset + i] and d != 1:
            return False
    return True


@dataclass
class OpRequirement:
    """One checkable input/internal/output requirement attached to a node (``R(v)``).

    ``kind`` is one of ``"input"``, ``"internal"``, ``"output"`` and matches the
    three constraint categories of Section "Mutation as Constrained Subgraph
    Replacement".  ``predicate`` is the name of a registered check function;
    ``args`` its parameters.
    """

    kind: str
    predicate: str
    args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "predicate": self.predicate,
            "args": dict(self.args),
            "description": self.description,
        }
