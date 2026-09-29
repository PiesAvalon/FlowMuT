"""Checkable operator input requirements.

Each function here implements one predicate named by
:class:`~flowmut.ir.specs.OpRequirement`.  A predicate receives a
:class:`ReqContext` and returns ``None`` when the requirement holds, or a short
human-readable message describing the violation.

The same predicate library is used for all three constraint categories of the
mutation abstraction:

``input``    requirements of nodes *inside* a replacement whose operands come
             from outside (checked against the boundary tensors),
``internal`` requirements among operands that both live inside the replacement,
``output``   requirements of *external consumers* that read the replacement's
             outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flowmut.ir.specs import (
    FLOAT_DTYPES,
    INT_DTYPES,
    TensorSpec,
    broadcast_shape,
    is_broadcastable_to,
)


@dataclass
class ReqContext:
    """Everything a predicate may inspect."""

    node: Any
    attrs: Dict[str, Any]
    #: Specs of the node inputs, positionally.  ``None`` entries mean "unknown".
    specs: List[Optional[TensorSpec]] = field(default_factory=list)
    #: Optional lookup for resolving specs of arbitrary boundary edge ids.
    spec_of: Dict[str, Optional[TensorSpec]] = field(default_factory=dict)
    graph: Any = None

    def spec(self, position: int) -> Optional[TensorSpec]:
        if position < 0 or position >= len(self.specs):
            return None
        return self.specs[position]

    def positions(self, value: Any) -> List[int]:
        if value == "all" or value is None:
            return list(range(len(self.specs)))
        if isinstance(value, int):
            return [value]
        return [int(v) for v in value]


Predicate = "callable"

_PREDICATES: Dict[str, Any] = {}


def register(name: str):
    def deco(fn):
        _PREDICATES[name] = fn
        return fn
    return deco


def predicate_names() -> List[str]:
    return sorted(_PREDICATES)


def evaluate(name: str, ctx: ReqContext) -> Optional[str]:
    fn = _PREDICATES.get(name)
    if fn is None:
        return None  # unknown predicate: treated as vacuously satisfied
    try:
        return fn(ctx)
    except Exception as exc:  # a broken predicate must never abort a campaign
        return f"predicate {name} raised {type(exc).__name__}: {exc}"


def _dims(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    try:
        return [int(v) for v in value]
    except TypeError:
        return None


# ---------------------------------------------------------------------------
# Single-operand predicates
# ---------------------------------------------------------------------------

@register("dtype_family")
def _dtype_family(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    family = ctx.attrs["family"]
    if family == "float" and spec.dtype not in FLOAT_DTYPES:
        return f"expected a floating point operand, found {spec.dtype}"
    if family == "integral" and spec.dtype not in INT_DTYPES:
        return f"expected an integral operand, found {spec.dtype}"
    if family == "bool" and spec.dtype != "bool":
        return f"expected a boolean operand, found {spec.dtype}"
    return None


@register("rank_eq")
def _rank_eq(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    if spec.rank != ctx.attrs["rank"]:
        return f"expected rank {ctx.attrs['rank']}, found {spec.rank}"
    return None


@register("rank_at_least")
def _rank_at_least(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    if spec.rank < ctx.attrs["rank"]:
        return f"expected rank >= {ctx.attrs['rank']}, found {spec.rank}"
    return None


@register("rank_at_most")
def _rank_at_most(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    if spec.rank > ctx.attrs["rank"]:
        return f"expected rank <= {ctx.attrs['rank']}, found {spec.rank}"
    return None


@register("value_finite")
def _value_finite(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None or spec.value is None:
        return None
    vp = spec.value
    if vp.nan_count or vp.inf_count:
        return (f"operand holds {vp.nan_count} NaN and {vp.inf_count} Inf elements; "
                f"the operator has no defined value for them")
    return None


@register("numel_positive")
def _numel_positive(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None or not spec.shape:
        return None
    if any(int(d) <= 0 for d in spec.shape):
        return f"operand shape {tuple(spec.shape)} is empty"
    return None


@register("value_range")
def _value_range(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None or spec.value is None:
        return None
    lo, hi = ctx.attrs.get("lo"), ctx.attrs.get("hi")
    vp = spec.value
    if lo is not None and vp.min < lo:
        return f"operand minimum {vp.min:.4g} is below the documented domain {lo}"
    if hi is not None and vp.max > hi:
        return f"operand maximum {vp.max:.4g} is above the documented domain {hi}"
    return None


@register("requires_grad_state")
def _requires_grad_state(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    want = bool(ctx.attrs.get("requires_grad", False))
    if spec.requires_grad != want:
        return (f"operand requires_grad={spec.requires_grad} but the replacement "
                f"expects {want}")
    return None


@register("layout_contiguous")
def _layout_contiguous(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    if spec.layout not in ("contiguous", "channels_last"):
        return f"operand layout {spec.layout} is not directly consumable"
    return None


# ---------------------------------------------------------------------------
# Multi-operand predicates
# ---------------------------------------------------------------------------

@register("device_agreement")
def _device_agreement(ctx: ReqContext) -> Optional[str]:
    positions = ctx.positions(ctx.attrs.get("positions"))
    specs = [ctx.spec(p) for p in positions]
    specs = [s for s in specs if s is not None]
    if len(specs) < 2:
        return None
    devices = {s.device for s in specs}
    if len(devices) > 1:
        return f"operands live on different devices: {sorted(devices)}"
    return None


@register("dtype_agreement")
def _dtype_agreement(ctx: ReqContext) -> Optional[str]:
    positions = ctx.positions(ctx.attrs.get("positions"))
    specs = [ctx.spec(p) for p in positions]
    specs = [s for s in specs if s is not None]
    if len(specs) < 2:
        return None
    dtypes = {s.dtype for s in specs}
    if len(dtypes) > 1:
        return f"operands have different dtypes: {sorted(dtypes)}"
    return None


@register("shape_broadcastable")
def _shape_broadcastable(ctx: ReqContext) -> Optional[str]:
    positions = ctx.positions(ctx.attrs.get("positions"))
    specs = [ctx.spec(p) for p in positions]
    if any(s is None for s in specs) or len(specs) < 2:
        return None
    shape = list(specs[0].shape)
    for s in specs[1:]:
        merged = broadcast_shape(shape, s.shape)
        if merged is None:
            return (f"shapes {tuple(specs[0].shape)} and {tuple(s.shape)} "
                    f"cannot be broadcast together")
        shape = list(merged)
    return None


@register("matmul_compatible")
def _matmul_compatible(ctx: ReqContext) -> Optional[str]:
    positions = ctx.positions(ctx.attrs.get("positions"))
    if len(positions) < 2:
        return None
    a, b = ctx.spec(positions[0]), ctx.spec(positions[1])
    if a is None or b is None:
        return None
    if a.rank < 2 or b.rank < 2:
        return "matrix multiplication needs operands of rank >= 2"
    if a.shape[-1] != b.shape[-2]:
        return (f"inner dimensions disagree: {a.shape[-1]} vs {b.shape[-2]}")
    if a.rank > 2 and b.rank > 2:
        if broadcast_shape(a.shape[:-2], b.shape[:-2]) is None:
            return "batch dimensions are not broadcastable"
    return None


@register("last_dim_match")
def _last_dim_match(ctx: ReqContext) -> Optional[str]:
    positions = ctx.positions(ctx.attrs.get("positions"))
    if len(positions) < 2:
        return None
    a, b = ctx.spec(positions[0]), ctx.spec(positions[1])
    if a is None or b is None or not a.shape or not b.shape:
        return None
    if a.shape[-1] != b.shape[-1]:
        return f"trailing dimensions disagree: {a.shape[-1]} vs {b.shape[-1]}"
    return None


@register("shapes_equal")
def _shapes_equal(ctx: ReqContext) -> Optional[str]:
    positions = ctx.positions(ctx.attrs.get("positions"))
    specs = [ctx.spec(p) for p in positions]
    if any(s is None for s in specs) or len(specs) < 2:
        return None
    first = tuple(specs[0].shape)
    for s in specs[1:]:
        if tuple(s.shape) != first:
            return f"shapes differ: {first} vs {tuple(s.shape)}"
    return None


@register("concat_dims_match")
def _concat_dims_match(ctx: ReqContext) -> Optional[str]:
    positions = ctx.positions(ctx.attrs.get("positions"))
    specs = [ctx.spec(p) for p in positions]
    if any(s is None for s in specs) or len(specs) < 2:
        return None
    dim = ctx.attrs.get("dim", 0)
    rank = specs[0].rank
    if any(s.rank != rank for s in specs):
        return f"concatenated operands must share a rank, found {[s.rank for s in specs]}"
    dim = dim if dim >= 0 else rank + dim
    if not (0 <= dim < rank):
        return f"concatenation dimension {ctx.attrs.get('dim')} is out of range for rank {rank}"
    for i in range(rank):
        if i == dim:
            continue
        first = specs[0].shape[i]
        for s in specs[1:]:
            if s.shape[i] != first:
                return (f"dimension {i} differs across operands "
                        f"({first} vs {s.shape[i]}) while concatenating on dim {dim}")
    return None


@register("attention_dims_match")
def _attention_dims_match(ctx: ReqContext) -> Optional[str]:
    positions = ctx.positions(ctx.attrs.get("positions"))
    specs = [ctx.spec(p) for p in positions]
    if any(s is None for s in specs) or len(specs) < 3:
        return None
    q, k, v = specs[0], specs[1], specs[2]
    if q.rank != k.rank or k.rank != v.rank:
        return f"query/key/value ranks disagree: {[s.rank for s in (q, k, v)]}"
    if q.shape[-1] != k.shape[-1]:
        return f"query/key embedding dims disagree: {q.shape[-1]} vs {k.shape[-1]}"
    if k.shape[-2] != v.shape[-2]:
        return f"key/value sequence lengths disagree: {k.shape[-2]} vs {v.shape[-2]}"
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        return "batch dimensions of query/key/value disagree"
    if q.shape[1] != k.shape[1] or k.shape[1] != v.shape[1]:
        return f"head counts disagree: {[s.shape[1] for s in (q, k, v)]}"
    return None


# ---------------------------------------------------------------------------
# Operator-specific internal predicates
# ---------------------------------------------------------------------------

@register("channel_groups_match")
def _channel_groups_match(ctx: ReqContext) -> Optional[str]:
    positions = ctx.positions(ctx.attrs.get("positions"))
    if len(positions) < 2:
        return None
    x, w = ctx.spec(positions[0]), ctx.spec(positions[1])
    if x is None or w is None or x.rank < 2 or w.rank < 2:
        return None
    groups = int(ctx.attrs.get("groups", 1) or 1)
    in_channels = x.shape[1]
    weight_in = w.shape[0] if ctx.attrs.get("transposed") else w.shape[1]
    if weight_in <= 0:
        return None
    if in_channels % weight_in != 0:
        return (f"input channels {in_channels} are not divisible by the weight's "
                f"per-group channels {weight_in}")
    if in_channels // weight_in != groups:
        return (f"declared groups={groups} disagrees with the weight layout "
                f"({in_channels} input channels / {weight_in} per group)")
    return None


@register("spatial_kernel_fits")
def _spatial_kernel_fits(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None or spec.rank != 4:
        return None
    node = ctx.node
    w = ctx.spec(1) if len(ctx.specs) > 1 else None
    if w is None or w.rank != 4:
        return None
    _, _, h, wd = spec.shape
    kh, kw = w.shape[2], w.shape[3]
    pad = _dims(node.attrs.get("padding", 0)) or [0]
    dil = _dims(node.attrs.get("dilation", 1)) or [1]
    if h + 2 * pad[0] < (kh - 1) * dil[0] + 1 or wd + 2 * pad[-1] < (kw - 1) * dil[-1] + 1:
        return (f"kernel {kh}x{kw} does not fit the padded spatial size "
                f"{h + 2 * pad[0]}x{wd + 2 * pad[-1]}")
    return None


@register("kernel_fits_input")
def _kernel_fits_input(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None or spec.rank != 4:
        return None
    node = ctx.node
    _, _, h, wd = spec.shape
    k = _dims(node.attrs.get("kernel_size", 2)) or [2]
    p = _dims(node.attrs.get("padding", 0)) or [0]
    if h + 2 * p[0] < k[0] or wd + 2 * p[-1] < k[-1]:
        return (f"pooling kernel {k[0]}x{k[-1]} is larger than the padded input "
                f"{h + 2 * p[0]}x{wd + 2 * p[-1]}")
    return None


@register("channel_dim_known")
def _channel_dim_known(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None or spec.rank < 2:
        return None
    num_features = ctx.node.attrs.get("num_features")
    if num_features is None:
        return None
    if int(spec.shape[1]) != int(num_features):
        return (f"normalisation expects {num_features} channels but the operand has "
                f"{spec.shape[1]}")
    return None


@register("channel_divisible_by_groups")
def _channel_divisible_by_groups(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None or spec.rank < 2:
        return None
    groups = int(ctx.node.attrs.get("num_groups", 1) or 1)
    if groups <= 0 or spec.shape[1] % groups:
        return f"channels {spec.shape[1]} are not divisible by num_groups={groups}"
    return None


@register("normalized_shape_matches")
def _normalized_shape_matches(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    norm = ctx.node.attrs.get("normalized_shape")
    if norm is None:
        return None
    norm = list(norm) if isinstance(norm, (tuple, list)) else [int(norm)]
    if len(norm) > spec.rank:
        return f"normalized_shape {tuple(norm)} exceeds the operand rank {spec.rank}"
    tail = list(spec.shape[len(spec.shape) - len(norm):])
    if tail != norm:
        return f"normalized_shape {tuple(norm)} does not match the trailing dims {tuple(tail)}"
    return None


@register("reshape_numel_preserved")
def _reshape_numel_preserved(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None or not spec.shape:
        return None
    from flowmut.ir.specs import shape_numel
    target = ctx.node.attrs.get("shape") or ctx.node.attrs.get("size")
    if target is None:
        return None
    target = [int(d) for d in target]
    known = [d for d in target if d > 0]
    src = shape_numel(spec.shape)
    if any(d < 0 for d in target):
        if known and src % shape_numel(known) != 0:
            return f"cannot infer the inferred dimension for {tuple(target)} from {src} elements"
        return None
    if shape_numel(target) != src:
        return (f"reshaping {tuple(spec.shape)} ({src} elements) into {tuple(target)} "
                f"({shape_numel(target)} elements) is not size preserving")
    return None


@register("permutation_valid")
def _permutation_valid(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    dims = ctx.node.attrs.get("dims")
    if dims is None:
        return None
    dims = [int(d) if d >= 0 else spec.rank + int(d) for d in dims]
    if sorted(dims) != list(range(spec.rank)):
        return f"dims {tuple(ctx.node.attrs['dims'])} is not a permutation of rank {spec.rank}"
    return None


@register("transpose_dims_valid")
def _transpose_dims_valid(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    d0 = ctx.node.attrs.get("dim0", 0)
    d1 = ctx.node.attrs.get("dim1", 1)
    d0 = d0 if d0 >= 0 else spec.rank + d0
    d1 = d1 if d1 >= 0 else spec.rank + d1
    if d0 == d1:
        return "transpose dimensions must be distinct"
    if not (0 <= d0 < spec.rank and 0 <= d1 < spec.rank):
        return f"transpose dimensions {(d0, d1)} are out of range for rank {spec.rank}"
    return None


@register("squeeze_dim_is_one")
def _squeeze_dim_is_one(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    dim = ctx.node.attrs.get("dim")
    if dim is None:
        return None
    dim = dim if dim >= 0 else spec.rank + dim
    if not (0 <= dim < spec.rank):
        return f"squeeze dimension {dim} is out of range for rank {spec.rank}"
    if spec.shape[dim] != 1:
        return f"cannot squeeze dimension {dim} of size {spec.shape[dim]}"
    return None


@register("expandable_to")
def _expandable_to(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    target = ctx.node.attrs.get("shape")
    if target is None:
        return None
    target = tuple(int(d) for d in target)
    if not is_broadcastable_to(spec.shape, target):
        return f"shape {tuple(spec.shape)} cannot be expanded to {target}"
    return None


@register("splittable")
def _splittable(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    dim = ctx.node.attrs.get("dim", 0)
    dim = dim if dim >= 0 else spec.rank + dim
    if not (0 <= dim < spec.rank):
        return f"split dimension {ctx.node.attrs.get('dim')} is out of range for rank {spec.rank}"
    n = ctx.node.attrs.get("sections") or ctx.node.attrs.get("num_outputs") or 2
    if isinstance(n, (list, tuple)):
        if sum(int(v) for v in n) != spec.shape[dim]:
            return f"split sections {tuple(n)} do not sum to dimension size {spec.shape[dim]}"
        return None
    if int(n) <= 0 or spec.shape[dim] % int(n):
        return f"dimension size {spec.shape[dim]} is not divisible into {n} parts"
    return None


@register("index_in_range")
def _index_in_range(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    node = ctx.node
    if node.op == "embedding":
        table = ctx.spec(1)
        if table is None or table.rank != 2 or spec.value is None:
            return None
        if spec.value.max >= table.shape[0] or spec.value.min < 0:
            return (f"index range [{spec.value.min:.0f}, {spec.value.max:.0f}] leaves the "
                    f"embedding table of size {table.shape[0]}")
        return None
    idx = node.attrs.get("index")
    if isinstance(idx, int):
        dim = node.attrs.get("dim", 0)
        dim = dim if dim >= 0 else spec.rank + dim
        if not (0 <= dim < spec.rank):
            return f"dimension {node.attrs.get('dim')} is out of range for rank {spec.rank}"
        if not (-spec.shape[dim] <= idx < spec.shape[dim]):
            return (f"index {idx} is out of range for dimension {dim} "
                    f"of size {spec.shape[dim]}")
    return None


@register("pad_width_valid")
def _pad_width_valid(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None:
        return None
    pad = ctx.node.attrs.get("pad")
    if pad is None:
        return None
    if len(pad) % 2 or len(pad) // 2 > spec.rank:
        return (f"pad list of length {len(pad)} cannot describe padding for rank "
                f"{spec.rank}")
    if any(int(p) < 0 for p in pad):
        return "negative padding is not defined for this operator"
    return None


@register("channel_divisible")
def _channel_divisible(ctx: ReqContext) -> Optional[str]:
    spec = ctx.spec(ctx.attrs["position"])
    if spec is None or spec.rank != 4:
        return None
    r = int(ctx.node.attrs.get("upscale_factor", 2) or 2)
    if spec.shape[1] % (r * r):
        return f"channels {spec.shape[1]} are not divisible by {r * r}"
    return None


@register("interpolation_align_valid")
def _interpolation_align_valid(ctx: ReqContext) -> Optional[str]:
    mode = ctx.node.attrs.get("mode", "nearest")
    align = ctx.node.attrs.get("align_corners")
    if align is not None and mode in ("nearest", "area"):
        return f"align_corners is not defined for mode={mode!r}"
    return None


@register("no_alias_with_external")
def _no_alias_with_external(ctx: ReqContext) -> Optional[str]:
    """Replacement outputs must not alias live storage outside the site."""
    node = ctx.node
    internal = set(ctx.attrs.get("internal_nodes", []) or [])
    for position, spec in enumerate(ctx.specs):
        if spec is None or not spec.alias_of:
            continue
        producer = ctx.graph.edges[node.inputs[position]].producer if (
            ctx.graph is not None and position < len(node.inputs) and node.inputs[position] in ctx.graph.edges
        ) else None
        if producer is not None and producer not in internal:
            return (f"operand {position} aliases storage produced outside the mutation "
                    f"site (alias {spec.alias_of})")
    return None


@register("inplace_safe")
def _inplace_safe(ctx: ReqContext) -> Optional[str]:
    """An in-place replacement must not overwrite a tensor with other consumers."""
    graph = ctx.graph
    node = ctx.node
    if graph is None or not node.inputs:
        return None
    edge_id = node.inputs[0]
    if edge_id not in graph.edges:
        return None
    others = [c for c in graph.data_consumers_of(edge_id) if c != node.id]
    if others:
        return (f"in-place mutation would clobber a tensor still consumed by "
                f"{len(others)} other operator(s)")
    return None
