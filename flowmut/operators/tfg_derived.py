"""The 48 TFG-derived mutation operators of FlowMuT.

These operators are *derived from the Tensor Flow Graph* ``TFG = (G, F, R)``:
they exploit information that a plain computation graph does not carry, namely

* ``F(e)`` -- the profiled runtime tensor state of an edge: shape/rank, dtype,
  layout, device, alias/storage identity, ``requires_grad`` state and the binned
  numerical profile (``zero_fraction``, ``magnitude_bin``, ``nan_count`` ...),
* ``R(v)`` -- the operator input requirements attached to a node,
* the *multiplicity* and *polymorphism* of data flow: how many consumers an edge
  serves, whether its shape varies across profiling inputs, and which bounded
  structures (residual/skip connections) the edge participates in.

Construction follows Section "TFG-Derived Mutation Operators" of the paper:
four mutable object types (``Tensor``, ``Operator``, ``Interface``, ``Subgraph``)
crossed with four mutation primitives (``Update``, ``Insertion``, ``Deletion``,
``Rewiring``).  The retained combinations give the cell counts the pool table
``PAPER_POOL_TABLE`` records::

    Tensor/Update       8      Interface/Update      2
    Tensor/Insertion    4      Interface/Insertion   6
    Operator/Update     6      Interface/Deletion    2
    Operator/Insertion  3      Interface/Rewiring    3
    Operator/Deletion   1      Subgraph/Update       4
                               Subgraph/Insertion    3
                               Subgraph/Deletion     2
                               Subgraph/Rewiring     4
    TOTAL              48

Implementation conventions
--------------------------
* Every operator anchors on a *tensor edge* ``e`` and reads ``F(e)`` through
  ``tfg.f(e)``; the ``Pattern.predicate`` names a site predicate registered in
  this module that either reads ``F(e)``, ``R(v)`` or the bounded data-flow
  structure around the edge.
* ``Tensor``/``Operator``/``Interface``/``Subgraph`` *Update* operators change a
  label or configuration without touching the connections: they emit
  ``Replacement(remove=[], attr_updates={node: {...}})``.  Tensor-state labels
  are recorded with the ``tensor_state`` attribute (``"zero"``, ``"saturate"``,
  ``"denormal"``, ``"negate"``), which the framework adapters apply as a
  post-op on the produced tensor; gradient state uses the ``requires_grad``
  attribute, and interface labels use ``input_dtype`` / ``input_layout``.
* *Insertion* operators place one new operator on the anchor edge and rewire the
  affected consumers.  ``branch_only=True`` rewires only the matched consumer,
  which is how the shared-edge / branch-local family changes a single data path.
* *Deletion* operators remove the matched bounded subgraph and reconnect its
  boundary to a surviving edge.
* ``build_replacement`` returns ``None`` when ``F(e)`` does not exhibit the
  property the operator is derived from, so a pattern may be broad while the
  applicability condition stays strictly data-flow driven.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from flowmut.ir.ops import canonical_op
from flowmut.ir.specs import broadcast_shape
from flowmut.operators.base import (
    ConstraintSpec,
    FunctionalOperator,
    MutationOperator,
    NodeTemplate,
    Pattern,
    Replacement,
    Site,
    register_site_predicate,
)
from flowmut.tfg.tfg import TFG

# ---------------------------------------------------------------------------
# Operator-name groups used by the patterns
# ---------------------------------------------------------------------------

#: Ops that materialise a strided/aliased view of their operand.
VIEW_OPS: Tuple[str, ...] = ("view", "permute", "transpose", "squeeze",
                             "unsqueeze", "expand", "broadcast_to", "getitem")
#: Ops that perform an implicit shape adaptation at a producer/consumer boundary.
SHAPE_ADAPTER_OPS: Tuple[str, ...] = ("flatten", "reshape", "view", "squeeze",
                                      "unsqueeze")
#: Convolutions and contractions, i.e. operators whose cost depends on layout.
CONV_OPS: Tuple[str, ...] = ("conv2d", "conv1d", "conv_transpose2d")
CONTRACTION_OPS: Tuple[str, ...] = ("linear", "matmul", "bmm", "einsum")
COMPUTE_OPS: Tuple[str, ...] = CONV_OPS + CONTRACTION_OPS
#: Normalisation operators.
NORM_OPS: Tuple[str, ...] = ("batchnorm", "layernorm", "groupnorm",
                             "instancenorm", "rmsnorm", "l2norm")
#: Floating-point activations.
ACTIVATION_OPS: Tuple[str, ...] = ("relu", "leaky_relu", "elu", "gelu", "silu",
                                   "sigmoid", "tanh", "mish", "hardswish",
                                   "hardsigmoid", "quick_gelu", "softplus")
#: Unary operators a deletion may collapse without changing the arity.
REMOVABLE_UNARY_OPS: Tuple[str, ...] = ACTIVATION_OPS + ("dropout", "identity",
                                                         "clamp")
#: Pooling operators.
POOL_OPS: Tuple[str, ...] = ("avgpool2d", "maxpool2d", "adaptive_avgpool2d",
                             "adaptive_maxpool2d", "global_avgpool",
                             "global_maxpool")
#: Binary operators that implement a bounded (residual) data-flow structure.
RESIDUAL_OPS: Tuple[str, ...] = ("add", "sub")

#: Convolution classes that can deliver a probabilistic profile.
_BINARY_RESIDUAL = RESIDUAL_OPS


# ---------------------------------------------------------------------------
# Site predicates (the data-flow-aware extension point of Pattern)
# ---------------------------------------------------------------------------

def _spec(tfg: TFG, edge: str):
    try:
        return tfg.f(edge)
    except Exception:  # pragma: no cover - defensive, F(e) is total
        return None


def _value(tfg: TFG, edge: str):
    spec = _spec(tfg, edge)
    return None if spec is None else spec.value


def _position_of(tfg: TFG, node, edge_id: str) -> Optional[int]:
    for position, edge in enumerate(node.inputs):
        if edge == edge_id:
            return position
    return None


def _data_inputs(tfg: TFG, node) -> List[str]:
    out: List[str] = []
    for edge in node.inputs:
        if not edge or edge not in tfg.graph.edges:
            continue
        if tfg.graph.nodes[tfg.graph.edges[edge].producer].is_param:
            continue
        out.append(edge)
    return out


def _is_residual(tfg: TFG, consumer: str) -> bool:
    """``True`` for a bounded structure where one input traces back to another.

    This is the residual/skip detector of Section "TFG-Derived Mutation
    Operators": a node with two data inputs where the producer of one input is
    reachable from the producer of the other input, i.e. the two paths share a
    subnet and the structure can be re-bound without changing the interface.
    """
    node = tfg.graph.nodes.get(consumer)
    if node is None or canonical_op(node.op) not in _BINARY_RESIDUAL:
        return False
    data_inputs = _data_inputs(tfg, node)
    if len(data_inputs) < 2:
        return False
    upstream = {e: tfg.graph.reachable_from([tfg.graph.edges[e].producer])
                for e in data_inputs}
    for a in data_inputs:
        for b in data_inputs:
            if a == b:
                continue
            producer_b = tfg.graph.edges[b].producer
            if producer_b == tfg.graph.edges[a].producer or producer_b in upstream[a]:
                return True
    return False


@register_site_predicate("tfg_value_profiled")
def _p_value_profiled(tfg: TFG, site: Site) -> bool:
    """The edge carries a binned numerical profile ``F(e).value``."""
    return _value(tfg, site.anchor_edge) is not None


@register_site_predicate("tfg_grad_annotated")
def _p_grad_annotated(tfg: TFG, site: Site) -> bool:
    """The edge records a gradient state, i.e. it is autograd-tracked or not."""
    return _spec(tfg, site.anchor_edge) is not None


@register_site_predicate("tfg_grad_tracked")
def _p_grad_tracked(tfg: TFG, site: Site) -> bool:
    """``F(e).requires_grad`` is set, so a gradient-state edit is meaningful."""
    spec = _spec(tfg, site.anchor_edge)
    return spec is not None and bool(spec.requires_grad)


@register_site_predicate("tfg_layout_annotated")
def _p_layout_annotated(tfg: TFG, site: Site) -> bool:
    """``F(e)`` records a memory layout for the edge."""
    spec = _spec(tfg, site.anchor_edge)
    return spec is not None and spec.layout is not None


@register_site_predicate("tfg_alias_or_view")
def _p_alias_or_view(tfg: TFG, site: Site) -> bool:
    """``F(e)`` is aliased storage, a view, or feeds a view operator."""
    spec = _spec(tfg, site.anchor_edge)
    if spec is not None and (spec.alias_of or spec.is_view):
        return True
    return any(canonical_op(tfg.graph.nodes[c].op) in VIEW_OPS
               for c in tfg.data_consumers(site.anchor_edge))


@register_site_predicate("tfg_polymorphic")
def _p_polymorphic(tfg: TFG, site: Site) -> bool:
    """The edge takes different shapes across the profiling inputs."""
    return tfg.is_polymorphic(site.anchor_edge)


@register_site_predicate("tfg_shape_variants_or_rank2")
def _p_shape_variants_or_rank2(tfg: TFG, site: Site) -> bool:
    """Broad gate for shape-adapter operators; ``build`` checks polymorphism."""
    spec = _spec(tfg, site.anchor_edge)
    if spec is None:
        return False
    return tfg.is_polymorphic(site.anchor_edge) or spec.rank >= 2


@register_site_predicate("tfg_residual_consumer")
def _p_residual_consumer(tfg: TFG, site: Site) -> bool:
    """The consumer participates in a bounded residual/skip structure."""
    return _is_residual(tfg, site.consumer)


@register_site_predicate("tfg_consumer_requires_broadcastable")
def _p_consumer_requires_broadcastable(tfg: TFG, site: Site) -> bool:
    """``R(v)`` of the consumer declares an internal ``shape_broadcastable``."""
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return False
    return any(req.predicate == "shape_broadcastable" for req in node.requirements)


@register_site_predicate("tfg_consumer_has_requirements")
def _p_consumer_has_requirements(tfg: TFG, site: Site) -> bool:
    """``R(v)`` of the consumer is non-empty, so a rewiring can be checked."""
    node = tfg.graph.nodes.get(site.consumer)
    return node is not None and bool(node.requirements)


# ---------------------------------------------------------------------------
# Replacement helpers
# ---------------------------------------------------------------------------

def _rewire_to(tfg: TFG, edge: str, ref: str, only: Optional[str] = None
               ) -> List[Tuple[str, int, str]]:
    """Rewire the data consumers of ``edge`` to ``ref``.

    ``only`` restricts the rewrite to a single consumer, which is how the
    branch-local (shared-edge) family changes exactly one data path.
    """
    out: List[Tuple[str, int, str]] = []
    for (consumer, position) in tfg.graph.edges[edge].consumers:
        if tfg.graph.nodes[consumer].is_param:
            continue
        if only is not None and consumer != only:
            continue
        out.append((consumer, position, ref))
    return out


def _insert_on_edge(tfg: TFG, site: Site, op: str, attrs: Optional[Dict[str, Any]] = None,
                    branch_only: bool = False, key: str = "ins") -> Optional[Replacement]:
    """Insert ``op`` on the anchor edge, preserving the tensor interface."""
    edge = site.anchor_edge
    only = site.consumer if branch_only else None
    rewires = _rewire_to(tfg, edge, f"@{key}", only=only)
    if not rewires:
        return None
    template = NodeTemplate(key=key, op=op, inputs=["{anchor}"], attrs=dict(attrs or {}))
    return Replacement(templates=[template], rewires=rewires, remove=[],
                       note=f"insert {op} on {edge}")


def _insert_after_node(tfg: TFG, site: Site, op: str,
                       attrs: Optional[Dict[str, Any]] = None) -> Optional[Replacement]:
    """Insert ``op`` directly after the matched operator, on its output edge."""
    node = tfg.graph.nodes.get(site.consumer)
    if node is None or node.num_outputs != 1:
        return None
    outs = [e for e in site.output_edges
            if e in tfg.graph.edges and tfg.graph.edges[e].producer == site.consumer]
    if not outs:
        return None
    rewires = _rewire_to(tfg, outs[0], "@ins")
    template = NodeTemplate(key="ins", op=op, inputs=[outs[0]], attrs=dict(attrs or {}))
    return Replacement(templates=[template], rewires=rewires, remove=[],
                       note=f"insert {op} after {site.consumer}")


def _bypass_node(site: Site, extra_remove: Sequence[str] = ()) -> Optional[Replacement]:
    """Delete the matched unary operator and reconnect its consumers upstream."""
    remove = [site.consumer] + [n for n in extra_remove if n != site.consumer]
    outs = [e for e in site.output_edges]
    if not outs:
        return None
    return Replacement(remove=remove, outputs={e: "{anchor}" for e in outs},
                       note=f"delete {site.consumer} and reconnect to the anchor")


def _value_bounds(tfg: TFG, edge: str) -> Optional[Dict[str, float]]:
    vp = _value(tfg, edge)
    if vp is None:
        return None
    lo, hi = float(vp.min), float(vp.max)
    if hi <= lo:
        hi = lo + 1e-6
    return {"min": lo, "max": hi}


def _profile_scale(tfg: TFG, edge: str) -> Optional[Dict[str, float]]:
    """A multiplicative perturbation derived from the profiled ``abs_mean``."""
    vp = _value(tfg, edge)
    if vp is None:
        return None
    return {"other": round(1.0 / (float(vp.abs_mean) + 1e-3), 6)}


def _build_tensor_insert_scale(tfg: TFG, site: Site) -> Optional[Replacement]:
    attrs = _profile_scale(tfg, site.anchor_edge)
    if attrs is None:
        return None
    return _insert_on_edge(tfg, site, "mul", attrs)


def _build_residual_insert_scale(tfg: TFG, site: Site) -> Optional[Replacement]:
    if not _is_residual(tfg, site.consumer):
        return None
    attrs = _profile_scale(tfg, site.anchor_edge)
    if attrs is None:
        return None
    return _insert_on_edge(tfg, site, "mul", attrs, branch_only=True)


def _build_operator_insert_activation(tfg: TFG, site: Site) -> Optional[Replacement]:
    if not site.output_edges:
        return None
    vp = _value(tfg, site.output_edges[0])
    op = "gelu" if (vp is not None and vp.mean < 0.0) else "relu"
    return _insert_after_node(tfg, site, op)


def _build_operator_insert_clamp(tfg: TFG, site: Site) -> Optional[Replacement]:
    if not site.output_edges:
        return None
    return _insert_after_node(tfg, site, "clamp", _value_bounds(tfg, site.output_edges[0]))


def _scaled(padding: Any, factor: int) -> Any:
    if isinstance(padding, (tuple, list)):
        return tuple(factor * int(p) for p in padding)
    return factor * int(padding)


def _cast_constraints() -> List[ConstraintSpec]:
    return [ConstraintSpec(
        kind="input", predicate="dtype_family", args={"position": 0, "family": "float"},
        description="an explicit dtype conversion is only inserted on float edges",
        applies_to_ops=("cast",))]


def _finite_constraints(op: str) -> List[ConstraintSpec]:
    return [ConstraintSpec(
        kind="input", predicate="value_finite", args={"position": 0},
        description=f"{op} has no defined value for NaN/Inf operands",
        applies_to_ops=(op,))]


# ---------------------------------------------------------------------------
# Tensor / Update -- value and label updates on a profiled tensor edge
# ---------------------------------------------------------------------------

def _tensor_state_update(state: str, needed: Callable[[Any], bool],
                         extra: Optional[Callable[[Any, Any], Dict[str, Any]]] = None
                         ) -> Callable[[TFG, Site], Optional[Replacement]]:
    def build(tfg: TFG, site: Site) -> Optional[Replacement]:
        spec = _spec(tfg, site.anchor_edge)
        vp = None if spec is None else spec.value
        if vp is None or not needed(vp):
            return None
        producer = tfg.graph.edges[site.anchor_edge].producer
        attrs: Dict[str, Any] = {"tensor_state": state}
        if extra is not None:
            attrs.update(extra(spec, vp))
        return Replacement(remove=[], attr_updates={producer: attrs},
                           note=f"{state} tensor state on {site.anchor_edge}")
    return build


def _tensor_label_update(label: str, value_fn: Callable[[Any], Any]
                         ) -> Callable[[TFG, Site], Optional[Replacement]]:
    def build(tfg: TFG, site: Site) -> Optional[Replacement]:
        producer = tfg.graph.edges[site.anchor_edge].producer
        spec = _spec(tfg, site.anchor_edge)
        if spec is None:
            return None
        return Replacement(remove=[], attr_updates={producer: {label: value_fn(spec)}},
                           note=f"relabel {label} on {site.anchor_edge}")
    return build


def _requires_grad_update(value: bool) -> Callable[[TFG, Site], Optional[Replacement]]:
    def build(tfg: TFG, site: Site) -> Optional[Replacement]:
        producer = tfg.graph.edges[site.anchor_edge].producer
        return Replacement(remove=[], attr_updates={producer: {"requires_grad": bool(value)}},
                           note=f"set requires_grad={value} on {site.anchor_edge}")
    return build


def _build_dtype_relabel(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Flip the declared dtype of an explicit conversion (a tensor label update)."""
    edge = tfg.graph.edges.get(site.anchor_edge)
    if edge is None:
        return None
    producer = tfg.graph.nodes.get(edge.producer)
    if producer is None:
        return None
    spec = _spec(tfg, site.anchor_edge)
    current = str(producer.attrs.get("dtype") or (spec.dtype if spec is not None else "float32"))
    new_dtype = "float16" if current != "float16" else "float32"
    return Replacement(remove=[], attr_updates={producer.id: {"dtype": new_dtype}},
                       note=f"relabel dtype of {producer.id} to {new_dtype}")


def _build_shape_adapter_insert(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Insert a size-preserving reshape in front of a contraction."""
    spec = _spec(tfg, site.anchor_edge)
    if spec is None or spec.rank < 2 or int(spec.shape[-1]) <= 0:
        return None
    return _insert_on_edge(tfg, site, "reshape", {"shape": (-1, int(spec.shape[-1]))})


# ---------------------------------------------------------------------------
# Operator / Interface / Subgraph update builders
# ---------------------------------------------------------------------------

def _consumer_attr_update(attrs_fn: Callable[[TFG, Site], Optional[Dict[str, Any]]]
                          ) -> Callable[[TFG, Site], Optional[Replacement]]:
    def build(tfg: TFG, site: Site) -> Optional[Replacement]:
        attrs = attrs_fn(tfg, site)
        if not attrs:
            return None
        return Replacement(remove=[], attr_updates={site.consumer: attrs},
                           note=f"update {site.consumer}")
    return build


def _residual_scale_update(scale: float) -> Callable[[TFG, Site], Optional[Replacement]]:
    def build(tfg: TFG, site: Site) -> Optional[Replacement]:
        if not _is_residual(tfg, site.consumer):
            return None
        return Replacement(remove=[],
                           attr_updates={site.consumer: {"branch_scales": (1.0, float(scale))}},
                           note=f"rescale the residual branch at {site.consumer}")
    return build


# --- Interface rewiring builders -------------------------------------------

def _upstream_edges(tfg: TFG, edge: str) -> List[str]:
    """Data edges produced strictly upstream of ``edge`` (cycle-free targets)."""
    upstream_nodes = tfg.upstream_of_edges([edge])
    return [e for e in tfg.graph.data_edges()
            if e != edge and tfg.graph.edges[e].producer in upstream_nodes]


def _build_interface_rewire_broadcastable(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Swap one operand of a consumer whose ``R(v)`` allows broadcasting."""
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    position = _position_of(tfg, node, site.anchor_edge)
    if position is None:
        return None
    others = [e for i, e in enumerate(node.inputs)
              if i != position and e and e in tfg.graph.edges]
    if not others:
        return None
    return Replacement(remove=[], rewires=[(site.consumer, position, others[0])],
                       note="operand swap permitted by shape_broadcastable")


def _build_interface_rewire_branch_local(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Rewire exactly one consumer of a shared edge to the producer's input."""
    edge = site.anchor_edge
    producers = tfg.graph.nodes[tfg.graph.edges[edge].producer]
    source = next((e for e in producers.inputs if e and e in tfg.graph.edges), None)
    if source is None:
        return None
    node = tfg.graph.nodes.get(site.consumer)
    position = None if node is None else _position_of(tfg, node, edge)
    if position is None:
        return None
    return Replacement(remove=[], rewires=[(site.consumer, position, source)],
                       note="branch-local producer bypass on a shared edge")


def _build_interface_rewire_dtype_preserving(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Substitute an upstream producer whose ``F(e)`` matches dtype and shape."""
    base = _spec(tfg, site.anchor_edge)
    node = tfg.graph.nodes.get(site.consumer)
    position = None if node is None else _position_of(tfg, node, site.anchor_edge)
    if base is None or position is None:
        return None
    candidates = []
    for edge in _upstream_edges(tfg, site.anchor_edge):
        spec = _spec(tfg, edge)
        if spec is None or spec.dtype != base.dtype or spec.rank != base.rank:
            continue
        candidates.append(edge)
    if not candidates:
        return None
    return Replacement(remove=[], rewires=[(site.consumer, position, sorted(candidates)[0])],
                       note="dtype/rank-preserving producer substitution")


# --- Subgraph rewiring builders --------------------------------------------

def _build_subgraph_retarget_residual(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Retarget one branch of a residual to another producer of the same subnet."""
    node = tfg.graph.nodes.get(site.consumer)
    if node is None or not _is_residual(tfg, site.consumer):
        return None
    data = _data_inputs(tfg, node)
    if len(data) < 2:
        return None
    anchor = site.anchor_edge
    if anchor in data:
        other = next(e for e in data if e != anchor)
    else:
        other = data[0]
    other_position = _position_of(tfg, node, other)
    base = _spec(tfg, other)
    if other_position is None or base is None:
        return None
    candidates = []
    for edge in _upstream_edges(tfg, other):
        spec = _spec(tfg, edge)
        if spec is None or tuple(spec.shape) != tuple(base.shape):
            continue
        candidates.append(edge)
    if not candidates:
        return None
    return Replacement(remove=[],
                       rewires=[(site.consumer, other_position, sorted(candidates)[0])],
                       note="retarget the residual/skip branch")


def _build_subgraph_branch_bypass(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Bypass the producer of a shared edge for exactly one consumer branch."""
    edge = site.anchor_edge
    producers = tfg.graph.nodes[tfg.graph.edges[edge].producer]
    source = next((e for e in producers.inputs if e and e in tfg.graph.edges), None)
    if source is None:
        return None
    others = [c for c in tfg.data_consumers(edge) if c != site.consumer]
    if not others:
        return None
    consumer = others[0]
    position = _position_of(tfg, tfg.graph.nodes[consumer], edge)
    if position is None:
        return None
    return Replacement(remove=[], rewires=[(consumer, position, source)],
                       note="short-circuit one branch of a shared edge")


def _build_subgraph_requirement_guided(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Pick a substitute producer that satisfies the consumer's ``R(v)``."""
    node = tfg.graph.nodes.get(site.consumer)
    position = None if node is None else _position_of(tfg, node, site.anchor_edge)
    if node is None or position is None:
        return None
    others = [e for i, e in enumerate(node.inputs)
              if i != position and e and e in tfg.graph.edges]
    if not others:
        return None
    other_spec = _spec(tfg, others[0])
    if other_spec is None:
        return None
    candidates = []
    for edge in _upstream_edges(tfg, site.anchor_edge):
        spec = _spec(tfg, edge)
        if spec is None or spec.rank != other_spec.rank:
            continue
        if broadcast_shape(spec.shape, other_spec.shape) is None:
            continue
        candidates.append(edge)
    if not candidates:
        return None
    return Replacement(remove=[], rewires=[(site.consumer, position, sorted(candidates)[0])],
                       note="requirement-guided producer substitution")


def _build_subgraph_polymorphic_realign(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Realign one shape variant of a polymorphic edge to a matching producer."""
    if not tfg.is_polymorphic(site.anchor_edge):
        return None
    node = tfg.graph.nodes.get(site.consumer)
    position = None if node is None else _position_of(tfg, node, site.anchor_edge)
    if position is None:
        return None
    variants = {tuple(s.shape) for s in tfg.specs_of(site.anchor_edge)}
    candidates = []
    for edge in _upstream_edges(tfg, site.anchor_edge):
        spec = _spec(tfg, edge)
        if spec is not None and tuple(spec.shape) in variants:
            candidates.append(edge)
    if not candidates:
        return None
    return Replacement(remove=[], rewires=[(site.consumer, position, sorted(candidates)[0])],
                       note="align a polymorphic edge with a concrete variant producer")


def _build_subgraph_collapse_residual(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Delete the residual merge and reconnect its consumers to the skip input."""
    node = tfg.graph.nodes.get(site.consumer)
    if node is None or node.num_outputs != 1 or not _is_residual(tfg, site.consumer):
        return None
    outs = [e for e in site.output_edges if e in tfg.graph.edges]
    if not outs:
        return None
    data = _data_inputs(tfg, node)
    skip = next((e for e in data if e != site.anchor_edge), data[0] if data else None)
    if skip is None:
        return None
    return Replacement(remove=[site.consumer], outputs={e: skip for e in outs},
                       note="collapse the residual structure onto its skip input")


def _build_subgraph_delete_norm_activation(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Delete a normalisation+activation block and reconnect the boundary."""
    consumer = tfg.graph.nodes.get(site.consumer)
    if consumer is None or consumer.num_outputs != 1:
        return None
    producer = tfg.graph.nodes.get(tfg.graph.edges[site.anchor_edge].producer)
    if producer is None:
        return None
    source = next((e for e in producer.inputs if e and e in tfg.graph.edges), None)
    if source is None:
        return None
    outs = [e for e in site.output_edges if e in tfg.graph.edges]
    if not outs:
        return None
    remove = [site.consumer]
    if producer.id not in remove:
        remove.append(producer.id)
    return Replacement(remove=remove, outputs={e: source for e in outs},
                       note="delete a bounded normalisation+activation block")


# ===========================================================================
# The 48 TFG-derived operators
# ===========================================================================

TFG_OPERATORS: List[MutationOperator] = []


def _add(name: str, target_object: str, primitive: str, pattern: Pattern,
         build: Callable[[TFG, Site], Optional[Replacement]],
         constraints: Optional[Callable[[TFG, Site], List[ConstraintSpec]]] = None,
         description: str = "", **params: Any) -> None:
    TFG_OPERATORS.append(FunctionalOperator(
        name=name, target_object=target_object, primitive=primitive, source="TFG",
        pattern=pattern, build=build, constraints=constraints, reference="TFG",
        description=description or pattern.description, **params))


# ---------------------------------------------------------------------------
# Tensor / Update (8)
# ---------------------------------------------------------------------------

_add(
    "tensor.update.profile.zero_high_zero_fraction",
    "Tensor", "Update",
    Pattern(predicate="tfg_value_profiled",
            description="a value-profiled tensor edge whose profile is mostly zeros"),
    _tensor_state_update("zero", lambda vp: vp.zero_fraction >= 0.5),
    description=("Zero out a tensor whose profiled zero_fraction is at least one half. "
                 "The sparsity is a runtime property of F(e), not visible in G, and the "
                 "edit changes only the tensor's value, not its connections."),
    feature="F(e).value.zero_fraction",
)

_add(
    "tensor.update.profile.saturate_wide_range",
    "Tensor", "Update",
    Pattern(predicate="tfg_value_profiled",
            description="a value-profiled tensor edge with an extreme dynamic range"),
    _tensor_state_update("saturate",
                         lambda vp: vp.magnitude_bin >= 2 or vp.inf_count > 0,
                         lambda spec, vp: {"sat_min": float(vp.min), "sat_max": float(vp.max)}),
    description=("Clamp a tensor to the range recorded in F(e).value when the profiled "
                 "magnitude_bin is wide or Inf values were observed. Saturating a "
                 "wide-range tensor exercises framework boundary-value handling."),
    feature="F(e).value.magnitude_bin, F(e).value.inf_count",
)

_add(
    "tensor.update.profile.denormalise_narrow_range",
    "Tensor", "Update",
    Pattern(predicate="tfg_value_profiled",
            description="a value-profiled tensor edge in a very small magnitude bin"),
    _tensor_state_update("denormal", lambda vp: vp.magnitude_bin <= -8),
    description=("Push a tensor whose profiled magnitude_bin is very small towards "
                 "subnormal values, probing denormal handling on the profiled edge."),
    feature="F(e).value.magnitude_bin",
)

_add(
    "tensor.update.profile.negate_negative_mean",
    "Tensor", "Update",
    Pattern(predicate="tfg_value_profiled",
            description="a value-profiled tensor edge with a negative profiled mean"),
    _tensor_state_update("negate", lambda vp: vp.mean < 0.0),
    description=("Negate a tensor whose F(e).value.mean is negative, flipping the "
                 "observed sign distribution without touching the graph topology."),
    feature="F(e).value.mean",
)

_add(
    "tensor.update.grad.freeze_requires_grad",
    "Tensor", "Update",
    Pattern(predicate="tfg_grad_annotated",
            description="an autograd-annotated tensor edge"),
    _requires_grad_update(False),
    description=("Relabel a profiled edge as non-differentiable (requires_grad=False). "
                 "The gradient state lives in F(e), so a plain CG cannot select this site."),
    feature="F(e).requires_grad",
)

_add(
    "tensor.update.grad.enable_requires_grad",
    "Tensor", "Update",
    Pattern(predicate="tfg_grad_annotated",
            description="an autograd-annotated tensor edge"),
    _requires_grad_update(True),
    description=("Relabel a profiled edge as differentiable (requires_grad=True), "
                 "forcing gradient tracking onto a tensor that F(e) recorded as detached."),
    feature="F(e).requires_grad",
)

_add(
    "tensor.update.label.dtype_relabel",
    "Tensor", "Update",
    Pattern(producer_op="cast",
            description="an explicit cast producer, i.e. a dtype-labelled tensor edge"),
    _build_dtype_relabel,
    description=("Change the target dtype label of an explicit conversion, updating the "
                 "tensor state recorded in F(e) while keeping the same producer and consumers."),
    feature="F(e).dtype",
)

_add(
    "tensor.update.label.layout_relabel",
    "Tensor", "Update",
    Pattern(producer_op_in=("contiguous",) + VIEW_OPS,
            description="a layout-producing edge (contiguous/permute/transpose/view)"),
    _tensor_label_update("layout", lambda spec: "channels_last" if spec.layout != "channels_last"
                         else "contiguous"),
    description=("Relabel the memory layout of a view-producing edge, changing F(e).layout "
                 "without changing the tensor's connections."),
    feature="F(e).layout",
)


# ---------------------------------------------------------------------------
# Tensor / Insertion (4)
# ---------------------------------------------------------------------------

_add(
    "tensor.insert.alias.break_storage_sharing",
    "Tensor", "Insertion",
    Pattern(require_alias=True,
            description="an edge whose F(e).alias_of records shared storage"),
    lambda tfg, site: _insert_on_edge(tfg, site, "clone"),
    description=("Insert a clone on an alias-annotated edge, breaking storage sharing "
                 "while preserving the tensor interface. This is the alias-annotated "
                 "edge + insertion construction of the paper."),
    feature="F(e).alias_of",
)

_add(
    "tensor.insert.alias.materialise_view",
    "Tensor", "Insertion",
    Pattern(require_view=True,
            description="a strided view edge (F(e).is_view)"),
    lambda tfg, site: _insert_on_edge(tfg, site, "contiguous"),
    lambda tfg, site: _finite_constraints("contiguous"),
    description=("Materialise a strided view into dense storage by inserting contiguous, "
                 "removing the view/stride state recorded in F(e)."),
    feature="F(e).is_view",
)

_add(
    "tensor.insert.grad.stop_gradient",
    "Tensor", "Insertion",
    Pattern(predicate="tfg_grad_tracked",
            description="a gradient-tracked edge (F(e).requires_grad)"),
    lambda tfg, site: _insert_on_edge(tfg, site, "detach"),
    description=("Insert a detach on a gradient-tracked edge, cutting the autograd path "
                 "at precisely the site where F(e) observed requires_grad=True."),
    feature="F(e).requires_grad",
)

_add(
    "tensor.insert.value.perturb_profiled_tensor",
    "Tensor", "Insertion",
    Pattern(predicate="tfg_value_profiled",
            description="a value-profiled tensor edge"),
    _build_tensor_insert_scale,
    description=("Insert a scale derived from the profiled magnitude F(e).value.abs_mean, "
                 "perturbing exactly this tensor's value domain."),
    feature="F(e).value.abs_mean",
)

# ---------------------------------------------------------------------------
# Operator / Update (6)
# ---------------------------------------------------------------------------

_add(
    "operator.update.norm.disable_affine",
    "Operator", "Update",
    Pattern(op_in=NORM_OPS,
            description="a normalisation operator consuming a profiled edge"),
    _consumer_attr_update(lambda tfg, site: {"affine": False, "track_running_stats": True}),
    description=("Disable the affine parameters of a normalisation operator in place, "
                 "keeping its connections so only the operator's label changes."),
    feature="F(e) identity of the normalised edge",
)

_add(
    "operator.update.activation.change_slope",
    "Operator", "Update",
    Pattern(op_in=ACTIVATION_OPS,
            description="an activation operator consuming a profiled edge"),
    _consumer_attr_update(
        lambda tfg, site: {"negative_slope":
                           0.01 if (_value(tfg, site.anchor_edge) is not None
                                    and _value(tfg, site.anchor_edge).zero_fraction > 0.5)
                           else 0.1}),
    description=("Change an activation's negative slope, choosing the value from the "
                 "profiled zero_fraction of its input edge F(e)."),
    feature="F(e).value.zero_fraction",
)

_add(
    "operator.update.pool.change_stride",
    "Operator", "Update",
    Pattern(op_in=POOL_OPS,
            description="a pooling operator consuming a profiled edge"),
    _consumer_attr_update(lambda tfg, site: {"stride": 1, "ceil_mode": True}),
    description=("Overwrite a pooling operator's stride in place, changing the operator "
                 "state while F(e) fixes the operand it sees."),
    feature="F(e).shape consumed by the pooling operator",
)

_add(
    "operator.update.softmax.change_dim",
    "Operator", "Update",
    Pattern(op="softmax", description="a softmax operator over a profiled edge"),
    _consumer_attr_update(
        lambda tfg, site: {"dim": 0 if (_spec(tfg, site.anchor_edge) is not None
                                        and _spec(tfg, site.anchor_edge).rank > 1) else -1}),
    description=("Redirect the normalisation axis of a softmax using the rank recorded in "
                 "F(e), updating the operator without changing connections."),
    feature="F(e).rank",
)

_add(
    "operator.update.conv.dilate_kernel",
    "Operator", "Update",
    Pattern(op_in=CONV_OPS, description="a convolution consuming a profiled edge"),
    _consumer_attr_update(
        lambda tfg, site: {"dilation": 2,
                           "padding": _scaled(tfg.graph.nodes[site.consumer].attrs.get("padding", 0), 2)}),
    description=("Dilate a convolution kernel in place, growing its receptive field over "
                 "the spatial extent recorded in F(e) while keeping the same operands."),
    feature="F(e).shape spatial extent",
)

_add(
    "operator.update.precision.accumulate_fp32",
    "Operator", "Update",
    Pattern(op_in=COMPUTE_OPS,
            description="a contraction/convolution consuming a profiled edge"),
    _consumer_attr_update(
        lambda tfg, site: {"accumulate_dtype":
                           "float32" if (_spec(tfg, site.anchor_edge) is not None
                                         and _spec(tfg, site.anchor_edge).dtype != "float32")
                           else "float64"}),
    description=("Change the accumulation dtype of a compute operator based on the profiled "
                 "operand dtype F(e).dtype, exercising mixed-precision accumulation."),
    feature="F(e).dtype",
)


# ---------------------------------------------------------------------------
# Operator / Insertion (3)
# ---------------------------------------------------------------------------

_add(
    "operator.insert.after.precision_clamp",
    "Operator", "Insertion",
    Pattern(op_in=COMPUTE_OPS,
            description="a compute operator whose profiled output becomes an interface"),
    _build_operator_insert_clamp,
    lambda tfg, site: _finite_constraints("clamp"),
    description=("Insert a value clamp directly after a compute operator, bounding its "
                 "result to the range profiled on the produced edge F(e)."),
    feature="F(e).value min/max of the operator output",
)

_add(
    "operator.insert.after.activation_from_profile",
    "Operator", "Insertion",
    Pattern(op_in=CONV_OPS + CONTRACTION_OPS,
            description="a compute operator whose output profile is available"),
    _build_operator_insert_activation,
    description=("Insert a non-linearity after a compute operator, picking the activation "
                 "from the profiled mean of the produced edge F(e)."),
    feature="F(e).value.mean of the operator output",
)

_add(
    "operator.insert.before.shape_adapter",
    "Operator", "Insertion",
    Pattern(op_in=("linear", "matmul", "flatten"),
            description="a contraction preceded by a profiled edge"),
    _build_shape_adapter_insert,
    description=("Insert a size-preserving reshape in front of a contraction, using the "
                 "trailing dimension recorded in F(e) to make the interface explicit."),
    feature="F(e).shape trailing dimension",
)


# ---------------------------------------------------------------------------
# Operator / Deletion (1)
# ---------------------------------------------------------------------------

_add(
    "operator.delete.redundant_activation",
    "Operator", "Deletion",
    Pattern(op_in=REMOVABLE_UNARY_OPS,
            description="a unary activation-like operator on a profiled edge"),
    lambda tfg, site: _bypass_node(site),
    description=("Delete a unary activation-like operator and reconnect its consumers to "
                 "the anchor edge, removing a differentiable stage from the data flow."),
    feature="F(e) boundary of the removed operator",
)


# ---------------------------------------------------------------------------
# Interface / Update (2)
# ---------------------------------------------------------------------------

_add(
    "interface.update.dtype.declare_explicit_dtype",
    "Interface", "Update",
    Pattern(op_in=COMPUTE_OPS + NORM_OPS,
            description="a producer-consumer interface with a profiled operand dtype"),
    _consumer_attr_update(
        lambda tfg, site: {"input_dtype": _spec(tfg, site.anchor_edge).dtype}
        if _spec(tfg, site.anchor_edge) is not None else None),
    description=("Declare the operand dtype observed in F(e) as an explicit interface "
                 "attribute of the consumer, without changing the connection."),
    feature="F(e).dtype",
)

_add(
    "interface.update.layout.declare_channels_last",
    "Interface", "Update",
    Pattern(op_in=COMPUTE_OPS + NORM_OPS + POOL_OPS, rank=(4,),
            description="a rank-4 producer-consumer interface"),
    _consumer_attr_update(lambda tfg, site: {"input_layout": "channels_last"}),
    description=("Declare a channels-last interface layout on a consumer whose operand "
                 "F(e) is rank 4, changing the interface label only."),
    feature="F(e).rank, F(e).layout",
)


# ---------------------------------------------------------------------------
# Interface / Insertion (6)
# ---------------------------------------------------------------------------

def _cast_insert(dtype: str) -> Callable[[TFG, Site], Optional[Replacement]]:
    def build(tfg: TFG, site: Site) -> Optional[Replacement]:
        if _spec(tfg, site.anchor_edge) is None:
            return None
        return _insert_on_edge(tfg, site, "cast", {"dtype": dtype})
    return build


_add(
    "interface.insert.cast.to_float16",
    "Interface", "Insertion",
    Pattern(op_in=COMPUTE_OPS + NORM_OPS,
            description="a producer-consumer interface on a float edge"),
    _cast_insert("float16"),
    lambda tfg, site: _cast_constraints(),
    description=("Make an implicit half-precision conversion explicit by inserting a cast "
                 "at the producer-consumer boundary."),
    feature="F(e).dtype",
)

_add(
    "interface.insert.cast.to_float64",
    "Interface", "Insertion",
    Pattern(op_in=COMPUTE_OPS + NORM_OPS,
            description="a producer-consumer interface on a float edge"),
    _cast_insert("float64"),
    lambda tfg, site: _cast_constraints(),
    description=("Make an implicit double-precision conversion explicit by inserting a cast "
                 "at the producer-consumer boundary."),
    feature="F(e).dtype",
)

_add(
    "interface.insert.contiguous.align_layout",
    "Interface", "Insertion",
    Pattern(predicate="tfg_layout_annotated",
            description="an interface whose F(e) records a memory layout"),
    lambda tfg, site: _insert_on_edge(tfg, site, "contiguous"),
    lambda tfg, site: _finite_constraints("contiguous"),
    description=("Insert a contiguous operator so the layout recorded in F(e) is "
                 "materialised at the interface instead of being assumed."),
    feature="F(e).layout",
)

_add(
    "interface.insert.polymorphic.pin_shape_variant",
    "Interface", "Insertion",
    Pattern(predicate="tfg_shape_variants_or_rank2",
            description="an interface edge with (possibly) several profiled shapes"),
    lambda tfg, site: _insert_on_edge(
        tfg, site, "reshape", {"shape": tuple(tfg.shapes_of(site.anchor_edge)[0])})
    if tfg.is_polymorphic(site.anchor_edge) and tfg.shapes_of(site.anchor_edge) else None,
    description=("Pin a polymorphic edge to the first shape variant observed during "
                 "profiling by inserting a reshape that is only valid for that variant."),
    feature="polymorphic F(e) shapes",
)

_add(
    "interface.insert.alias.break_sharing_at_interface",
    "Interface", "Insertion",
    Pattern(predicate="tfg_alias_or_view",
            description="an aliased/viewed edge crossing an interface"),
    lambda tfg, site: _insert_on_edge(tfg, site, "clone"),
    description=("Break storage sharing at a producer-consumer interface by inserting a "
                 "clone on the alias-annotated edge."),
    feature="F(e).alias_of",
)

_add(
    "interface.insert.clamp.value_domain_guard",
    "Interface", "Insertion",
    Pattern(predicate="tfg_value_profiled",
            description="an interface with a profiled value domain"),
    lambda tfg, site: _insert_on_edge(tfg, site, "clamp", _value_bounds(tfg, site.anchor_edge)),
    lambda tfg, site: _finite_constraints("clamp"),
    description=("Insert a value-domain guard at the interface, clamping the operand to "
                 "the range profiled in F(e)."),
    feature="F(e).value min/max",
)


# ---------------------------------------------------------------------------
# Interface / Deletion (2)
# ---------------------------------------------------------------------------

def _delete_cast_and_fold_dtype(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Delete an explicit cast and fold the conversion back into the interface.

    Deleting the cast is not enough on its own: the consumers that used to read
    the cast's output now read the cast's input directly, so the dtype they
    expect at the boundary has to be re-declared from ``F(e)``.  Recording that
    on the surviving consumers is what makes this operator a *data-flow* deletion
    rather than a plain bypass.
    """
    replacement = _bypass_node(site)
    if replacement is None:
        return None
    spec = _spec(tfg, site.anchor_edge)
    if spec is None:
        return replacement
    for (consumer, _position) in tfg.graph.edges[site.anchor_edge].consumers:
        node = tfg.graph.nodes.get(consumer)
        if node is None or node.is_param or consumer == site.consumer:
            continue
        replacement.attr_updates.setdefault(consumer, {})["input_dtype"] = spec.dtype
    if not replacement.attr_updates:
        return None
    replacement.note = (f"delete the cast {site.consumer} and re-declare "
                        f"input_dtype={spec.dtype} on the boundary consumers")
    return replacement


_add(
    "interface.delete.explicit_cast",
    "Interface", "Deletion",
    Pattern(op_in=("cast",),
            description="an explicit cast at a producer-consumer interface"),
    _delete_cast_and_fold_dtype,
    description=("Delete an explicit cast and reconnect the boundary, folding the "
                 "conversion back into the interface."),
    feature="F(e).dtype boundary",
)

_add(
    "interface.delete.shape_adapter",
    "Interface", "Deletion",
    Pattern(op_in=SHAPE_ADAPTER_OPS,
            description="an implicit shape adapter at an interface"),
    lambda tfg, site: _bypass_node(site),
    description=("Delete an implicit shape adapter (flatten/reshape/view) so the raw "
                 "producer edge reaches the consumer directly."),
    feature="F(e) shape boundary",
)


# ---------------------------------------------------------------------------
# Interface / Rewiring (3)
# ---------------------------------------------------------------------------

_add(
    "interface.rewire.broadcastable_producer",
    "Interface", "Rewiring",
    Pattern(predicate="tfg_consumer_requires_broadcastable",
            description="a consumer whose R(v) declares shape_broadcastable"),
    _build_interface_rewire_broadcastable,
    description=("Swap one operand of a consumer whose R(v) guarantees broadcasting, a "
                 "rewiring that is legal only because the requirement is declared."),
    feature="R(v).shape_broadcastable",
)

_add(
    "interface.rewire.branch_local_producer",
    "Interface", "Rewiring",
    Pattern(min_consumers=2,
            description="a shared edge with at least two data consumers"),
    _build_interface_rewire_branch_local,
    description=("Rewire exactly one consumer of a shared edge to the producer's input, "
                 "changing a single data path while the other consumers keep F(e)."),
    feature="shared edge multiplicity",
)

_add(
    "interface.rewire.dtype_preserving_substitute",
    "Interface", "Rewiring",
    Pattern(predicate="tfg_value_profiled",
            description="an interface edge with a profiled tensor state"),
    _build_interface_rewire_dtype_preserving,
    description=("Substitute the producer feeding an interface with an upstream edge whose "
                 "F(e) matches dtype and rank, preserving the profiled tensor interface."),
    feature="F(e).dtype, F(e).rank",
)


# ---------------------------------------------------------------------------
# Subgraph / Update (4)
# ---------------------------------------------------------------------------

_add(
    "subgraph.update.residual.zero_skip_branch",
    "Subgraph", "Update",
    Pattern(predicate="tfg_residual_consumer",
            description="a residual/skip merge node"),
    _residual_scale_update(0.0),
    description=("Zero one branch of a bounded residual structure by updating the merge "
                 "node's branch scales, keeping the subgraph topology intact."),
    feature="bounded residual structure around F(e)",
)

_add(
    "subgraph.update.residual.rescale_skip_branch",
    "Subgraph", "Update",
    Pattern(predicate="tfg_residual_consumer",
            description="a residual/skip merge node"),
    _residual_scale_update(0.5),
    description=("Halve the residual branch contribution in place, changing the bounded "
                 "data-flow structure's semantics without reconnecting it."),
    feature="bounded residual structure around F(e)",
)

_add(
    "subgraph.update.residual.accumulate_higher_precision",
    "Subgraph", "Update",
    Pattern(predicate="tfg_residual_consumer",
            description="a residual/skip merge node with a profiled operand dtype"),
    _consumer_attr_update(
        lambda tfg, site: {"accumulate_dtype":
                           "float64" if (_spec(tfg, site.anchor_edge) is not None
                                         and _spec(tfg, site.anchor_edge).dtype != "float64")
                           else "float32"}),
    description=("Change the accumulation precision of a bounded residual merge in place, "
                 "using the operand dtype recorded in F(e) to pick the wider type."),
    feature="F(e).dtype at the residual boundary",
)

_add(
    "subgraph.update.conv.dilate_receptive_field",
    "Subgraph", "Update",
    Pattern(op_in=CONV_OPS,
            description="a convolution whose operand carries a profiled spatial shape"),
    _consumer_attr_update(
        lambda tfg, site: {"dilation": 2,
                           "padding": _scaled(tfg.graph.nodes[site.consumer].attrs.get("padding", 0), 2)}
        if _spec(tfg, site.anchor_edge) is not None
        and _spec(tfg, site.anchor_edge).rank == 4 else None),
    description=("Dilate a convolution in place, growing the bounded subgraph's receptive "
                 "field over the rank-4 operand recorded in F(e)."),
    feature="F(e).rank, F(e).shape spatial extent",
)


# ---------------------------------------------------------------------------
# Subgraph / Insertion (3)
# ---------------------------------------------------------------------------

_add(
    "subgraph.insert.residual.inject_scaled_skip",
    "Subgraph", "Insertion",
    Pattern(predicate="tfg_residual_consumer",
            description="a residual/skip merge node"),
    _build_residual_insert_scale,
    description=("Inject a scaled identity into one branch of a residual structure, adding "
                 "an operator to the bounded data flow while preserving shapes."),
    feature="bounded residual structure around F(e)",
)

_add(
    "subgraph.insert.shared_edge.branch_local_scale",
    "Subgraph", "Insertion",
    Pattern(min_consumers=2,
            description="a shared edge with at least two data consumers"),
    lambda tfg, site: _insert_on_edge(tfg, site, "mul", {"other": 1.5}, branch_only=True),
    description=("Insert a branch-local scale on one consumer path of a shared edge, the "
                 "shared-edge + rewiring construction of the paper: the other consumers "
                 "still read F(e) unchanged."),
    feature="shared edge multiplicity",
)

_add(
    "subgraph.insert.polymorphic.reshape_variant_branch",
    "Subgraph", "Insertion",
    Pattern(predicate="tfg_shape_variants_or_rank2",
            description="an edge with (possibly) several profiled shapes"),
    lambda tfg, site: _insert_on_edge(
        tfg, site, "reshape", {"shape": tuple(tfg.shapes_of(site.anchor_edge)[0])},
        branch_only=True)
    if tfg.is_polymorphic(site.anchor_edge) and tfg.shapes_of(site.anchor_edge) else None,
    description=("Insert a branch-local reshape on a polymorphic edge that is only valid "
                 "for the first observed shape variant, exposing variant-dependent bugs."),
    feature="polymorphic F(e) shapes",
)


# ---------------------------------------------------------------------------
# Subgraph / Deletion (2)
# ---------------------------------------------------------------------------

_add(
    "subgraph.delete.residual.collapse_skip",
    "Subgraph", "Deletion",
    Pattern(predicate="tfg_residual_consumer",
            description="a residual/skip merge node"),
    _build_subgraph_collapse_residual,
    description=("Delete the residual merge and reconnect its consumers to the skip input, "
                 "collapsing the bounded data-flow structure."),
    feature="bounded residual structure around F(e)",
)

_add(
    "subgraph.delete.block.norm_activation_pair",
    "Subgraph", "Deletion",
    Pattern(op_in=REMOVABLE_UNARY_OPS, producer_op_in=NORM_OPS,
            description="a normalisation+activation block inside a feature subgraph"),
    _build_subgraph_delete_norm_activation,
    description=("Delete a bounded normalisation+activation block and reconnect the "
                 "boundary to the block's input edge."),
    feature="bounded normalisation/activation structure",
)


# ---------------------------------------------------------------------------
# Subgraph / Rewiring (4)
# ---------------------------------------------------------------------------

_add(
    "subgraph.rewire.residual.retarget_skip",
    "Subgraph", "Rewiring",
    Pattern(predicate="tfg_residual_consumer",
            description="a residual/skip merge node"),
    _build_subgraph_retarget_residual,
    description=("Retarget one branch of a residual structure to another producer in the "
                 "same subnet, preserving the profiled shape of the branch."),
    feature="bounded residual structure around F(e)",
)

_add(
    "subgraph.rewire.shared_edge.branch_bypass",
    "Subgraph", "Rewiring",
    Pattern(min_consumers=2,
            description="a shared edge with at least two data consumers"),
    _build_subgraph_branch_bypass,
    description=("Short-circuit exactly one consumer branch of a shared edge to the "
                 "producer's input, changing one data path inside the bounded region."),
    feature="shared edge multiplicity",
)

_add(
    "subgraph.rewire.requirement_guided_producer",
    "Subgraph", "Rewiring",
    Pattern(predicate="tfg_consumer_has_requirements",
            description="a consumer with a non-empty requirement set R(v)"),
    _build_subgraph_requirement_guided,
    description=("Substitute an upstream producer that satisfies the consumer's R(v) "
                 "(rank and broadcastability), i.e. a rewiring justified by requirements."),
    feature="R(v) of the consumer",
)

_add(
    "subgraph.rewire.polymorphic.variant_realign",
    "Subgraph", "Rewiring",
    Pattern(predicate="tfg_polymorphic",
            description="an edge whose shape varies across the profiling inputs"),
    _build_subgraph_polymorphic_realign,
    description=("Realign a polymorphic edge onto a producer that matches one concrete "
                 "shape variant observed during profiling."),
    feature="polymorphic F(e) shapes",
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def tfg_source_count() -> int:
    """Number of operators this module contributes to the pool."""
    return len(TFG_OPERATORS)


__all__ = ["TFG_OPERATORS", "tfg_source_count"]
