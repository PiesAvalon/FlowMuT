"""CG-derived mutation operators: prior framework-testing rules mapped to the TFG.

This module adapts concrete mutation rules published by earlier deep-learning
framework testing studies (LEMON, COMET, DevMuT, DeepHunter, FreeFuzz,
TitanFuzz, Audee, MUFFIN, GANDALF, D3, ORTHRUS, MoCo, ...) to the FlowMuT
abstraction: every rule becomes a :class:`~flowmut.operators.base.FunctionalOperator`
with

* a :class:`~flowmut.operators.base.Pattern` stating where it applies (anchored
  at a tensor edge, matching the node that *consumes* that edge),
* a :class:`~flowmut.operators.base.Replacement` stating how the matched bounded
  subgraph is rewritten, and
* optional :class:`~flowmut.operators.base.ConstraintSpec` records adding an
  operator-specific requirement on top of the automatically re-evaluated
  ``R(v)``.

The pool follows the paper's operator-construction table exactly.  Concrete
operators are produced by parameterised families (activation swaps, convolution
and pooling hyper-parameters, normalisation parameters, boundary adapters,
residual/block edits, ...); the ``params`` dictionary of each instance carries
the parameters that distinguish it from its siblings, which is what the pool's
deduplication key uses.

The 120 operators are distributed as::

    Tensor/Update        5
    Tensor/Insertion     1
    Operator/Update     37
    Operator/Insertion   3
    Operator/Deletion    1
    Interface/Update    43
    Interface/Insertion  0
    Interface/Deletion   1
    Interface/Rewiring   0
    Subgraph/Update     15
    Subgraph/Insertion   9
    Subgraph/Deletion    1
    Subgraph/Rewiring    4
    ----------------------
    TOTAL              120

Implementation conventions
--------------------------
* ``Replacement(remove=[])`` means "nothing is deleted"; leaving ``remove`` at
  ``None`` deletes the site's node(s).
* An *update* of a surviving node is expressed with ``attr_updates`` and
  ``keep=[site.consumer]``.
* An *insertion* on the anchor edge inserts a template consuming
  ``"{anchor}"`` and rewires every data consumer of the anchor edge to the last
  template output.
* A *deletion* lists the removed node in ``remove`` and reconnects the boundary
  through ``outputs={old_output_edge: site.anchor_edge}``.
* A *rewiring* keeps every node and only moves producer/consumer connections
  through ``rewires=[(consumer, position, edge)]``.

Because the framework's no-op detector compares node and edge counts (and
in-place attribute updates), the two count-neutral edits -- a pure rewiring and
an op swap -- additionally record their edit in the attributes of the node that
survives them (``rewired_*`` / ``activation``).  The recorded attribute is part
of the mutation description, not a substitute for it: the rewiring really moves
the connection and the swap really replaces the operator.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from flowmut.ir.ops import canonical_op
from flowmut.operators.base import (
    ConstraintSpec,
    FunctionalOperator,
    MutationOperator,
    NodeTemplate,
    Pattern,
    Replacement,
    Site,
)
from flowmut.tfg.tfg import TFG

# ---------------------------------------------------------------------------
# Operator vocabulary used by the patterns
# ---------------------------------------------------------------------------

ACTIVATION_OPS: Tuple[str, ...] = (
    "relu", "gelu", "silu", "sigmoid", "tanh", "leaky_relu", "elu", "mish",
    "hardswish", "hardsigmoid", "quick_gelu", "softplus",
)
NORM_OPS: Tuple[str, ...] = (
    "batchnorm", "layernorm", "groupnorm", "instancenorm", "rmsnorm", "l2norm",
)
POOL_OPS: Tuple[str, ...] = (
    "maxpool2d", "avgpool2d", "adaptive_avgpool2d", "adaptive_maxpool2d",
    "global_avgpool", "global_maxpool",
)
REDUCE_OPS: Tuple[str, ...] = (
    "mean", "sum", "amax", "amin", "var", "std", "prod", "argmax", "argmin",
)
BINARY_OPS: Tuple[str, ...] = (
    "add", "sub", "mul", "div", "pow", "maximum", "minimum",
)
ATTENTION_OPS: Tuple[str, ...] = ("sdpa", "scaled_dot_product_attention")
SOFTMAX_OPS: Tuple[str, ...] = ("softmax", "log_softmax")
#: Unary operators that a deletion operator may remove and bypass.
REMOVABLE_UNARY_OPS: Tuple[str, ...] = (
    "dropout", "identity", "clone", "cast", "contiguous", "detach",
    "stop_gradient", "neg", "abs", "square", "relu", "gelu", "silu",
    "leaky_relu", "elu", "mish", "hardswish", "sigmoid", "tanh",
)
#: Boundary adapters an Interface/Deletion operator may remove.
ADAPTER_OPS: Tuple[str, ...] = (
    "cast", "reshape", "view", "flatten", "contiguous", "clone", "permute",
    "transpose", "squeeze", "unsqueeze", "identity", "detach", "stop_gradient",
)

#: Generic pattern: any tensor edge (produced by a non-parameter node) whose
#: consumer is a data operator.
ANY_EDGE = Pattern(description="any tensor edge feeding a data operator")


# ---------------------------------------------------------------------------
# Small graph helpers
# ---------------------------------------------------------------------------

def _numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape or ():
        n *= int(d)
    return int(n)


def _spec_of(tfg: TFG, edge: str):
    try:
        return tfg.f(edge)
    except Exception:  # pragma: no cover - defensive
        return None


def _shape_of(tfg: TFG, edge: str) -> Optional[Tuple[int, ...]]:
    spec = _spec_of(tfg, edge)
    if spec is None or not spec.shape:
        return None
    return tuple(int(d) for d in spec.shape)


def _anchor_consumers(tfg: TFG, site: Site) -> List[Tuple[str, int]]:
    """``(consumer, input_position)`` pairs reading the anchor edge (no params)."""
    edge = tfg.graph.edges.get(site.anchor_edge)
    if edge is None:
        return []
    out: List[Tuple[str, int]] = []
    for (consumer, position) in edge.consumers:
        node = tfg.graph.nodes.get(consumer)
        if node is None or node.is_param:
            continue
        out.append((consumer, position))
    return out


def _anchor_position(tfg: TFG, site: Site) -> int:
    edge = tfg.graph.edges.get(site.anchor_edge)
    if edge is None:
        return 0
    for (consumer, position) in edge.consumers:
        if consumer == site.consumer:
            return position
    return 0


def _data_inputs(tfg: TFG, node) -> List[str]:
    out: List[str] = []
    for edge_id in node.inputs:
        if not edge_id or edge_id not in tfg.graph.edges:
            continue
        if tfg.graph.nodes[tfg.graph.edges[edge_id].producer].is_param:
            continue
        out.append(edge_id)
    return out


def _param_weight_of(tfg: TFG, node, position: int):
    """Spec of the parameter edge feeding ``node`` at ``position`` (or None)."""
    if position >= len(node.inputs):
        return None
    edge_id = node.inputs[position]
    if not edge_id or edge_id not in tfg.graph.edges:
        return None
    if not tfg.graph.nodes[tfg.graph.edges[edge_id].producer].is_param:
        return None
    return _spec_of(tfg, edge_id)


def _channels_of(tfg: TFG, edge: str) -> Optional[int]:
    shape = _shape_of(tfg, edge)
    if shape is None or len(shape) < 2:
        return None
    channels = int(shape[1])
    return channels if channels > 0 else None


# ---------------------------------------------------------------------------
# Replacement builders
# ---------------------------------------------------------------------------

def _insert_chain(tfg: TFG, site: Site,
                  specs: Sequence[Dict[str, Any]]) -> Optional[Replacement]:
    """Insert ``specs`` on the anchor edge and rewire its consumers to the tail.

    Each spec is ``{"key", "op", "inputs"?, "attrs"?}``; ``inputs`` defaults to
    ``["{anchor}"]`` so a one-element ``specs`` is the ordinary insertion.
    """
    consumers = _anchor_consumers(tfg, site)
    if not consumers:
        return None
    templates: List[NodeTemplate] = []
    for spec in specs:
        inputs = [str(ref) for ref in spec.get("inputs", ["{anchor}"])]
        if any(not ref for ref in inputs):
            return None
        templates.append(NodeTemplate(
            key=str(spec["key"]), op=str(spec["op"]), inputs=inputs,
            attrs=dict(spec.get("attrs", {})), scope=str(spec.get("scope", ""))))
    if not templates:
        return None
    tail = templates[-1].key
    return Replacement(
        templates=templates, remove=[], keep=[],
        rewires=[(consumer, position, f"@{tail}")
                 for (consumer, position) in consumers])


def _insert_one(op: str, attrs: Optional[Dict[str, Any]] = None
                ) -> List[Dict[str, Any]]:
    return [{"key": "new", "op": op, "inputs": ["{anchor}"],
             "attrs": dict(attrs or {})}]


def _insert_builder(specs):
    """Wrap a static spec list or a ``(tfg, site) -> spec list`` callable."""
    if callable(specs):
        def build(tfg: TFG, site: Site) -> Optional[Replacement]:
            resolved = specs(tfg, site)
            if not resolved:
                return None
            return _insert_chain(tfg, site, resolved)
    else:
        frozen = list(specs)

        def build(tfg: TFG, site: Site) -> Optional[Replacement]:
            return _insert_chain(tfg, site, frozen)
    return build


def _consumer_update(tfg: TFG, site: Site, attrs: Dict[str, Any]
                     ) -> Optional[Replacement]:
    """In-place update of the matched node, skipping updates that change nothing."""
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    changed = {key: value for key, value in attrs.items()
               if node.attrs.get(key) != value}
    if not changed:
        return None
    return Replacement(remove=[], keep=[site.consumer],
                       attr_updates={site.consumer: changed})


def _consumer_update_builder(attrs):
    if callable(attrs):
        def build(tfg: TFG, site: Site) -> Optional[Replacement]:
            resolved = attrs(tfg, site)
            if not resolved:
                return None
            return _consumer_update(tfg, site, resolved)
    else:
        frozen = dict(attrs)

        def build(tfg: TFG, site: Site) -> Optional[Replacement]:
            return _consumer_update(tfg, site, frozen)
    return build


def _insert_after_consumer(tfg: TFG, site: Site, op: str,
                           attrs: Optional[Dict[str, Any]] = None
                           ) -> Optional[Replacement]:
    """Insert ``op`` on the (single) output edge of the anchor consumer."""
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    outputs = list(site.output_edges)
    if len(outputs) != 1:
        return None
    edge_id = outputs[0]
    edge = tfg.graph.edges.get(edge_id)
    if edge is None:
        return None
    consumers = [(c, p) for (c, p) in edge.consumers
                 if not tfg.graph.nodes[c].is_param]
    if not consumers:
        return None
    return Replacement(
        templates=[NodeTemplate(key="new", op=op, inputs=[edge_id],
                                attrs=dict(attrs or {}))],
        remove=[], keep=[site.consumer],
        rewires=[(consumer, position, "@new")
                 for (consumer, position) in consumers])


def _insert_after_builder(op: str, attrs: Optional[Dict[str, Any]] = None):
    def build(tfg: TFG, site: Site) -> Optional[Replacement]:
        return _insert_after_consumer(tfg, site, op, attrs)
    return build


def _delete_consumer(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Delete the matched unary operator and reconnect its consumers upstream."""
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    outputs = list(site.output_edges)
    if not outputs:
        return None
    if any(edge in tfg.graph.outputs for edge in outputs):
        return None
    return Replacement(remove=[site.consumer],
                       outputs={edge: site.anchor_edge for edge in outputs})


def _activation_swap_builder(to_op: str, from_op: str):
    """Replace the matched activation and record it on the producing block."""
    def build(tfg: TFG, site: Site) -> Optional[Replacement]:
        node = tfg.graph.nodes.get(site.consumer)
        if node is None or canonical_op(node.op) != from_op:
            return None
        outputs = list(site.output_edges)
        if not outputs:
            return None
        if any(edge in tfg.graph.outputs for edge in outputs):
            return None
        inputs = [edge for edge in node.inputs if edge]
        if len(inputs) != len(node.inputs) or not inputs:
            return None
        producer = tfg.graph.edges[site.anchor_edge].producer
        return Replacement(
            templates=[NodeTemplate(key="act", op=to_op, inputs=inputs,
                                    attrs=dict(node.attrs))],
            outputs={edge: "@act" for edge in outputs},
            remove=[site.consumer],
            attr_updates={producer: {"activation": to_op,
                                     "activation_replaced": from_op}},
            note=f"replace {from_op} by {to_op}")
    return build


def _repeat_consumer(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Deepen a repeated pattern: apply the matched operator one more time.

    Parameter operands of the matched node are *shared* with the repeated copy
    (weight tying), which keeps the repeated block's tensor states inferable from
    the profiled annotation of the original parameters.
    """
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    outputs = list(site.output_edges)
    if len(outputs) != 1:
        return None
    edge = tfg.graph.edges.get(outputs[0])
    if edge is None:
        return None
    consumers = [(c, p) for (c, p) in edge.consumers
                 if not tfg.graph.nodes[c].is_param]
    if not consumers:
        return None
    inputs: List[str] = []
    for edge_id in node.inputs:
        if not edge_id or edge_id not in tfg.graph.edges:
            return None
        inputs.append(edge_id)
    template = NodeTemplate(key="repeat", op=node.op, inputs=inputs,
                            attrs=dict(node.attrs),
                            scope=f"{(node.scope or 'block')}.repeat")
    return Replacement(
        templates=[template], remove=[], keep=[site.consumer],
        rewires=[(consumer, position, "@repeat")
                 for (consumer, position) in consumers])


def _make(name: str, target_object: str, primitive: str, pattern: Pattern,
          build: Callable[[TFG, Site], Optional[Replacement]],
          reference: str = "", description: str = "",
          constraints: Optional[Callable[[TFG, Site], List[ConstraintSpec]]] = None,
          **params: Any) -> MutationOperator:
    return FunctionalOperator(
        name=name, target_object=target_object, primitive=primitive, source="CG",
        pattern=pattern, build=build, constraints=constraints,
        reference=reference,
        description=description or pattern.description, **params)


# ===========================================================================
# Tensor / Update  (5)
# ===========================================================================

def tensor_update_operators() -> List[MutationOperator]:
    """Change a recorded tensor state without touching the topology shape."""
    ops: List[MutationOperator] = []

    # -- dtype ------------------------------------------------------------
    for dtype, reference in (("float16", "FreeFuzz"), ("bfloat16", "TitanFuzz")):
        ops.append(_make(
            f"tensor.update.dtype.{dtype}", "Tensor", "Update", ANY_EDGE,
            _insert_builder(_insert_one("cast", {"dtype": dtype})),
            reference=reference,
            description=f"cast the tensor carried by the anchor edge to {dtype}",
            dtype=dtype, edit="dtype_cast"))

    # -- layout / contiguity ---------------------------------------------
    ops.append(_make(
        "tensor.update.layout.contiguous", "Tensor", "Update", ANY_EDGE,
        _insert_builder(_insert_one("contiguous")),
        reference="COMET",
        description="materialise the anchor tensor as a contiguous buffer",
        layout="contiguous", edit="layout"))

    # -- value distribution: zeroed tensor --------------------------------
    ops.append(_make(
        "tensor.update.value.zero", "Tensor", "Update", ANY_EDGE,
        _insert_builder(_insert_one("clamp", {"min": 0.0, "max": 0.0})),
        reference="DeepHunter",
        description="collapse the anchor tensor to the zero value distribution",
        value="zero", edit="value_distribution"))

    # -- gradient state ----------------------------------------------------
    ops.append(_make(
        "tensor.update.grad.detach", "Tensor", "Update", ANY_EDGE,
        _insert_builder(_insert_one("detach")),
        reference="DevMuT",
        description="detach the anchor tensor from the autograd graph",
        grad="detached", edit="gradient_state",
        constraints=lambda tfg, site: [
            ConstraintSpec(kind="input", predicate="layout_contiguous",
                           args={"position": 0},
                           description="a tensor must be materialised before it is detached")]))

    return ops


# ===========================================================================
# Tensor / Insertion  (1)
# ===========================================================================

def tensor_insertion_operators() -> List[MutationOperator]:
    """Insert a defensive copy on the anchor edge, breaking aliasing."""
    return [_make(
        "tensor.insertion.clone", "Tensor", "Insertion", ANY_EDGE,
        _insert_builder(_insert_one("clone")),
        reference="DevMuT",
        description="insert a defensive copy on the anchor edge so consumers "
                    "stop sharing storage with the producer",
        edit="clone", breaks_alias=True)]


# ===========================================================================
# Operator / Update  (37)
# ===========================================================================

_ACTIVATION_SWAPS: Tuple[str, ...] = (
    "gelu", "silu", "leaky_relu", "elu", "mish", "hardswish", "sigmoid",
    "tanh", "quick_gelu",
)

_CONV_UPDATES: Tuple[Tuple[str, Any, str, str], ...] = (
    ("stride", 2, "stride_2", "COMET"),
    ("stride", 3, "stride_3", "COMET"),
    ("padding", 0, "padding_0", "LEMON"),
    ("dilation", 2, "dilation_2", "DeepHunter"),
    ("dilation", 3, "dilation_3", "DeepHunter"),
    ("groups", 2, "groups_2", "COMET"),
)

_POOL_UPDATES: Tuple[Tuple[str, Any, str, str], ...] = (
    ("kernel_size", 3, "kernel_size_3", "LEMON"),
    ("stride", 1, "stride_1", "COMET"),
    ("padding", 1, "padding_1", "DeepHunter"),
    ("ceil_mode", True, "ceil_mode_true", "Audee"),
    ("count_include_pad", False, "count_include_pad_false", "Audee"),
)

_NORM_UPDATES: Tuple[Tuple[str, Any, str, str], ...] = (
    ("eps", 1e-3, "eps_1e-3", "COMET"),
    ("eps", 1e-5, "eps_1e-5", "COMET"),
    ("momentum", 0.01, "momentum_0_01", "LEMON"),
    ("affine", False, "affine_false", "DeepHunter"),
    ("track_running_stats", False, "track_running_stats_false", "DeepHunter"),
)


def operator_update_operators() -> List[MutationOperator]:
    ops: List[MutationOperator] = []

    # -- activations: swap the applied non-linearity (9) -------------------
    for to_op in _ACTIVATION_SWAPS:
        ops.append(_make(
            f"operator.update.activation.relu_to_{to_op}", "Operator", "Update",
            Pattern(op="relu", description="matmul/batch-norm output activated by ReLU"),
            _activation_swap_builder(to_op, "relu"),
            reference="LEMON",
            description=f"replace the ReLU activation applied to the anchor "
                        f"tensor by {to_op}",
            from_op="relu", to_op=to_op, edit="activation"))

    # -- convolution hyper-parameters (7) ---------------------------------
    for attr, value, label, reference in _CONV_UPDATES:
        pattern = Pattern(op="conv2d",
                          description="2-D convolution configuration")
        ops.append(_make(
            f"operator.update.conv2d.{label}", "Operator", "Update", pattern,
            _consumer_update_builder({attr: value}),
            reference=reference,
            description=f"change the convolution {attr} to {value!r}",
            attr=attr, value=value, edit="conv_hyperparameter"))
    ops.append(_make(
        "operator.update.conv2d.out_channels_double", "Operator", "Update",
        Pattern(op="conv2d", description="2-D convolution configuration"),
        _consumer_update_builder(_double_conv_channels),
        reference="COMET",
        description="double the convolution output channel count",
        attr="out_channels", factor=2, edit="conv_width"))

    # -- pooling hyper-parameters (5) -------------------------------------
    pool_pattern = Pattern(op_in=POOL_OPS, description="spatial pooling operator")
    for attr, value, label, reference in _POOL_UPDATES:
        ops.append(_make(
            f"operator.update.pool.{label}", "Operator", "Update", pool_pattern,
            _consumer_update_builder({attr: value}),
            reference=reference,
            description=f"change the pooling {attr} to {value!r}",
            attr=attr, value=value, edit="pool_hyperparameter"))

    # -- normalisation parameters (6) -------------------------------------
    norm_pattern = Pattern(op="batchnorm",
                           description="batch normalisation operator")
    for attr, value, label, reference in _NORM_UPDATES:
        ops.append(_make(
            f"operator.update.batchnorm.{label}", "Operator", "Update",
            norm_pattern, _consumer_update_builder({attr: value}),
            reference=reference,
            description=f"change the batch-normalisation {attr} to {value!r}",
            attr=attr, value=value, edit="norm_parameter"))
    ops.append(_make(
        "operator.update.batchnorm.num_features_off_by_one", "Operator", "Update",
        norm_pattern, _consumer_update_builder(_batchnorm_off_by_one),
        reference="COMET",
        description="declare one more feature map than the operand really has",
        attr="num_features", delta=1, edit="norm_parameter"))

    # -- linear layer configuration (3) -----------------------------------
    linear_pattern = Pattern(op="linear", description="fully connected operator")
    ops.append(_make(
        "operator.update.linear.bias_false", "Operator", "Update",
        linear_pattern, _consumer_update_builder({"bias": False}),
        reference="LEMON",
        description="drop the additive bias term of the linear operator",
        attr="bias", value=False, edit="linear_bias"))
    ops.append(_make(
        "operator.update.linear.out_features_double", "Operator", "Update",
        linear_pattern, _consumer_update_builder(_double_linear_features),
        reference="COMET",
        description="double the number of output features of the linear operator",
        attr="out_features", factor=2, edit="linear_width"))
    ops.append(_make(
        "operator.update.linear.weight_transposed", "Operator", "Update",
        linear_pattern, _consumer_update_builder({"transposed": True}),
        reference="DeepHunter",
        description="declare the linear weight as transposed",
        attr="transposed", value=True, edit="linear_layout"))

    # -- reductions (2) ----------------------------------------------------
    reduce_pattern = Pattern(op_in=REDUCE_OPS, description="reduction operator")
    ops.append(_make(
        "operator.update.reduce.keepdim_flip", "Operator", "Update",
        reduce_pattern, _consumer_update_builder(_keepdim_flip),
        reference="Audee",
        description="flip whether the reduced dimension is kept",
        attr="keepdim", edit="reduce_dim"))
    ops.append(_make(
        "operator.update.reduce.dim_zero", "Operator", "Update",
        reduce_pattern, _consumer_update_builder({"dim": 0}),
        reference="Audee",
        description="reduce over the leading dimension instead of the default",
        attr="dim", value=0, edit="reduce_dim"))

    # -- softmax axis (2) --------------------------------------------------
    softmax_pattern = Pattern(op_in=SOFTMAX_OPS,
                              description="normalising exponential operator")
    ops.append(_make(
        "operator.update.softmax.dim_zero", "Operator", "Update",
        softmax_pattern, _consumer_update_builder({"dim": 0}),
        reference="LEMON",
        description="normalise over the leading dimension",
        attr="dim", value=0, edit="softmax_axis"))
    ops.append(_make(
        "operator.update.softmax.dim_one", "Operator", "Update",
        softmax_pattern, _consumer_update_builder({"dim": 1}),
        reference="LEMON",
        description="normalise over the channel dimension",
        attr="dim", value=1, edit="softmax_axis"))

    # -- dropout probability (1) ------------------------------------------
    ops.append(_make(
        "operator.update.dropout.p_0_9", "Operator", "Update",
        Pattern(op="dropout", description="dropout regularisation operator"),
        _consumer_update_builder({"p": 0.9}),
        reference="DeepHunter",
        description="raise the dropout probability to 0.9",
        attr="p", value=0.9, edit="dropout_rate"))

    # -- interpolation (1) -------------------------------------------------
    ops.append(_make(
        "operator.update.interpolate.mode_flip", "Operator", "Update",
        Pattern(op="interpolate", description="resampling operator"),
        _consumer_update_builder(_interpolate_mode_flip),
        reference="COMET",
        description="switch the resampling mode and its align_corners setting",
        attr="mode", edit="interpolation"))

    # -- attention (1) -----------------------------------------------------
    ops.append(_make(
        "operator.update.attention.dropout_0_1", "Operator", "Update",
        Pattern(op_in=ATTENTION_OPS, description="scaled dot-product attention"),
        _consumer_update_builder({"dropout_p": 0.1}),
        reference="DevMuT",
        description="enable attention dropout with probability 0.1",
        attr="dropout_p", value=0.1, edit="attention"))

    return ops


def _double_conv_channels(tfg: TFG, site: Site) -> Optional[Dict[str, Any]]:
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    weight = _param_weight_of(tfg, node, 1)
    if weight is None or not weight.shape:
        return None
    return {"out_channels": max(1, int(weight.shape[0]) * 2)}


def _double_linear_features(tfg: TFG, site: Site) -> Optional[Dict[str, Any]]:
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    weight = _param_weight_of(tfg, node, 1)
    if weight is None or not weight.shape:
        return None
    return {"out_features": max(1, int(weight.shape[0]) * 2)}


def _batchnorm_off_by_one(tfg: TFG, site: Site) -> Optional[Dict[str, Any]]:
    channels = _channels_of(tfg, site.anchor_edge)
    if channels is None:
        return None
    return {"num_features": channels + 1}


def _keepdim_flip(tfg: TFG, site: Site) -> Optional[Dict[str, Any]]:
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    return {"keepdim": not bool(node.attrs.get("keepdim", False))}


def _interpolate_mode_flip(tfg: TFG, site: Site) -> Optional[Dict[str, Any]]:
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    mode = str(node.attrs.get("mode", "nearest"))
    if mode in ("bilinear", "bicubic", "linear", "trilinear"):
        return {"mode": "nearest", "align_corners": None}
    return {"mode": "bilinear", "align_corners": True}


# ===========================================================================
# Operator / Insertion  (3)
# ===========================================================================

def operator_insertion_operators() -> List[MutationOperator]:
    """Insert an identity-preserving operator after the matched consumer."""
    return [
        _make("operator.insertion.identity_after", "Operator", "Insertion",
              ANY_EDGE, _insert_after_builder("identity"),
              reference="DeepHunter",
              description="insert an identity operator after the matched consumer",
              inserted="identity", edit="identity_insertion"),
        _make("operator.insertion.dropout_after", "Operator", "Insertion",
              ANY_EDGE, _insert_after_builder("dropout", {"p": 0.5}),
              reference="LEMON",
              description="insert a dropout layer after the matched consumer",
              inserted="dropout", edit="regularisation_insertion"),
        _make("operator.insertion.clone_after", "Operator", "Insertion",
              ANY_EDGE, _insert_after_builder("clone"),
              reference="DevMuT",
              description="insert a defensive clone after the matched consumer",
              inserted="clone", edit="clone_insertion"),
    ]


# ===========================================================================
# Operator / Deletion  (1)
# ===========================================================================

def operator_deletion_operators() -> List[MutationOperator]:
    """Delete a removable elementwise operator and reconnect its consumers."""
    return [_make(
        "operator.deletion.elementwise", "Operator", "Deletion",
        Pattern(op_in=REMOVABLE_UNARY_OPS,
                description="elementwise operator whose output shape matches its input"),
        _delete_consumer,
        reference="DeepHunter",
        description="delete the matched elementwise operator (dropout, identity, "
                    "bias-like activation) and reconnect its consumers upstream",
        edit="remove_elementwise")]


# ===========================================================================
# Interface / Update  (43)
# ===========================================================================

_CAST_DTYPES: Tuple[Tuple[str, str], ...] = (
    ("float16", "FreeFuzz"),
    ("bfloat16", "TitanFuzz"),
    ("float64", "FreeFuzz"),
    ("int64", "FreeFuzz"),
    ("int32", "DeepHunter"),
    ("int16", "DeepHunter"),
    ("int8", "DevMuT"),
    ("uint8", "DevMuT"),
    ("bool", "MUFFIN"),
    ("complex64", "COMET"),
)

_LAYOUT_ADAPTERS: Tuple[Tuple[str, str], ...] = (
    ("contiguous", "COMET"),
    ("permute_reverse", "LEMON"),
    ("transpose_trailing", "DeepHunter"),
    ("channels_last", "COMET"),
    ("strided_view", "LEMON"),
    ("reshape_flat", "MUFFIN"),
)

_SHAPE_ADAPTERS: Tuple[Tuple[str, str], ...] = (
    ("unsqueeze_dim0", "LEMON"),
    ("unsqueeze_dim1", "LEMON"),
    ("squeeze_dim0", "Audee"),
    ("squeeze_last", "Audee"),
    ("flatten_all", "MUFFIN"),
    ("reshape_rank2", "DeepHunter"),
    ("getitem_dim0", "GANDALF"),
    ("expand_last", "COMET"),
)

_VALUE_ADAPTERS: Tuple[Tuple[str, str, str], ...] = (
    ("abs", "absolute value", "DeepHunter"),
    ("neg", "sign flip", "DeepHunter"),
    ("square", "quadratic amplification", "Audee"),
    ("reciprocal", "reciprocal blow-up", "MUFFIN"),
    ("exp", "exponential overflow", "COMET"),
    ("log", "logarithmic domain mismatch", "COMET"),
)

_GRAD_ADAPTERS: Tuple[Tuple[str, str], ...] = (
    ("detach", "DevMuT"),
    ("stop_gradient", "DevMuT"),
    ("clone", "DeepHunter"),
    ("detach_then_clone", "COMET"),
    ("clone_then_detach", "COMET"),
)


def interface_update_operators() -> List[MutationOperator]:
    """Change the producer-consumer boundary (dtype, layout, shape, value, grad)."""
    ops: List[MutationOperator] = []

    # -- dtype mismatch at the boundary (10) -------------------------------
    for dtype, reference in _CAST_DTYPES:
        ops.append(_make(
            f"interface.update.dtype.cast_to_{dtype}", "Interface", "Update",
            ANY_EDGE,
            _insert_builder(_insert_one("cast", {"dtype": dtype})),
            reference=reference,
            description=f"make the boundary conversion to {dtype} explicit by "
                        f"inserting a cast between producer and consumer",
            dtype=dtype, edit="boundary_dtype"))

    # -- layout mismatch at the boundary (6) ------------------------------
    for kind, reference in _LAYOUT_ADAPTERS:
        ops.append(_make(
            f"interface.update.layout.{kind}", "Interface", "Update", ANY_EDGE,
            _insert_builder(_layout_specs(kind)),
            reference=reference,
            description=f"insert an explicit {kind} adapter on the boundary",
            layout=kind, edit="boundary_layout"))

    # -- rank/shape mismatch at the boundary (8) --------------------------
    for kind, reference in _SHAPE_ADAPTERS:
        ops.append(_make(
            f"interface.update.shape.{kind}", "Interface", "Update", ANY_EDGE,
            _insert_builder(_shape_specs(kind)),
            reference=reference,
            description=f"insert a {kind} adapter so the consumer sees a "
                        f"different rank/shape",
            shape=kind, edit="boundary_shape"))

    # -- value-domain mismatch at the boundary (6) -------------------------
    for op, label, reference in _VALUE_ADAPTERS:
        constraints = None
        if op == "log":
            constraints = _log_domain_constraint
        ops.append(_make(
            f"interface.update.range.{op}", "Interface", "Update", ANY_EDGE,
            _insert_builder(_insert_one(op)),
            reference=reference,
            description=f"insert {op} on the boundary ({label}) so the consumer "
                        f"observes an out-of-domain value range",
            value_transform=op, edit="boundary_value", constraints=constraints))

    # -- gradient-state mismatch at the boundary (5) ----------------------
    for kind, reference in _GRAD_ADAPTERS:
        ops.append(_make(
            f"interface.update.grad.{kind}", "Interface", "Update", ANY_EDGE,
            _insert_builder(_grad_specs(kind)),
            reference=reference,
            description=f"insert a {kind} adapter so the consumer observes a "
                        f"different gradient state",
            grad=kind, edit="boundary_grad"))

    # -- consumer-side interface expectations (8) -------------------------
    ops.append(_make(
        "interface.update.norm.num_features_off_by_one", "Interface", "Update",
        Pattern(op="batchnorm", description="normalisation boundary"),
        _consumer_update_builder(_batchnorm_off_by_one),
        reference="COMET",
        description="make the normalisation expect one channel more than the "
                    "producer really supplies",
        expectation="num_features", edit="boundary_attr"))
    ops.append(_make(
        "interface.update.conv.groups_mismatch", "Interface", "Update",
        Pattern(op="conv2d", description="convolution boundary"),
        _consumer_update_builder({"groups": 4}),
        reference="COMET",
        description="declare four convolution groups while the weight layout "
                    "only supports one",
        expectation="groups", edit="boundary_attr"))
    ops.append(_make(
        "interface.update.conv.dilation_mismatch", "Interface", "Update",
        Pattern(op="conv2d", description="convolution boundary"),
        _consumer_update_builder({"dilation": 16}),
        reference="DeepHunter",
        description="declare a dilation whose kernel no longer fits the operand",
        expectation="dilation", edit="boundary_attr"))
    ops.append(_make(
        "interface.update.linear.in_features_mismatch", "Interface", "Update",
        Pattern(op="linear", description="linear boundary"),
        _consumer_update_builder(_linear_in_features_mismatch),
        reference="LEMON",
        description="declare a different input feature count than the producer "
                    "supplies",
        expectation="in_features", edit="boundary_attr"))
    ops.append(_make(
        "interface.update.softmax.dim_out_of_range", "Interface", "Update",
        Pattern(op_in=SOFTMAX_OPS, description="softmax boundary"),
        _consumer_update_builder({"dim": 7}),
        reference="Audee",
        description="normalise over a dimension the operand does not have",
        expectation="dim", edit="boundary_attr"))
    ops.append(_make(
        "interface.update.pool.kernel_oversized", "Interface", "Update",
        Pattern(op_in=POOL_OPS, description="pooling boundary"),
        _consumer_update_builder({"kernel_size": 64}),
        reference="LEMON",
        description="declare a pooling kernel larger than the padded operand",
        expectation="kernel_size", edit="boundary_attr"))
    ops.append(_make(
        "interface.update.attr.expect_channels_last", "Interface", "Update",
        ANY_EDGE, _consumer_update_builder({"input_layout": "channels_last"}),
        reference="COMET",
        description="make the consumer expect a channels-last producer while "
                    "the producer emits contiguous storage",
        expectation="input_layout", edit="boundary_attr"))
    ops.append(_make(
        "interface.update.attr.expect_rank", "Interface", "Update", ANY_EDGE,
        _consumer_update_builder({"input_rank": 5}),
        reference="DevMuT",
        description="make the consumer expect a rank the producer never emits",
        expectation="input_rank", edit="boundary_attr"))

    return ops


def _layout_specs(kind: str):
    def specs(tfg: TFG, site: Site) -> Optional[List[Dict[str, Any]]]:
        shape = _shape_of(tfg, site.anchor_edge)
        if shape is None:
            return None
        rank = len(shape)
        if kind == "contiguous":
            return _insert_one("contiguous")
        if kind == "permute_reverse":
            if rank < 2:
                return None
            return _insert_one("permute", {"dims": tuple(reversed(range(rank)))})
        if kind == "transpose_trailing":
            if rank < 2:
                return None
            return _insert_one("transpose", {"dim0": -2, "dim1": -1})
        if kind == "channels_last":
            if rank != 4:
                return None
            return _insert_one("permute", {"dims": (0, 2, 3, 1)})
        if kind == "strided_view":
            return _insert_one("view", {"shape": shape})
        if kind == "reshape_flat":
            return _insert_one("reshape", {"shape": (_numel(shape),)})
        return None
    return specs


def _shape_specs(kind: str):
    def specs(tfg: TFG, site: Site) -> Optional[List[Dict[str, Any]]]:
        shape = _shape_of(tfg, site.anchor_edge)
        if shape is None:
            return None
        if kind == "unsqueeze_dim0":
            return _insert_one("unsqueeze", {"dim": 0})
        if kind == "unsqueeze_dim1":
            return _insert_one("unsqueeze", {"dim": 1})
        if kind == "squeeze_dim0":
            return _insert_one("squeeze", {"dim": 0})
        if kind == "squeeze_last":
            return _insert_one("squeeze", {"dim": -1})
        if kind == "flatten_all":
            return _insert_one("flatten", {"start_dim": 0, "end_dim": -1})
        if kind == "reshape_rank2":
            if len(shape) < 2 or shape[0] <= 0:
                return None
            total = _numel(shape)
            if total % shape[0]:
                return None
            return _insert_one("reshape", {"shape": (shape[0], total // shape[0])})
        if kind == "getitem_dim0":
            return _insert_one("getitem", {"index": 0})
        if kind == "expand_last":
            target = tuple(list(shape[:-1]) + [max(1, shape[-1] * 2)])
            return _insert_one("expand", {"shape": target})
        return None
    return specs


def _grad_specs(kind: str) -> List[Dict[str, Any]]:
    if kind == "detach":
        return _insert_one("detach")
    if kind == "stop_gradient":
        return _insert_one("stop_gradient")
    if kind == "clone":
        return _insert_one("clone")
    if kind == "detach_then_clone":
        return [{"key": "detach", "op": "detach", "inputs": ["{anchor}"]},
                {"key": "clone", "op": "clone", "inputs": ["@detach"]}]
    if kind == "clone_then_detach":
        return [{"key": "clone", "op": "clone", "inputs": ["{anchor}"]},
                {"key": "detach", "op": "detach", "inputs": ["@clone"]}]
    return []


def _log_domain_constraint(tfg: TFG, site: Site) -> List[ConstraintSpec]:
    return [ConstraintSpec(
        kind="input", predicate="value_range", args={"position": 0, "lo": 0.0},
        description="the logarithm is only defined on non-negative operands")]


def _linear_in_features_mismatch(tfg: TFG, site: Site) -> Optional[Dict[str, Any]]:
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    weight = _param_weight_of(tfg, node, 1)
    if weight is None or not weight.shape:
        return None
    return {"in_features": max(1, int(weight.shape[-1]) * 2)}


# ===========================================================================
# Interface / Deletion  (1)
# ===========================================================================

def interface_deletion_operators() -> List[MutationOperator]:
    """Delete an explicit adapter on the producer-consumer boundary."""
    return [_make(
        "interface.deletion.adapter", "Interface", "Deletion",
        Pattern(op_in=ADAPTER_OPS,
                description="explicit boundary adapter (cast/reshape/view/...), "
                            "which exposes the implicit conversion it performed"),
        _delete_consumer,
        reference="COMET",
        description="delete the explicit boundary adapter and reconnect the "
                    "consumer directly to the producer",
        edit="remove_adapter")]


# ===========================================================================
# Subgraph / Update  (15)
# ===========================================================================

_BLOCK_GAINS: Tuple[Tuple[float, str], ...] = (
    (0.5, "half"),
    (2.0, "double"),
    (0.0, "zero"),
    (-1.0, "negate"),
)


def subgraph_update_operators() -> List[MutationOperator]:
    ops: List[MutationOperator] = []

    # -- residual / block gain (4) ----------------------------------------
    for gain, label in _BLOCK_GAINS:
        ops.append(_make(
            f"subgraph.update.block.gain_{label}", "Subgraph", "Update", ANY_EDGE,
            _insert_builder(_scaled_gain_specs(gain)),
            reference="COMET",
            description=f"scale the bounded computation's input by {gain}",
            scale=gain, edit="block_gain"))

    # -- block width multiplier (3) ---------------------------------------
    ops.append(_make(
        "subgraph.update.block.width.conv_double", "Subgraph", "Update",
        Pattern(op="conv2d", description="convolution block"),
        _consumer_update_builder(_double_conv_channels),
        reference="LEMON",
        description="double the block width of the convolution stage",
        width_factor=2, edit="block_width"))
    ops.append(_make(
        "subgraph.update.block.width.conv_half", "Subgraph", "Update",
        Pattern(op="conv2d", description="convolution block"),
        _consumer_update_builder(_halve_conv_channels),
        reference="LEMON",
        description="halve the block width of the convolution stage",
        width_factor=0.5, edit="block_width"))
    ops.append(_make(
        "subgraph.update.block.width.linear_double", "Subgraph", "Update",
        Pattern(op="linear", description="fully connected block"),
        _consumer_update_builder(_double_linear_features),
        reference="COMET",
        description="double the block width of the fully connected stage",
        width_factor=2, edit="block_width"))

    # -- depth of a repeated pattern (3) ----------------------------------
    ops.append(_make(
        "subgraph.update.block.depth.relu_repeat", "Subgraph", "Update",
        Pattern(op="relu", description="repeated activation block"),
        _repeat_consumer,
        reference="DeepHunter",
        description="increase the depth of the repeated activation pattern by one",
        depth_delta=1, edit="block_depth"))
    ops.append(_make(
        "subgraph.update.block.depth.batchnorm_repeat", "Subgraph", "Update",
        Pattern(op="batchnorm", description="repeated normalisation block"),
        _repeat_consumer,
        reference="COMET",
        description="increase the depth of the repeated normalisation pattern by one",
        depth_delta=1, edit="block_depth"))
    ops.append(_make(
        "subgraph.update.block.depth.conv_repeat", "Subgraph", "Update",
        Pattern(op="conv2d", description="repeated convolution block"),
        _repeat_consumer,
        reference="LEMON",
        description="increase the depth of the repeated convolution pattern by one",
        depth_delta=1, edit="block_depth"))

    # -- normalisation placement (5) --------------------------------------
    activation_pattern = Pattern(op_in=ACTIVATION_OPS,
                                 description="activation block boundary")
    pool_output_pattern = Pattern(producer_op_in=POOL_OPS,
                                  description="pooling block output")
    linear_output_pattern = Pattern(producer_op_in=("linear",),
                                    description="fully connected block output")
    ops.append(_make(
        "subgraph.update.norm.pre_activation.batchnorm", "Subgraph", "Update",
        activation_pattern, _insert_builder(_norm_specs("batchnorm")),
        reference="COMET",
        description="place a batch-normalisation block before the activation",
        placement="pre_activation", norm="batchnorm", edit="norm_placement"))
    ops.append(_make(
        "subgraph.update.norm.pre_activation.layernorm", "Subgraph", "Update",
        activation_pattern, _insert_builder(_norm_specs("layernorm")),
        reference="COMET",
        description="place a layer-normalisation block before the activation",
        placement="pre_activation", norm="layernorm", edit="norm_placement"))
    ops.append(_make(
        "subgraph.update.norm.post_pool.batchnorm", "Subgraph", "Update",
        pool_output_pattern, _insert_builder(_norm_specs("batchnorm")),
        reference="LEMON",
        description="place a batch-normalisation block after the pooling stage",
        placement="post_pool", norm="batchnorm", edit="norm_placement"))
    ops.append(_make(
        "subgraph.update.norm.post_pool.groupnorm", "Subgraph", "Update",
        pool_output_pattern, _insert_builder(_norm_specs("groupnorm")),
        reference="LEMON",
        description="place a group-normalisation block after the pooling stage",
        placement="post_pool", norm="groupnorm", edit="norm_placement"))
    ops.append(_make(
        "subgraph.update.norm.post_linear.layernorm", "Subgraph", "Update",
        linear_output_pattern, _insert_builder(_norm_specs("layernorm")),
        reference="DeepHunter",
        description="place a layer-normalisation block after the linear stage",
        placement="post_linear", norm="layernorm", edit="norm_placement"))

    return ops


def _scaled_gain_specs(gain: float):
    def specs(tfg: TFG, site: Site) -> Optional[List[Dict[str, Any]]]:
        return [{"key": "gain", "op": "param", "inputs": [],
                 "attrs": {"shape": (1,), "init": "ones", "value": gain}},
                {"key": "scale", "op": "mul",
                 "inputs": ["{anchor}", "@gain"], "attrs": {}}]
    return specs


def _halve_conv_channels(tfg: TFG, site: Site) -> Optional[Dict[str, Any]]:
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    weight = _param_weight_of(tfg, node, 1)
    if weight is None or not weight.shape:
        return None
    return {"out_channels": max(1, int(weight.shape[0]) // 2)}


def _norm_specs(kind: str):
    def specs(tfg: TFG, site: Site) -> Optional[List[Dict[str, Any]]]:
        shape = _shape_of(tfg, site.anchor_edge)
        if shape is None:
            return None
        if kind == "batchnorm":
            if len(shape) < 2 or shape[1] <= 0:
                return None
            return _insert_one("batchnorm", {"num_features": int(shape[1])})
        if kind == "layernorm":
            return _insert_one("layernorm",
                               {"normalized_shape": (int(shape[-1]),)})
        if kind == "groupnorm":
            if len(shape) < 2 or shape[1] <= 0:
                return None
            groups = 2 if int(shape[1]) % 2 == 0 else 1
            return _insert_one("groupnorm", {"num_groups": groups})
        if kind == "instancenorm":
            return _insert_one("instancenorm")
        return None
    return specs


# ===========================================================================
# Subgraph / Insertion  (9)
# ===========================================================================

def subgraph_insertion_operators() -> List[MutationOperator]:
    ops: List[MutationOperator] = []

    # -- residual / skip connections (3) ----------------------------------
    ops.append(_make(
        "subgraph.insertion.residual.plain", "Subgraph", "Insertion", ANY_EDGE,
        _insert_builder(_residual_specs("plain")),
        reference="LEMON",
        description="wrap the bounded region in a residual connection with an "
                    "activation branch",
        residual="plain", edit="residual_insertion"))
    ops.append(_make(
        "subgraph.insertion.residual.scaled", "Subgraph", "Insertion", ANY_EDGE,
        _insert_builder(_residual_specs("scaled")),
        reference="COMET",
        description="insert a scaled residual connection around the bounded region",
        residual="scaled", edit="residual_insertion"))
    ops.append(_make(
        "subgraph.insertion.residual.identity", "Subgraph", "Insertion", ANY_EDGE,
        _insert_builder(_residual_specs("clone")),
        reference="DevMuT",
        description="insert an identity (clone) residual branch on the bounded region",
        residual="identity", edit="residual_insertion"))

    # -- normalisation blocks (3) -----------------------------------------
    for kind, reference in (("batchnorm", "COMET"), ("layernorm", "DeepHunter"),
                            ("groupnorm", "LEMON")):
        ops.append(_make(
            f"subgraph.insertion.norm.{kind}", "Subgraph", "Insertion", ANY_EDGE,
            _insert_builder(_norm_specs(kind)),
            reference=reference,
            description=f"insert a whole {kind} block on the anchor edge",
            norm=kind, edit="norm_insertion"))

    # -- activation blocks (2) ---------------------------------------------
    ops.append(_make(
        "subgraph.insertion.activation.relu", "Subgraph", "Insertion", ANY_EDGE,
        _insert_builder(_insert_one("relu")),
        reference="DeepHunter",
        description="insert a ReLU activation block on the anchor edge",
        activation="relu", edit="activation_insertion"))
    ops.append(_make(
        "subgraph.insertion.activation.gelu", "Subgraph", "Insertion", ANY_EDGE,
        _insert_builder(_insert_one("gelu")),
        reference="LEMON",
        description="insert a GELU activation block on the anchor edge",
        activation="gelu", edit="activation_insertion"))

    # -- regularisation block (1) ------------------------------------------
    ops.append(_make(
        "subgraph.insertion.regularisation.dropout", "Subgraph", "Insertion",
        ANY_EDGE, _insert_builder(_insert_one("dropout", {"p": 0.5})),
        reference="DeepHunter",
        description="insert a dropout regularisation block on the anchor edge",
        regularisation="dropout", edit="regularisation_insertion"))

    return ops


def _residual_specs(kind: str) -> List[Dict[str, Any]]:
    if kind == "plain":
        return [{"key": "branch", "op": "relu", "inputs": ["{anchor}"], "attrs": {}},
                {"key": "residual", "op": "add",
                 "inputs": ["{anchor}", "@branch"], "attrs": {}}]
    if kind == "scaled":
        return [{"key": "gain", "op": "param", "inputs": [],
                 "attrs": {"shape": (1,), "init": "ones", "value": 0.5}},
                {"key": "branch", "op": "mul",
                 "inputs": ["{anchor}", "@gain"], "attrs": {}},
                {"key": "residual", "op": "add",
                 "inputs": ["{anchor}", "@branch"], "attrs": {}}]
    return [{"key": "branch", "op": "clone", "inputs": ["{anchor}"], "attrs": {}},
            {"key": "residual", "op": "add",
             "inputs": ["{anchor}", "@branch"], "attrs": {}}]


# ===========================================================================
# Subgraph / Deletion  (1)
# ===========================================================================

def subgraph_deletion_operators() -> List[MutationOperator]:
    """Delete a normalisation block (or a residual merge) and reconnect."""
    return [_make(
        "subgraph.deletion.norm_block", "Subgraph", "Deletion",
        Pattern(op_in=("batchnorm", "layernorm", "groupnorm", "instancenorm",
                       "rmsnorm", "add"),
                description="bounded normalisation block or residual merge"),
        _delete_consumer,
        reference="COMET",
        description="delete the bounded normalisation block and reconnect its "
                    "consumers to its input boundary",
        edit="remove_block")]


# ===========================================================================
# Subgraph / Rewiring  (4)
# ===========================================================================

def subgraph_rewiring_operators() -> List[MutationOperator]:
    """Change producer-consumer connections inside a bounded region."""
    binary_pattern = Pattern(op_in=BINARY_OPS,
                             description="bounded binary merge of two branches")
    return [
        _make("subgraph.rewiring.residual_target", "Subgraph", "Rewiring",
              binary_pattern, _rewire_residual_target,
              reference="LEMON",
              description="rewire the residual branch so it merges with itself "
                          "instead of the other branch",
              edit="residual_rewire"),
        _make("subgraph.rewiring.swap_producers", "Subgraph", "Rewiring",
              Pattern(description="bounded region with two data producers"),
              _rewire_swap_producers,
              reference="COMET",
              description="swap the two producers feeding the bounded region",
              edit="producer_swap"),
        _make("subgraph.rewiring.share_producer", "Subgraph", "Rewiring",
              Pattern(min_consumers=2,
                      description="edge shared by several consumers"),
              _rewire_share_producer,
              reference="DevMuT",
              description="rewire one consumer path onto the other consumer's "
                          "output so a single producer feeds the whole region",
              edit="branch_local_rewire"),
        _make("subgraph.rewiring.skip_forward", "Subgraph", "Rewiring",
              Pattern(description="bounded region that may be bypassed"),
              _rewire_skip_forward,
              reference="LEMON",
              description="rewire a consumer onto an earlier same-rank tensor, "
                          "introducing a skip across the bounded region",
              edit="skip_rewire"),
    ]


def _rewire_residual_target(tfg: TFG, site: Site) -> Optional[Replacement]:
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    data_inputs = _data_inputs(tfg, node)
    if len(data_inputs) < 2 or data_inputs[0] == data_inputs[1]:
        return None
    position = None
    for index, edge_id in enumerate(node.inputs):
        if edge_id == data_inputs[1]:
            position = index
            break
    if position is None:
        return None
    return Replacement(
        remove=[], keep=[site.consumer],
        rewires=[(site.consumer, position, data_inputs[0])],
        attr_updates={site.consumer: {"residual_target": data_inputs[0]}})


def _rewire_swap_producers(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Swap the two *data* producers of the matched node."""
    node = tfg.graph.nodes.get(site.consumer)
    if node is None:
        return None
    data_inputs = _data_inputs(tfg, node)
    if len(data_inputs) < 2 or data_inputs[0] == data_inputs[1]:
        return None
    positions: List[int] = []
    for wanted in data_inputs[:2]:
        positions.append(next(index for index, edge_id in enumerate(node.inputs)
                              if edge_id == wanted))
    first, second = positions
    return Replacement(
        remove=[], keep=[site.consumer],
        rewires=[(site.consumer, first, data_inputs[1]),
                 (site.consumer, second, data_inputs[0])],
        attr_updates={site.consumer: {"swapped_inputs": (first, second)}})


def _rewire_share_producer(tfg: TFG, site: Site) -> Optional[Replacement]:
    consumers = _anchor_consumers(tfg, site)
    if len(consumers) < 2:
        return None
    (first, _), (second, second_position) = consumers[0], consumers[1]
    outputs = tfg.graph.out_edges(first)
    if not outputs:
        return None
    target = outputs[0]
    if target == site.anchor_edge:
        return None
    return Replacement(
        remove=[], keep=[site.consumer],
        rewires=[(second, second_position, target)],
        attr_updates={second: {"rewired_from": site.anchor_edge,
                               "rewired_to": target}})


def _rewire_skip_forward(tfg: TFG, site: Site) -> Optional[Replacement]:
    """Bypass the anchor's producer so the consumer reads an earlier tensor.

    The new producer is an input of the current producer, i.e. an ancestor of
    the anchor edge, so the rewiring can never introduce a cycle.
    """
    spec = _spec_of(tfg, site.anchor_edge)
    edge = tfg.graph.edges.get(site.anchor_edge)
    if spec is None or edge is None:
        return None
    producer = tfg.graph.nodes.get(edge.producer)
    if producer is None:
        return None
    position = _anchor_position(tfg, site)
    target = None
    for candidate in _data_inputs(tfg, producer):
        other = _spec_of(tfg, candidate)
        if other is not None and other.rank == spec.rank:
            target = candidate
            break
    if target is None:
        return None
    return Replacement(
        remove=[], keep=[site.consumer],
        rewires=[(site.consumer, position, target)],
        attr_updates={site.consumer: {"rewired_from": site.anchor_edge,
                                      "rewired_to": target}})


# ===========================================================================
# The 120-operator list
# ===========================================================================

CG_OPERATORS: List[MutationOperator] = [
    *tensor_update_operators(),
    *tensor_insertion_operators(),
    *operator_update_operators(),
    *operator_insertion_operators(),
    *operator_deletion_operators(),
    *interface_update_operators(),
    *interface_deletion_operators(),
    *subgraph_update_operators(),
    *subgraph_insertion_operators(),
    *subgraph_deletion_operators(),
    *subgraph_rewiring_operators(),
]

#: Expected size of each ``(target object, primitive)`` cell of the paper's table.
EXPECTED_COMPOSITION: Dict[Tuple[str, str], int] = {
    ("Tensor", "Update"): 5,
    ("Tensor", "Insertion"): 1,
    ("Operator", "Update"): 37,
    ("Operator", "Insertion"): 3,
    ("Operator", "Deletion"): 1,
    ("Interface", "Update"): 43,
    ("Interface", "Insertion"): 0,
    ("Interface", "Deletion"): 1,
    ("Interface", "Rewiring"): 0,
    ("Subgraph", "Update"): 15,
    ("Subgraph", "Insertion"): 9,
    ("Subgraph", "Deletion"): 1,
    ("Subgraph", "Rewiring"): 4,
}


def composition(operators: Optional[Sequence[MutationOperator]] = None
                ) -> Dict[Tuple[str, str], int]:
    """Count the pool by ``(target object, primitive)`` cell."""
    counts: Dict[Tuple[str, str], int] = {}
    for operator in (operators if operators is not None else CG_OPERATORS):
        cell = (operator.target_object, operator.primitive)
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def check_composition(operators: Optional[Sequence[MutationOperator]] = None
                      ) -> None:
    """Raise when the pool deviates from the paper's construction table."""
    pool = list(operators if operators is not None else CG_OPERATORS)
    counts = composition(pool)
    problems: List[str] = []
    for cell, expected in EXPECTED_COMPOSITION.items():
        found = counts.get(cell, 0)
        if found != expected:
            problems.append(f"{cell[0]}/{cell[1]}: expected {expected}, found {found}")
    for cell, found in counts.items():
        if cell not in EXPECTED_COMPOSITION:
            problems.append(f"{cell[0]}/{cell[1]}: unexpected cell with {found}")
    if len(pool) != 120:
        problems.append(f"total: expected 120, found {len(pool)}")
    names = [operator.name for operator in pool]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(f"duplicate operator names: {duplicates}")
    if problems:
        raise AssertionError("CG operator pool is inconsistent: " + "; ".join(problems))


def cg_source_count() -> int:
    """Number of CG-derived operators exported by this module."""
    return len(CG_OPERATORS)


check_composition()
