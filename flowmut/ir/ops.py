"""The IR operator registry.

Every op that can appear in a :class:`~flowmut.ir.graph.Node` is declared here
together with

* its arity and which inputs are *parameters*,
* the operator input requirements ``R(v)`` that FlowMuT attaches to the node,
* a static shape-inference function used when building replacement subgraphs.

Requirements are emitted as :class:`~flowmut.ir.specs.OpRequirement` records
whose ``predicate`` names are resolved by :mod:`flowmut.tfg.predicates`.  This
keeps the declaration close to the operator while the checking logic stays in
one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from flowmut.ir.specs import (
    FLOAT_DTYPES,
    INT_DTYPES,
    OpRequirement,
    TensorSpec,
    broadcast_shape,
    shape_numel,
)

# ---------------------------------------------------------------------------
# Op definition
# ---------------------------------------------------------------------------

ShapeFn = Callable[[Sequence[TensorSpec], Dict[str, Any]], Optional[Tuple[int, ...]]]


@dataclass
class OpDef:
    name: str
    min_inputs: int = 1
    max_inputs: int = 1                 # -1 == variadic
    #: Input positions that must be parameter (weight) edges.
    param_positions: Tuple[int, ...] = ()
    #: Requirement templates: ``(kind, predicate, args, description)``.  Args may
    #: use ``"{n}"``/``"{i}"`` placeholders resolved against the node's arity.
    requirements: Tuple[Tuple[str, str, Dict[str, Any], str], ...] = ()
    infer: Optional[ShapeFn] = None
    group: str = "misc"
    doc: str = ""

    def arity_ok(self, n: int) -> bool:
        if n < self.min_inputs:
            return False
        if self.max_inputs >= 0 and n > self.max_inputs:
            return False
        return True


# ---------------------------------------------------------------------------
# Shape inference helpers
# ---------------------------------------------------------------------------

def _unary(specs: Sequence[TensorSpec], attrs: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    return tuple(specs[0].shape) if specs else None


def _broadcast(specs: Sequence[TensorSpec], attrs: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    if len(specs) < 2:
        return tuple(specs[0].shape) if specs else None
    return broadcast_shape(specs[0].shape, specs[1].shape)


def _same_as_first(specs: Sequence[TensorSpec], attrs: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    return tuple(specs[0].shape) if specs else None


def _infer_matmul(specs, attrs):
    if len(specs) < 2:
        return None
    a, b = list(specs[0].shape), list(specs[1].shape)
    if len(a) < 2 or len(b) < 2:
        return None
    if a[-1] != b[-2]:
        return None
    out = a[:-1] + [b[-1]]
    if len(a) > 2 and len(b) > 2:
        batch = broadcast_shape(a[:-2], b[:-2])
        if batch is None:
            return None
        out = list(batch) + [a[-2], b[-1]]
    return tuple(out)


def _infer_linear(specs, attrs):
    if len(specs) < 2:
        return None
    x, w = list(specs[0].shape), list(specs[1].shape)
    if not x or len(w) != 2 or x[-1] != w[-1]:
        return None
    return tuple(x[:-1] + [w[0]])


def _infer_conv2d(specs, attrs):
    if len(specs) < 2:
        return None
    x, w = specs[0].shape, specs[1].shape
    if len(x) != 4 or len(w) != 4:
        return None
    n, _, h, wd = x
    out_c, _, kh, kw = w
    stride = attrs.get("stride", 1)
    pad = attrs.get("padding", 0)
    dil = attrs.get("dilation", 1)
    stride = stride[0] if isinstance(stride, (tuple, list)) else stride
    pad = pad[0] if isinstance(pad, (tuple, list)) else pad
    dil = dil[0] if isinstance(dil, (tuple, list)) else dil
    oh = (h + 2 * pad - dil * (kh - 1) - 1) // stride + 1
    ow = (wd + 2 * pad - dil * (kw - 1) - 1) // stride + 1
    if oh <= 0 or ow <= 0:
        return None
    return (n, out_c, oh, ow)


def _infer_pool(specs, attrs):
    if not specs:
        return None
    x = specs[0].shape
    if len(x) != 4:
        return tuple(x)
    n, c, h, w = x
    k = attrs.get("kernel_size", 2)
    s = attrs.get("stride", k)
    p = attrs.get("padding", 0)
    k = k[0] if isinstance(k, (tuple, list)) else k
    s = s[0] if isinstance(s, (tuple, list)) else s
    p = p[0] if isinstance(p, (tuple, list)) else p
    if attrs.get("ceil_mode"):
        oh = -(-(h + 2 * p - k) // s) + 1
        ow = -(-(w + 2 * p - k) // s) + 1
    else:
        oh = (h + 2 * p - k) // s + 1
        ow = (w + 2 * p - k) // s + 1
    if oh <= 0 or ow <= 0:
        return None
    return (n, c, oh, ow)


def _infer_adaptive_pool(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    out = attrs.get("output_size", 1)
    if isinstance(out, (tuple, list)):
        dims = [int(o) for o in out]
    else:
        dims = [int(out)] * max(0, len(x) - 2)
    if len(x) < 2:
        return tuple(x)
    return tuple(x[:2] + dims)


def _infer_global_pool(specs, attrs):
    if not specs:
        return None
    x = specs[0].shape
    keepdim = attrs.get("keepdim", False)
    if attrs.get("spatial", True) and len(x) >= 3:
        return tuple(x[:2] + ((1,) * (len(x) - 2) if keepdim else ()))
    return tuple(x)


def _infer_reshape(specs, attrs):
    if not specs:
        return None
    target = attrs.get("shape")
    if target is None:
        return None
    return tuple(int(d) for d in target)


def _infer_flatten(specs, attrs):
    if not specs:
        return None
    x = specs[0].shape
    start = attrs.get("start_dim", 1)
    end = attrs.get("end_dim", -1)
    start = start if start >= 0 else len(x) + start
    end = end if end >= 0 else len(x) + end
    if start > end or end >= len(x):
        return None
    flat = shape_numel(x[start:end + 1])
    return tuple(list(x[:start]) + [flat] + list(x[end + 1:]))


def _infer_permute(specs, attrs):
    if not specs:
        return None
    x = specs[0].shape
    dims = attrs.get("dims")
    if dims is None or len(dims) != len(x):
        return None
    try:
        return tuple(int(x[d]) for d in dims)
    except IndexError:
        return None


def _infer_squeeze(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    dim = attrs.get("dim")
    if dim is None:
        return tuple(d for d in x if d != 1)
    dim = dim if dim >= 0 else len(x) + dim
    if 0 <= dim < len(x) and x[dim] == 1:
        x.pop(dim)
    return tuple(x)


def _infer_unsqueeze(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    dim = attrs.get("dim", 0)
    dim = dim if dim >= 0 else len(x) + dim + 1
    x.insert(max(0, min(dim, len(x))), 1)
    return tuple(x)


def _infer_concat(specs, attrs):
    if not specs:
        return None
    dim = attrs.get("dim", 0)
    shapes = [list(s.shape) for s in specs]
    rank = len(shapes[0])
    dim = dim if dim >= 0 else rank + dim
    if any(len(s) != rank for s in shapes) or not (0 <= dim < rank):
        return None
    out = list(shapes[0])
    total = 0
    for s in shapes:
        for i in range(rank):
            if i != dim and s[i] != shapes[0][i]:
                return None
        total += s[dim]
    out[dim] = total
    return tuple(out)


def _infer_split(specs, attrs):
    if not specs:
        return None
    x = specs[0].shape
    dim = attrs.get("dim", 0)
    dim = dim if dim >= 0 else len(x) + dim
    if not (0 <= dim < len(x)):
        return None
    n = attrs.get("num_outputs", 2) or 2
    if x[dim] % n:
        return None
    out = list(x)
    out[dim] = x[dim] // n
    return tuple(out)


def _infer_stack(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    dim = attrs.get("dim", 0)
    dim = dim if dim >= 0 else len(x) + dim + 1
    x.insert(max(0, min(dim, len(x))), len(specs))
    return tuple(x)


def _infer_getitem(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    if not x:
        return None
    dim = attrs.get("dim", 0)
    dim = dim if dim >= 0 else len(x) + dim
    if not (0 <= dim < len(x)):
        return None
    index = attrs.get("index", 0)
    if not isinstance(index, int):
        return None
    if not (-x[dim] <= index < x[dim]):
        return None
    return tuple(x[:dim] + x[dim + 1:])


def _infer_interpolate(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    size = attrs.get("size")
    if size is None:
        return None
    if len(x) != 4:
        return None
    return (x[0], x[1], int(size[0]), int(size[1]))


def _infer_pixel_shuffle(specs, attrs):
    if not specs:
        return None
    x = specs[0].shape
    if len(x) != 4:
        return None
    r = attrs.get("upscale_factor", 2)
    n, c, h, w = x
    if r <= 0 or c % (r * r):
        return None
    return (n, c // (r * r), h * r, w * r)


def _infer_softmax(specs, attrs):
    return _unary(specs, attrs)


def _infer_sdpa(specs, attrs):
    if len(specs) < 3:
        return None
    q, k, v = specs[0].shape, specs[1].shape, specs[2].shape
    if len(q) < 2 or len(k) < 2 or len(v) < 2:
        return None
    if q[-1] != k[-1] or k[-2] != v[-2]:
        return None
    return tuple(list(q[:-1]) + [v[-1]])


def _infer_embedding(specs, attrs):
    if len(specs) < 2:
        return None
    idx, w = specs[0].shape, specs[1].shape
    if len(w) != 2:
        return None
    return tuple(list(idx) + [w[1]])


def _infer_reduce(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    dim = attrs.get("dim")
    keepdim = attrs.get("keepdim", False)
    if dim is None:
        return tuple(1 for _ in x) if keepdim else ()
    dims = [dim] if isinstance(dim, int) else list(dim)
    dims = [d if d >= 0 else len(x) + d for d in dims]
    if any(not (0 <= d < len(x)) for d in dims):
        return None
    if keepdim:
        for d in dims:
            x[d] = 1
        return tuple(x)
    return tuple(v for i, v in enumerate(x) if i not in set(dims))


def _infer_argmax(specs, attrs):
    return _infer_reduce(specs, {**attrs, "keepdim": attrs.get("keepdim", False)})


def _infer_cast(specs, attrs):
    return _unary(specs, attrs)


def _infer_norm(specs, attrs):
    return _unary(specs, attrs)


def _infer_conv_transpose(specs, attrs):
    if len(specs) < 2:
        return None
    x, w = specs[0].shape, specs[1].shape
    if len(x) != 4 or len(w) != 4:
        return None
    n, _, h, wd = x
    _, out_c, kh, kw = w
    stride = attrs.get("stride", 1)
    pad = attrs.get("padding", 0)
    op = attrs.get("output_padding", 0)
    stride = stride[0] if isinstance(stride, (tuple, list)) else stride
    pad = pad[0] if isinstance(pad, (tuple, list)) else pad
    op = op[0] if isinstance(op, (tuple, list)) else op
    oh = (h - 1) * stride - 2 * pad + kh + op
    ow = (wd - 1) * stride - 2 * pad + kw + op
    return (n, out_c, oh, ow)


def _infer_conv1d(specs, attrs):
    if len(specs) < 2:
        return None
    x, w = specs[0].shape, specs[1].shape
    if len(x) != 3 or len(w) != 3:
        return None
    n, _, l = x
    out_c, _, kw = w
    stride = attrs.get("stride", 1)
    pad = attrs.get("padding", 0)
    dil = attrs.get("dilation", 1)
    stride = stride[0] if isinstance(stride, (tuple, list)) else stride
    pad = pad[0] if isinstance(pad, (tuple, list)) else pad
    dil = dil[0] if isinstance(dil, (tuple, list)) else dil
    ol = (l + 2 * pad - dil * (kw - 1) - 1) // stride + 1
    if ol <= 0:
        return None
    return (n, out_c, ol)


def _infer_pad(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    pad = attrs.get("pad")
    if pad is None or len(pad) % 2:
        return None
    out = list(x)
    pairs = len(pad) // 2
    if pairs > len(x):
        return None
    for i in range(pairs):
        dim = len(x) - 1 - i
        out[dim] = x[dim] + int(pad[2 * i]) + int(pad[2 * i + 1])
    return tuple(out)


def _infer_topk(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    k = attrs.get("k", 1)
    dim = attrs.get("dim", -1)
    dim = dim if dim >= 0 else len(x) + dim
    if not (0 <= dim < len(x)):
        return None
    x[dim] = min(int(k), x[dim])
    return tuple(x)


def _infer_tile(specs, attrs):
    if not specs:
        return None
    x = list(specs[0].shape)
    reps = attrs.get("reps")
    if reps is None or len(reps) != len(x):
        return None
    return tuple(int(a) * int(b) for a, b in zip(x, reps))


def _infer_expand(specs, attrs):
    if not specs:
        return None
    shape = attrs.get("shape")
    if shape is None:
        return None
    return broadcast_shape(specs[0].shape, tuple(int(d) for d in shape)) or \
        tuple(int(d) for d in shape)


def _infer_masked(specs, attrs):
    return _broadcast(specs, attrs)


def _infer_identity(specs, attrs):
    return _unary(specs, attrs)


# ---------------------------------------------------------------------------
# Requirement templates
# ---------------------------------------------------------------------------

FLOAT_IN = ("input", "dtype_family", {"position": 0, "family": "float"},
            "operand must be floating point")
RANK1 = ("input", "rank_at_least", {"position": 0, "rank": 1}, "operand needs rank >= 1")
RANK2 = ("input", "rank_at_least", {"position": 0, "rank": 2}, "operand needs rank >= 2")
RANK4 = ("input", "rank_eq", {"position": 0, "rank": 4}, "operand must be 4-D (NCHW)")
FINITE = ("input", "value_finite", {"position": 0}, "operand values must be finite")
NONEMPTY = ("input", "numel_positive", {"position": 0}, "operand must not be empty")
SAME_DEVICE = ("internal", "device_agreement", {"positions": "all"},
               "all operands must live on the same device")
SAME_DTYPE_FLOAT = ("internal", "dtype_agreement", {"positions": "all"},
                    "operands must share a dtype")

#: Ops whose semantics are only defined for floating point tensors.
_FLOAT_UNARY = ("relu", "gelu", "silu", "sigmoid", "tanh", "leaky_relu", "elu", "mish",
                "hardswish", "hardsigmoid", "quick_gelu", "softplus", "exp", "log", "sqrt",
                "rsqrt", "erf", "sin", "cos", "reciprocal", "softmax", "log_softmax")

_ELEMENTWISE_UNARY = _FLOAT_UNARY + ("neg", "abs", "square", "floor", "round", "sign",
                                     "dropout", "cast", "clone", "detach", "contiguous",
                                     "stop_gradient", "identity", "flip", "roll")

_BINARY = ("add", "sub", "mul", "div", "pow", "maximum", "minimum")


def _reg(**kwargs) -> OpDef:
    return OpDef(**kwargs)


def _build_registry() -> Dict[str, OpDef]:
    R: Dict[str, OpDef] = {}

    def add(defn: OpDef) -> None:
        R[defn.name] = defn

    # -- graph structure ---------------------------------------------------
    add(_reg(name="input", min_inputs=0, max_inputs=0, group="io",
             requirements=(), infer=lambda s, a: tuple(a.get("shape", ())) or None))
    add(_reg(name="param", min_inputs=0, max_inputs=0, group="io",
             requirements=(), infer=lambda s, a: tuple(a.get("shape", ())) or None))
    add(_reg(name="output", min_inputs=1, max_inputs=-1, group="io",
             requirements=(), infer=_infer_identity))

    # -- elementwise unary -------------------------------------------------
    for op in _ELEMENTWISE_UNARY:
        reqs: Tuple = (NONEMPTY,)
        if op in _FLOAT_UNARY:
            reqs = (FLOAT_IN, NONEMPTY)
        if op in ("cast", "clone", "detach", "contiguous", "stop_gradient", "identity"):
            reqs = (NONEMPTY,)
        add(_reg(name=op, min_inputs=1, max_inputs=1, group="elementwise",
                 requirements=reqs, infer=_unary))
    # dropout carries an extra parameter edge only when training-mode masks are
    # materialised; the IR keeps it as a pure attribute.
    R["softmax"].requirements = (FLOAT_IN, ("input", "rank_at_least", {"position": 0, "rank": 1},
                                            "logits need rank >= 1"))
    R["log_softmax"].requirements = R["softmax"].requirements

    # -- elementwise binary / ternary --------------------------------------
    for op in _BINARY:
        add(_reg(name=op, min_inputs=2, max_inputs=2, group="elementwise",
                 requirements=(NONEMPTY, ("internal", "shape_broadcastable",
                                          {"positions": [0, 1]},
                                          "operands must broadcast"), SAME_DEVICE),
                 infer=_broadcast))
    add(_reg(name="where", min_inputs=3, max_inputs=3, group="elementwise",
             requirements=((NONEMPTY,), ("internal", "shape_broadcastable",
                                         {"positions": [1, 2]}, "branches must broadcast")),
             infer=_broadcast))
    add(_reg(name="clamp", min_inputs=1, max_inputs=1, group="elementwise",
             requirements=(NONEMPTY,), infer=_unary))
    add(_reg(name="masked_fill", min_inputs=2, max_inputs=2, group="elementwise",
             requirements=(NONEMPTY, ("internal", "shape_broadcastable",
                                      {"positions": [0, 1]}, "mask must broadcast")),
             infer=_unary))

    # -- reductions --------------------------------------------------------
    for op in ("mean", "sum", "amax", "amin", "var", "std", "prod"):
        add(_reg(name=op, min_inputs=1, max_inputs=1, group="reduce",
                 requirements=(NONEMPTY,), infer=_infer_reduce))
    for op in ("argmax", "argmin"):
        add(_reg(name=op, min_inputs=1, max_inputs=1, group="reduce",
                 requirements=(NONEMPTY,), infer=_infer_argmax))

    # -- linear algebra ----------------------------------------------------
    add(_reg(name="matmul", min_inputs=2, max_inputs=2, group="linalg",
             requirements=(RANK2, ("input", "rank_at_least", {"position": 1, "rank": 2},
                                   "second operand needs rank >= 2"),
                           ("internal", "matmul_compatible", {"positions": [0, 1]},
                            "inner dimensions must agree"), SAME_DTYPE_FLOAT, SAME_DEVICE),
             infer=_infer_matmul))
    add(_reg(name="linear", min_inputs=2, max_inputs=3, group="linalg",
             param_positions=(1, 2),
             requirements=(RANK1, ("input", "rank_eq", {"position": 1, "rank": 2},
                                   "weight must be 2-D"),
                           ("internal", "last_dim_match", {"positions": [0, 1]},
                            "input feature dim must match the weight"),
                           ("internal", "dtype_agreement", {"positions": [0, 1]},
                            "input and weight must share a dtype"),
                           FLOAT_IN, FINITE),
             infer=_infer_linear))
    add(_reg(name="bmm", min_inputs=2, max_inputs=2, group="linalg",
             requirements=(("input", "rank_eq", {"position": 0, "rank": 3}, "operand must be 3-D"),
                           ("internal", "matmul_compatible", {"positions": [0, 1]},
                            "batch matrix dimensions must agree"),
                           ("internal", "dtype_agreement", {"positions": [0, 1]},
                            "operands must share a dtype")),
             infer=_infer_matmul))
    add(_reg(name="einsum", min_inputs=2, max_inputs=-1, group="linalg",
             requirements=(NONEMPTY,), infer=None))

    # -- convolution -------------------------------------------------------
    add(_reg(name="conv2d", min_inputs=2, max_inputs=3, group="conv",
             param_positions=(1, 2),
             requirements=(RANK4, ("input", "rank_eq", {"position": 1, "rank": 4},
                                   "weight must be 4-D (OIHW)"),
                           ("internal", "channel_groups_match", {"positions": [0, 1],
                                                                 "groups": "groups"},
                            "in_channels must be divisible by groups"),
                           ("internal", "spatial_kernel_fits", {"position": 0},
                            "kernel must fit the padded input"),
                           ("internal", "dtype_agreement", {"positions": [0, 1]},
                            "input and weight must share a dtype"),
                           FLOAT_IN, FINITE),
             infer=_infer_conv2d))
    add(_reg(name="conv1d", min_inputs=2, max_inputs=3, group="conv",
             param_positions=(1, 2),
             requirements=(("input", "rank_eq", {"position": 0, "rank": 3}, "operand must be 3-D"),
                           ("internal", "channel_groups_match", {"positions": [0, 1],
                                                                 "groups": "groups"},
                            "in_channels must be divisible by groups"),
                           ("internal", "dtype_agreement", {"positions": [0, 1]},
                            "input and weight must share a dtype"),
                           FLOAT_IN),
             infer=_infer_conv1d))
    add(_reg(name="conv_transpose2d", min_inputs=2, max_inputs=3, group="conv",
             param_positions=(1, 2),
             requirements=(RANK4, ("input", "rank_eq", {"position": 1, "rank": 4},
                                   "weight must be 4-D (IOHW)"),
                           ("internal", "channel_groups_match", {"positions": [0, 1],
                                                                 "groups": "groups",
                                                                 "transposed": True},
                            "in_channels must be divisible by groups"),
                           ("internal", "dtype_agreement", {"positions": [0, 1]},
                            "input and weight must share a dtype"), FLOAT_IN),
             infer=_infer_conv_transpose))

    # -- pooling -----------------------------------------------------------
    for op in ("maxpool2d", "avgpool2d"):
        add(_reg(name=op, min_inputs=1, max_inputs=1, group="pool",
                 requirements=(RANK4, FLOAT_IN,
                               ("internal", "kernel_fits_input", {"position": 0},
                                "pooling kernel must fit the padded input")),
                 infer=_infer_pool))
    for op in ("adaptive_avgpool2d", "adaptive_maxpool2d"):
        add(_reg(name=op, min_inputs=1, max_inputs=1, group="pool",
                 requirements=(RANK4, FLOAT_IN), infer=_infer_adaptive_pool))
    for op in ("global_avgpool", "global_maxpool"):
        add(_reg(name=op, min_inputs=1, max_inputs=1, group="pool",
                 requirements=(RANK2, FLOAT_IN), infer=_infer_global_pool))

    # -- normalisation -----------------------------------------------------
    add(_reg(name="batchnorm", min_inputs=1, max_inputs=1, group="norm",
             requirements=(RANK2, FLOAT_IN,
                           ("internal", "channel_dim_known", {"position": 0},
                            "channel dimension must be statically known"),
                           ("internal", "dtype_agreement", {"positions": "all"},
                            "input and normalisation parameters must share a dtype")),
             infer=_infer_norm))
    add(_reg(name="layernorm", min_inputs=1, max_inputs=1, group="norm",
             requirements=(RANK1, FLOAT_IN,
                           ("internal", "normalized_shape_matches", {"position": 0},
                            "normalized_shape must match the trailing dimensions")),
             infer=_infer_norm))
    add(_reg(name="groupnorm", min_inputs=1, max_inputs=1, group="norm",
             requirements=(RANK2, FLOAT_IN,
                           ("internal", "channel_divisible_by_groups", {"position": 0},
                            "channels must be divisible by groups")),
             infer=_infer_norm))
    add(_reg(name="instancenorm", min_inputs=1, max_inputs=1, group="norm",
             requirements=(RANK2, FLOAT_IN), infer=_infer_norm))
    add(_reg(name="rmsnorm", min_inputs=1, max_inputs=1, group="norm",
             requirements=(RANK1, FLOAT_IN), infer=_infer_norm))
    add(_reg(name="l2norm", min_inputs=1, max_inputs=1, group="norm",
             requirements=(RANK1, FLOAT_IN), infer=_infer_norm))

    # -- shape manipulation ------------------------------------------------
    add(_reg(name="reshape", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY, ("internal", "reshape_numel_preserved", {"position": 0},
                                      "reshape must preserve the element count")),
             infer=_infer_reshape))
    add(_reg(name="view", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY, ("internal", "reshape_numel_preserved", {"position": 0},
                                      "view must preserve the element count")),
             infer=_infer_reshape))
    add(_reg(name="flatten", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY,), infer=_infer_flatten))
    add(_reg(name="permute", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY, ("internal", "permutation_valid", {"position": 0},
                                      "dims must be a permutation of the input rank")),
             infer=_infer_permute))
    add(_reg(name="transpose", min_inputs=1, max_inputs=1, group="shape",
             requirements=(RANK2, ("internal", "transpose_dims_valid", {"position": 0},
                                   "transpose dimensions must be distinct and in range")),
             infer=None))
    add(_reg(name="squeeze", min_inputs=1, max_inputs=1, group="shape",
             requirements=(RANK1, ("internal", "squeeze_dim_is_one", {"position": 0},
                                   "the squeezed dimension must have size 1")),
             infer=_infer_squeeze))
    add(_reg(name="unsqueeze", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY,), infer=_infer_unsqueeze))
    add(_reg(name="expand", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY, ("internal", "expandable_to", {"position": 0},
                                      "tensor must be broadcastable to the target shape")),
             infer=_infer_expand))
    add(_reg(name="broadcast_to", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY, ("internal", "expandable_to", {"position": 0},
                                      "tensor must be broadcastable to the target shape")),
             infer=_infer_expand))
    add(_reg(name="concat", min_inputs=2, max_inputs=-1, group="shape",
             requirements=(("internal", "concat_dims_match", {"positions": "all"},
                            "all inputs must share every non-concatenated dimension"),),
             infer=_infer_concat))
    add(_reg(name="stack", min_inputs=2, max_inputs=-1, group="shape",
             requirements=(("internal", "shapes_equal", {"positions": "all"},
                            "all inputs must have identical shapes"),),
             infer=_infer_stack))
    add(_reg(name="split", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY, ("internal", "splittable", {"position": 0},
                                      "the split dimension must divide evenly")),
             infer=_infer_split))
    add(_reg(name="chunk", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY,), infer=_infer_split))
    add(_reg(name="getitem", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY, ("internal", "index_in_range", {"position": 0},
                                      "index must address an existing dimension")),
             infer=_infer_getitem))
    add(_reg(name="pad", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY, ("internal", "pad_width_valid", {"position": 0},
                                      "pad list must have at most two entries per dimension")),
             infer=_infer_pad))
    add(_reg(name="pixel_shuffle", min_inputs=1, max_inputs=1, group="shape",
             requirements=(RANK4, ("internal", "channel_divisible", {"position": 0},
                                   "channels must be divisible by upscale_factor^2")),
             infer=_infer_pixel_shuffle))
    add(_reg(name="pixel_unshuffle", min_inputs=1, max_inputs=1, group="shape",
             requirements=(RANK4,), infer=None))
    add(_reg(name="tile", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY,), infer=_infer_tile))
    add(_reg(name="repeat_interleave", min_inputs=1, max_inputs=1, group="shape",
             requirements=(NONEMPTY,), infer=None))
    add(_reg(name="interpolate", min_inputs=1, max_inputs=1, group="shape",
             requirements=(RANK4, FLOAT_IN,
                           ("internal", "interpolation_align_valid", {"position": 0},
                            "align_corners is only defined for linear/cubic modes")),
             infer=_infer_interpolate))

    # -- attention ---------------------------------------------------------
    add(_reg(name="sdpa", min_inputs=3, max_inputs=4, group="attention",
             requirements=(("input", "rank_eq", {"position": 0, "rank": 4},
                            "query must be 4-D (B,H,L,E)"),
                           ("internal", "attention_dims_match", {"positions": [0, 1, 2]},
                            "query/key/value embedding and head dims must agree"),
                           ("internal", "dtype_agreement", {"positions": [0, 1, 2]},
                            "query/key/value must share a dtype"),
                           FLOAT_IN),
             infer=_infer_sdpa))
    add(_reg(name="scaled_dot_product_attention", min_inputs=3, max_inputs=4, group="attention",
             requirements=(("internal", "attention_dims_match", {"positions": [0, 1, 2]},
                            "query/key/value embedding and head dims must agree"),),
             infer=_infer_sdpa))

    # -- embeddings / misc -------------------------------------------------
    add(_reg(name="embedding", min_inputs=2, max_inputs=2, group="misc",
             param_positions=(1,),
             requirements=(("input", "dtype_family", {"position": 0, "family": "integral"},
                            "indices must be integral"),
                           ("internal", "index_in_range", {"position": 0},
                            "indices must be within the embedding table")),
             infer=_infer_embedding))
    add(_reg(name="topk", min_inputs=1, max_inputs=1, group="misc",
             requirements=(NONEMPTY,), infer=_infer_topk))
    add(_reg(name="identity", min_inputs=1, max_inputs=1, group="misc",
             requirements=(NONEMPTY,), infer=_infer_identity))
    return R


OP_REGISTRY: Dict[str, OpDef] = _build_registry()

#: Aliases accepted when writing seed models or mutation replacements.
OP_ALIASES: Dict[str, str] = {
    "bn": "batchnorm",
    "bn2d": "batchnorm",
    "batchnorm2d": "batchnorm",
    "batch_norm": "batchnorm",
    "ln": "layernorm",
    "layer_norm": "layernorm",
    "gn": "groupnorm",
    "group_norm": "groupnorm",
    "in": "_invalid_",
    "relu6": "hardswish",
    "swish": "silu",
    "hard_swish": "hardswish",
    "leakyrelu": "leaky_relu",
    "matmul_": "matmul",
    "add_": "add",
    "mul_": "mul",
    "cat": "concat",
    "concatenate": "concat",
    "view_": "view",
    "transpose_": "transpose",
    "unsqueeze_": "unsqueeze",
    "squeeze_": "squeeze",
    "avg_pool2d": "avgpool2d",
    "max_pool2d": "maxpool2d",
    "maxpool": "maxpool2d",
    "avgpool": "avgpool2d",
    "global_avg_pool": "global_avgpool",
    "global_max_pool": "global_maxpool",
    "adaptive_avg_pool2d": "adaptive_avgpool2d",
    "adaptive_max_pool2d": "adaptive_maxpool2d",
    "logsoftmax": "log_softmax",
    "reshape_": "reshape",
    "unsqueeze_dim": "unsqueeze",
    "stopgradient": "stop_gradient",
    "arg_max": "argmax",
    "arg_min": "argmin",
}


def canonical_op(name: str) -> str:
    """Map an alias such as ``bn`` to its canonical op name."""
    if name in OP_REGISTRY:
        return name
    return OP_ALIASES.get(name, name)


def get_op(name: str) -> OpDef:
    op = canonical_op(name)
    if op not in OP_REGISTRY:
        raise KeyError(f"unknown IR op {name!r}")
    return OP_REGISTRY[op]


def op_names() -> List[str]:
    return sorted(OP_REGISTRY)


# ---------------------------------------------------------------------------
# Requirement instantiation
# ---------------------------------------------------------------------------

def _resolve_arg(value: Any, node) -> Any:
    if isinstance(value, str):
        if value == "all":
            return list(range(len(node.inputs)))
        if value == "groups":
            return int(node.attrs.get("groups", 1))
        if value == "transposed":
            return bool(node.attrs.get("transposed", False))
        return value
    if isinstance(value, list):
        return [_resolve_arg(v, node) for v in value]
    return value


def requirements_for(node) -> List[OpRequirement]:
    """Instantiate ``R(v)`` for ``node`` (an ``OpRequirement`` list)."""
    op = canonical_op(node.op)
    if op not in OP_REGISTRY:
        return []
    defn = OP_REGISTRY[op]
    out: List[OpRequirement] = []
    n_inputs = len(node.inputs)
    for (kind, predicate, args, desc) in defn.requirements:
        resolved = {k: _resolve_arg(v, node) for k, v in args.items()}
        # Drop requirements that address an input the node does not have.
        pos = resolved.get("position")
        if isinstance(pos, int) and pos >= n_inputs:
            continue
        positions = resolved.get("positions")
        if isinstance(positions, list) and any(
                isinstance(p, int) and p >= n_inputs for p in positions):
            continue
        out.append(OpRequirement(kind=kind, predicate=predicate, args=resolved,
                                 description=desc))
    return out


def infer_shape(op: str, inputs: Sequence[TensorSpec],
                attrs: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """Static shape inference used while constructing replacement subgraphs."""
    name = canonical_op(op)
    if name not in OP_REGISTRY:
        return None
    defn = OP_REGISTRY[name]
    if defn.name in ("input", "param"):
        shape = attrs.get("shape") or attrs.get("_shape")
        return tuple(int(d) for d in shape) if shape else None
    if defn.infer is None:
        return None
    try:
        return defn.infer(list(inputs), dict(attrs))
    except Exception:
        return None


def attach_requirements(graph) -> None:
    """Recompute ``R(v)`` for every node of ``graph`` (used after mutation)."""
    for node in graph.nodes.values():
        node.requirements = requirements_for(node)
