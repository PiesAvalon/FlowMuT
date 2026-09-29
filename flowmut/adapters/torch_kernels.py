"""PyTorch implementations of every IR operator.

Each kernel takes ``(inputs, attrs)`` -- the already-materialised input tensors
and the node attributes -- and returns either one tensor or a sequence of
tensors (for multi-output ops such as ``split``/``topk``).

Kernels are intentionally plain functions so that the same table can be reused
by the eager and the compiled adapter.  Nothing here inspects the TFG: the
adapter is responsible for profiling and tracing.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from flowmut.ir.ops import canonical_op

Kernel = Callable[[List[torch.Tensor], Dict[str, Any]], Union[torch.Tensor, Sequence[torch.Tensor]]]

KERNELS: Dict[str, Kernel] = {}

#: Ops that have no PyTorch counterpart (they only exist in the IR).
UNSUPPORTED = {"groups"}


def kernel(op: str):
    def deco(fn: Kernel) -> Kernel:
        KERNELS[op] = fn
        return fn
    return deco


class KernelMissing(NotImplementedError):
    pass


def apply(op: str, inputs: Sequence[torch.Tensor], attrs: Dict[str, Any]
          ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    name = canonical_op(op)
    fn = KERNELS.get(name)
    if fn is None:
        raise KernelMissing(f"no PyTorch kernel for IR op {op!r}")
    return fn(list(inputs), dict(attrs))


def _dim(attrs: Dict[str, Any], key: str, default: Any = -1) -> Any:
    return attrs.get(key, default)


def _pair(value: Any, default: int = 1) -> Tuple[int, int]:
    if value is None:
        return (default, default)
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


# ---------------------------------------------------------------------------
# Elementwise unary
# ---------------------------------------------------------------------------

@kernel("relu")
def k_relu(x, a):
    return F.relu(x[0], inplace=bool(a.get("inplace", False)))


@kernel("gelu")
def k_gelu(x, a):
    return F.gelu(x[0], approximate=a.get("approximate", "none"))


@kernel("quick_gelu")
def k_quick_gelu(x, a):
    return x[0] * torch.sigmoid(1.702 * x[0])


@kernel("silu")
def k_silu(x, a):
    return F.silu(x[0])


@kernel("mish")
def k_mish(x, a):
    return F.mish(x[0])


@kernel("hardswish")
def k_hardswish(x, a):
    return F.hardswish(x[0])


@kernel("hardsigmoid")
def k_hardsigmoid(x, a):
    return F.hardsigmoid(x[0])


@kernel("elu")
def k_elu(x, a):
    return F.elu(x[0], alpha=float(a.get("alpha", 1.0)))


@kernel("leaky_relu")
def k_leaky_relu(x, a):
    return F.leaky_relu(x[0], negative_slope=float(a.get("negative_slope", 0.01)))


@kernel("softplus")
def k_softplus(x, a):
    return F.softplus(x[0], beta=float(a.get("beta", 1.0)),
                      threshold=float(a.get("threshold", 20.0)))


@kernel("sigmoid")
def k_sigmoid(x, a):
    return torch.sigmoid(x[0])


@kernel("tanh")
def k_tanh(x, a):
    return torch.tanh(x[0])


@kernel("exp")
def k_exp(x, a):
    return torch.exp(x[0])


@kernel("log")
def k_log(x, a):
    return torch.log(x[0])


@kernel("sqrt")
def k_sqrt(x, a):
    return torch.sqrt(x[0])


@kernel("rsqrt")
def k_rsqrt(x, a):
    return torch.rsqrt(x[0])


@kernel("reciprocal")
def k_reciprocal(x, a):
    return torch.reciprocal(x[0])


@kernel("sin")
def k_sin(x, a):
    return torch.sin(x[0])


@kernel("cos")
def k_cos(x, a):
    return torch.cos(x[0])


@kernel("erf")
def k_erf(x, a):
    return torch.erf(x[0])


@kernel("floor")
def k_floor(x, a):
    return torch.floor(x[0])


@kernel("round")
def k_round(x, a):
    return torch.round(x[0])


@kernel("sign")
def k_sign(x, a):
    return torch.sign(x[0])


@kernel("neg")
def k_neg(x, a):
    return torch.neg(x[0])


@kernel("abs")
def k_abs(x, a):
    return torch.abs(x[0])


@kernel("square")
def k_square(x, a):
    return torch.square(x[0])


@kernel("clamp")
def k_clamp(x, a):
    lo = a.get("min")
    hi = a.get("max")
    if lo is None and hi is None:
        lo, hi = -1.0, 1.0
    return torch.clamp(x[0], min=lo, max=hi)


@kernel("dropout")
def k_dropout(x, a):
    # Deterministic by design: mutants must be compared under identical
    # stochastic conditions, so dropout behaves as the identity here.
    return x[0]


@kernel("identity")
def k_identity(x, a):
    return x[0]


@kernel("clone")
def k_clone(x, a):
    return x[0].clone()


@kernel("detach")
def k_detach(x, a):
    return x[0].detach()


@kernel("stop_gradient")
def k_stop_gradient(x, a):
    return x[0].detach()


@kernel("contiguous")
def k_contiguous(x, a):
    memory_format = torch.channels_last if a.get("memory_format") == "channels_last" \
        else torch.contiguous_format
    return x[0].contiguous(memory_format=memory_format)


@kernel("cast")
def k_cast(x, a):
    return x[0].to(_torch_dtype(a.get("dtype", "float32")))


@kernel("flip")
def k_flip(x, a):
    dims = a.get("dims")
    if dims is None:
        dims = a.get("dim", 0)
    if isinstance(dims, int):
        dims = [dims]
    return torch.flip(x[0], dims=list(dims))


@kernel("roll")
def k_roll(x, a):
    shifts = a.get("shifts", 1)
    dims = a.get("dims")
    if dims is None:
        dims = a.get("dim", 0)
    return torch.roll(x[0], shifts=shifts, dims=dims)


# ---------------------------------------------------------------------------
# Elementwise binary / ternary
# ---------------------------------------------------------------------------

@kernel("add")
def k_add(x, a):
    other = a.get("other")
    if other is None:
        other = a.get("alpha")
    if other is not None and len(x) < 2:
        return torch.add(x[0], float(other))
    return torch.add(x[0], x[1])


@kernel("sub")
def k_sub(x, a):
    other = a.get("other")
    if other is not None and len(x) < 2:
        return torch.sub(x[0], float(other))
    return torch.sub(x[0], x[1])


@kernel("mul")
def k_mul(x, a):
    other = a.get("other")
    if other is not None and len(x) < 2:
        return torch.mul(x[0], float(other))
    return torch.mul(x[0], x[1])


@kernel("div")
def k_div(x, a):
    other = a.get("other")
    if other is not None and len(x) < 2:
        return torch.div(x[0], float(other))
    return torch.div(x[0], x[1])


@kernel("pow")
def k_pow(x, a):
    exponent = a.get("exponent")
    if exponent is not None:
        return torch.pow(x[0], float(exponent))
    return torch.pow(x[0], x[1])


@kernel("maximum")
def k_maximum(x, a):
    return torch.maximum(x[0], x[1])


@kernel("minimum")
def k_minimum(x, a):
    return torch.minimum(x[0], x[1])


@kernel("where")
def k_where(x, a):
    cond = x[0]
    if cond.dtype != torch.bool:
        cond = cond.to(torch.bool)
    return torch.where(cond, x[1], x[2])


@kernel("masked_fill")
def k_masked_fill(x, a):
    mask = x[1]
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    return x[0].masked_fill(mask, float(a.get("value", 0.0)))


# ---------------------------------------------------------------------------
# Activations with dimension
# ---------------------------------------------------------------------------

@kernel("softmax")
def k_softmax(x, a):
    return torch.softmax(x[0], dim=int(_dim(a, "dim", -1)))


@kernel("log_softmax")
def k_log_softmax(x, a):
    return torch.log_softmax(x[0], dim=int(_dim(a, "dim", -1)))


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def _reduce(fn):
    def run(x, a):
        dim = a.get("dim", None)
        keepdim = bool(a.get("keepdim", False))
        if isinstance(dim, list):
            dim = tuple(dim)
        if dim is None:
            return fn(x[0])
        return fn(x[0], dim=dim, keepdim=keepdim)
    return run


KERNELS["mean"] = _reduce(lambda t, dim=None, keepdim=False: torch.mean(t, dim=dim, keepdim=keepdim)
                          if dim is not None else torch.mean(t))
KERNELS["sum"] = _reduce(lambda t, dim=None, keepdim=False: torch.sum(t, dim=dim, keepdim=keepdim)
                         if dim is not None else torch.sum(t))
KERNELS["amax"] = _reduce(lambda t, dim=None, keepdim=False: torch.amax(t, dim=dim, keepdim=keepdim)
                          if dim is not None else torch.amax(t))
KERNELS["amin"] = _reduce(lambda t, dim=None, keepdim=False: torch.amin(t, dim=dim, keepdim=keepdim)
                          if dim is not None else torch.amin(t))
KERNELS["prod"] = _reduce(lambda t, dim=None, keepdim=False: torch.prod(t, dim=dim, keepdim=keepdim)
                          if dim is not None else torch.prod(t))


def k_var(x, a):
    dim = a.get("dim", None)
    keepdim = bool(a.get("keepdim", False))
    correction = int(a.get("correction", 1))
    if dim is None:
        return torch.var(x[0], correction=correction)
    if isinstance(dim, list):
        dim = tuple(dim)
    return torch.var(x[0], dim=dim, keepdim=keepdim, correction=correction)


def k_std(x, a):
    dim = a.get("dim", None)
    keepdim = bool(a.get("keepdim", False))
    correction = int(a.get("correction", 1))
    if dim is None:
        return torch.std(x[0], correction=correction)
    if isinstance(dim, list):
        dim = tuple(dim)
    return torch.std(x[0], dim=dim, keepdim=keepdim, correction=correction)


KERNELS["var"] = k_var
KERNELS["std"] = k_std


def k_argmax(x, a):
    dim = a.get("dim", None)
    keepdim = bool(a.get("keepdim", False))
    if dim is None:
        return torch.argmax(x[0])
    return torch.argmax(x[0], dim=int(dim), keepdim=keepdim)


def k_argmin(x, a):
    dim = a.get("dim", None)
    keepdim = bool(a.get("keepdim", False))
    if dim is None:
        return torch.argmin(x[0])
    return torch.argmin(x[0], dim=int(dim), keepdim=keepdim)


KERNELS["argmax"] = k_argmax
KERNELS["argmin"] = k_argmin


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------

@kernel("matmul")
def k_matmul(x, a):
    return torch.matmul(x[0], x[1])


@kernel("bmm")
def k_bmm(x, a):
    return torch.bmm(x[0], x[1])


@kernel("linear")
def k_linear(x, a):
    bias = x[2] if len(x) > 2 else None
    return F.linear(x[0], x[1], bias)


@kernel("einsum")
def k_einsum(x, a):
    equation = a.get("equation")
    if not equation:
        raise ValueError("einsum requires an 'equation' attribute")
    return torch.einsum(equation, *x)


@kernel("sdpa")
def k_sdpa(x, a):
    attn_mask = x[3] if len(x) > 3 else None
    dropout_p = float(a.get("dropout_p", 0.0))
    is_causal = bool(a.get("is_causal", False))
    return F.scaled_dot_product_attention(x[0], x[1], x[2], attn_mask=attn_mask,
                                          dropout_p=dropout_p, is_causal=is_causal)


KERNELS["scaled_dot_product_attention"] = k_sdpa


# ---------------------------------------------------------------------------
# Convolution / pooling
# ---------------------------------------------------------------------------

@kernel("conv2d")
def k_conv2d(x, a):
    bias = x[2] if len(x) > 2 else None
    return F.conv2d(x[0], x[1], bias,
                    stride=_pair(a.get("stride"), 1),
                    padding=_pair(a.get("padding"), 0),
                    dilation=_pair(a.get("dilation"), 1),
                    groups=int(a.get("groups", 1) or 1))


@kernel("conv1d")
def k_conv1d(x, a):
    bias = x[2] if len(x) > 2 else None
    stride = a.get("stride", 1)
    padding = a.get("padding", 0)
    dilation = a.get("dilation", 1)
    return F.conv1d(x[0], x[1], bias,
                    stride=(int(stride[0]) if isinstance(stride, (tuple, list)) else int(stride)),
                    padding=(int(padding[0]) if isinstance(padding, (tuple, list)) else int(padding)),
                    dilation=(int(dilation[0]) if isinstance(dilation, (tuple, list)) else int(dilation)),
                    groups=int(a.get("groups", 1) or 1))


@kernel("conv_transpose2d")
def k_conv_transpose2d(x, a):
    bias = x[2] if len(x) > 2 else None
    return F.conv_transpose2d(x[0], x[1], bias,
                              stride=_pair(a.get("stride"), 1),
                              padding=_pair(a.get("padding"), 0),
                              output_padding=_pair(a.get("output_padding"), 0),
                              groups=int(a.get("groups", 1) or 1),
                              dilation=_pair(a.get("dilation"), 1))


@kernel("maxpool2d")
def k_maxpool2d(x, a):
    return F.max_pool2d(x[0], kernel_size=_pair(a.get("kernel_size"), 2),
                        stride=_pair(a.get("stride"), None) if a.get("stride") is not None else None,
                        padding=_pair(a.get("padding"), 0),
                        dilation=_pair(a.get("dilation"), 1),
                        ceil_mode=bool(a.get("ceil_mode", False)))


@kernel("avgpool2d")
def k_avgpool2d(x, a):
    return F.avg_pool2d(x[0], kernel_size=_pair(a.get("kernel_size"), 2),
                        stride=_pair(a.get("stride"), None) if a.get("stride") is not None else None,
                        padding=_pair(a.get("padding"), 0),
                        ceil_mode=bool(a.get("ceil_mode", False)),
                        count_include_pad=bool(a.get("count_include_pad", True)))


@kernel("adaptive_avgpool2d")
def k_adaptive_avgpool2d(x, a):
    out = a.get("output_size", 1)
    out = tuple(int(v) for v in out) if isinstance(out, (tuple, list)) else (int(out), int(out))
    return F.adaptive_avg_pool2d(x[0], out)


@kernel("adaptive_maxpool2d")
def k_adaptive_maxpool2d(x, a):
    out = a.get("output_size", 1)
    out = tuple(int(v) for v in out) if isinstance(out, (tuple, list)) else (int(out), int(out))
    return F.adaptive_max_pool2d(x[0], out)


@kernel("global_avgpool")
def k_global_avgpool(x, a):
    t = x[0]
    keepdim = bool(a.get("keepdim", False))
    if not a.get("spatial", True) or t.dim() < 3:
        return torch.mean(t, dim=-1, keepdim=keepdim)
    dims = tuple(range(2, t.dim()))
    return torch.mean(t, dim=dims, keepdim=keepdim)


@kernel("global_maxpool")
def k_global_maxpool(x, a):
    t = x[0]
    keepdim = bool(a.get("keepdim", False))
    if not a.get("spatial", True) or t.dim() < 3:
        return torch.amax(t, dim=-1, keepdim=keepdim)
    dims = tuple(range(2, t.dim()))
    return torch.amax(t, dim=dims, keepdim=keepdim)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@kernel("batchnorm")
def k_batchnorm(x, a):
    t = x[0]
    num_features = a.get("num_features")
    if num_features is None:
        num_features = t.shape[1] if t.dim() > 1 else t.shape[0]
    channels = int(num_features)
    weight = x[1] if len(x) > 1 else None
    bias = x[2] if len(x) > 2 else None
    eps = float(a.get("eps", 1e-5))
    # ``F.batch_norm`` expects 1-D running statistics of length C; it broadcasts
    # them over the spatial dimensions itself.  Inference mode is used so that a
    # mutant never mutates the running statistics of the seed model.
    mean = torch.zeros(channels, dtype=t.dtype, device=t.device)
    var = torch.ones(channels, dtype=t.dtype, device=t.device)
    if weight is not None:
        weight = weight.reshape(channels)
    if bias is not None:
        bias = bias.reshape(channels)
    return F.batch_norm(t, mean, var, weight, bias, training=False,
                        momentum=float(a.get("momentum", 0.1)), eps=eps)


@kernel("layernorm")
def k_layernorm(x, a):
    t = x[0]
    norm = a.get("normalized_shape")
    if norm is None:
        norm = (t.shape[-1],)
    if isinstance(norm, int):
        norm = (norm,)
    weight = x[1] if len(x) > 1 else None
    bias = x[2] if len(x) > 2 else None
    return F.layer_norm(t, tuple(int(v) for v in norm), weight, bias,
                        eps=float(a.get("eps", 1e-5)))


@kernel("groupnorm")
def k_groupnorm(x, a):
    t = x[0]
    groups = int(a.get("num_groups", 1) or 1)
    weight = x[1] if len(x) > 1 else None
    bias = x[2] if len(x) > 2 else None
    return F.group_norm(t, groups, weight, bias, eps=float(a.get("eps", 1e-5)))


@kernel("instancenorm")
def k_instancenorm(x, a):
    return F.instance_norm(x[0], eps=float(a.get("eps", 1e-5)))


@kernel("rmsnorm")
def k_rmsnorm(x, a):
    t = x[0]
    norm = a.get("normalized_shape")
    if norm is None:
        norm = (t.shape[-1],)
    if isinstance(norm, int):
        norm = (norm,)
    dims = tuple(range(t.dim() - len(norm), t.dim()))
    eps = float(a.get("eps", 1e-6))
    weight = x[1] if len(x) > 1 else None
    out = t * torch.rsqrt(t.pow(2).mean(dim=dims, keepdim=True) + eps)
    if weight is not None:
        out = out * weight
    return out


@kernel("l2norm")
def k_l2norm(x, a):
    return F.normalize(x[0], p=2.0, dim=int(_dim(a, "dim", -1)), eps=float(a.get("eps", 1e-12)))


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------

def _target_shape(t: torch.Tensor, attrs: Dict[str, Any]) -> List[int]:
    shape = attrs.get("shape") or attrs.get("size")
    if shape is None:
        raise ValueError("reshape/view requires a 'shape' attribute")
    shape = [int(v) for v in shape]
    if -1 in shape:
        known = 1
        for v in shape:
            if v > 0:
                known *= v
        if known == 0:
            raise ValueError("cannot infer the -1 dimension")
        shape[shape.index(-1)] = max(1, t.numel() // known)
    return shape


@kernel("reshape")
def k_reshape(x, a):
    return torch.reshape(x[0], _target_shape(x[0], a))


@kernel("view")
def k_view(x, a):
    return x[0].reshape(_target_shape(x[0], a))


@kernel("flatten")
def k_flatten(x, a):
    return torch.flatten(x[0], start_dim=int(a.get("start_dim", 1)),
                         end_dim=int(a.get("end_dim", -1)))


@kernel("permute")
def k_permute(x, a):
    dims = a.get("dims")
    if dims is None:
        raise ValueError("permute requires 'dims'")
    return x[0].permute(*[int(d) for d in dims])


@kernel("transpose")
def k_transpose(x, a):
    return torch.transpose(x[0], int(a.get("dim0", 0)), int(a.get("dim1", 1)))


@kernel("squeeze")
def k_squeeze(x, a):
    dim = a.get("dim")
    if dim is None:
        return torch.squeeze(x[0])
    return torch.squeeze(x[0], int(dim))


@kernel("unsqueeze")
def k_unsqueeze(x, a):
    return torch.unsqueeze(x[0], int(a.get("dim", 0)))


@kernel("expand")
def k_expand(x, a):
    shape = a.get("shape")
    if shape is None:
        raise ValueError("expand requires a target 'shape'")
    return x[0].expand(*[int(v) for v in shape])


@kernel("broadcast_to")
def k_broadcast_to(x, a):
    shape = a.get("shape")
    if shape is None:
        raise ValueError("broadcast_to requires a target 'shape'")
    return torch.broadcast_to(x[0], [int(v) for v in shape])


@kernel("concat")
def k_concat(x, a):
    return torch.cat(list(x), dim=int(_dim(a, "dim", 0)))


@kernel("stack")
def k_stack(x, a):
    return torch.stack(list(x), dim=int(_dim(a, "dim", 0)))


@kernel("split")
def k_split(x, a):
    dim = int(_dim(a, "dim", 0))
    sections = a.get("sections")
    n = int(a.get("num_outputs", 2) or 2)
    if sections is not None:
        if isinstance(sections, int):
            size = sections
            remainder = x[0].shape[dim] - size * n
            parts = [size] * n
            if parts:
                parts[-1] += remainder
        else:
            parts = [int(v) for v in sections]
        return torch.split(x[0], parts, dim=dim)
    size = x[0].shape[dim] // max(1, n)
    if size == 0:
        raise ValueError(f"cannot split dimension {dim} of size {x[0].shape[dim]} into {n} parts")
    return torch.split(x[0], size, dim=dim)


@kernel("chunk")
def k_chunk(x, a):
    chunks = int(a.get("num_outputs", 2) or 2)
    return torch.chunk(x[0], chunks, dim=int(_dim(a, "dim", 0)))


@kernel("getitem")
def k_getitem(x, a):
    index = a.get("index", 0)
    dim = int(a.get("dim", 0))
    if not isinstance(index, int):
        raise ValueError("getitem supports integer indices only")
    return torch.select(x[0], dim, index)


@kernel("pad")
def k_pad(x, a):
    pad = a.get("pad")
    if pad is None:
        raise ValueError("pad requires a 'pad' attribute")
    mode = a.get("mode", "constant")
    if mode == "constant":
        return F.pad(x[0], tuple(int(v) for v in pad), mode="constant",
                     value=float(a.get("value", 0.0)))
    if mode in ("reflect", "replicate", "circular"):
        return F.pad(x[0], tuple(int(v) for v in pad), mode=mode)
    return F.pad(x[0], tuple(int(v) for v in pad), mode="constant",
                 value=float(a.get("value", 0.0)))


@kernel("pixel_shuffle")
def k_pixel_shuffle(x, a):
    return F.pixel_shuffle(x[0], int(a.get("upscale_factor", 2)))


@kernel("pixel_unshuffle")
def k_pixel_unshuffle(x, a):
    return F.pixel_unshuffle(x[0], int(a.get("upscale_factor", 2)))


@kernel("tile")
def k_tile(x, a):
    reps = a.get("reps")
    if reps is None:
        raise ValueError("tile requires 'reps'")
    return torch.tile(x[0], [int(v) for v in reps])


@kernel("repeat_interleave")
def k_repeat_interleave(x, a):
    repeats = a.get("repeats", 2)
    dim = a.get("dim", None)
    if isinstance(repeats, (tuple, list)):
        repeats = torch.tensor([int(v) for v in repeats], device=x[0].device)
    return torch.repeat_interleave(x[0], repeats if isinstance(repeats, torch.Tensor)
                                  else int(repeats), dim=dim)


@kernel("interpolate")
def k_interpolate(x, a):
    size = a.get("size")
    scale = a.get("scale_factor")
    mode = a.get("mode", "nearest")
    kwargs: Dict[str, Any] = {}
    align = a.get("align_corners")
    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = bool(align) if align is not None else False
    if a.get("recompute_scale_factor") is not None:
        kwargs["recompute_scale_factor"] = bool(a["recompute_scale_factor"])
    if mode == "nearest" and a.get("antialias") is not None:
        kwargs["antialias"] = bool(a["antialias"])
    if size is not None:
        return F.interpolate(x[0], size=tuple(int(v) for v in size), mode=mode, **kwargs)
    if scale is not None:
        if isinstance(scale, (tuple, list)):
            scale = tuple(float(v) for v in scale)
        else:
            scale = float(scale)
        return F.interpolate(x[0], scale_factor=scale, mode=mode, **kwargs)
    raise ValueError("interpolate requires 'size' or 'scale_factor'")


# ---------------------------------------------------------------------------
# Embedding / misc
# ---------------------------------------------------------------------------

@kernel("embedding")
def k_embedding(x, a):
    ids = x[0]
    if ids.dtype not in (torch.int64, torch.int32):
        ids = ids.to(torch.int64)
    return F.embedding(ids, x[1])


@kernel("topk")
def k_topk(x, a):
    k = int(a.get("k", 1))
    dim = int(_dim(a, "dim", -1))
    largest = bool(a.get("largest", True))
    sorted_ = bool(a.get("sorted", True))
    k = max(1, min(k, x[0].shape[dim]))
    return torch.topk(x[0], k, dim=dim, largest=largest, sorted=sorted_)


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_TORCH_DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "complex64": torch.complex64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _torch_dtype(name: str) -> torch.dtype:
    return _TORCH_DTYPES.get(str(name), torch.float32)


def dtype_name(dtype: torch.dtype) -> str:
    for name, dt in _TORCH_DTYPES.items():
        if dt == dtype:
            return name
    return str(dtype).replace("torch.", "")


#: Ops whose kernels return more than one tensor.
MULTI_OUTPUT_DEFAULT: Dict[str, int] = {"split": 2, "chunk": 2, "topk": 2}


def init_parameter(tensor: torch.Tensor, init: str, generator: torch.Generator) -> None:
    """Deterministic parameter initialisation driven by ``init``."""
    shape = list(tensor.shape)
    if not shape:
        tensor.zero_()
        return
    if len(shape) >= 2 and init in ("kaiming_uniform", "kaiming_normal"):
        if len(shape) == 2:
            fan_in, fan_out = shape[1], shape[0]
        else:
            receptive = 1
            for d in shape[2:]:
                receptive *= d
            fan_in, fan_out = shape[1] * receptive, shape[0] * receptive
    else:
        fan_in = shape[0] if shape else 1
        fan_out = shape[-1] if len(shape) > 1 else shape[0]
    with torch.no_grad():
        if init == "zeros":
            tensor.zero_()
        elif init == "ones":
            tensor.fill_(1.0)
        elif init == "kaiming_normal":
            std = math.sqrt(2.0 / max(1, fan_in))
            tensor.normal_(0.0, std, generator=generator)
        elif init == "xavier_uniform":
            bound = math.sqrt(6.0 / max(1, fan_in + fan_out))
            tensor.uniform_(-bound, bound, generator=generator)
        elif init == "normal":
            tensor.normal_(0.0, 0.02, generator=generator)
        elif init == "truncated_normal":
            tensor.normal_(0.0, 0.02, generator=generator)
            tensor.clamp_(-0.04, 0.04)
        elif init == "uniform":
            tensor.uniform_(-0.1, 0.1, generator=generator)
        elif init == "identity":
            tensor.zero_()
            n = min(shape[0], shape[1] if len(shape) > 1 else shape[0])
            for i in range(n):
                tensor[i, i] = 1.0
        else:  # kaiming_uniform
            bound = math.sqrt(6.0 / max(1, fan_in))
            tensor.uniform_(-bound, bound, generator=generator)
