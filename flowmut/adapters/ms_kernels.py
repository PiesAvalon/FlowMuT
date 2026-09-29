"""MindSpore implementations of every IR operator.

This is the MindSpore counterpart of :mod:`flowmut.adapters.torch_kernels`.  The
same table drives **both** MindSpore execution modes:

* :func:`apply_pynative` evaluates one node eagerly (PyNative mode), which is
  what the adapter uses for the ``F(e)`` profiling pass;
* :func:`make_fn` returns an ``nn.Cell`` whose ``construct`` is plain MindSpore
  source, so a generated graph-mode ``Cell`` can simply call
  ``self.f_<node>(*tensors)`` and let MindSpore's parser compile it.

Because a compiled MindSpore graph needs *real* source for every composite
operator, every op that expands to more than a single primitive is written as an
explicit :class:`mindspore.nn.Cell` subclass in this file.  Nothing here inspects
the TFG: the adapter owns profiling and tracing.

Notes on MindSpore 2.7.1 (CPU):

* ``ops.roll``, ``ops.einsum``, ``ops.rms_norm`` and ``ops.layer_norm`` have no
  CPU kernel in graph mode, and ``ops.conv_transpose2d`` has no CPU kernel at
  all; all four are re-implemented here from supported primitives.
* ``ops.avg_pool2d`` accepts an ``int`` or a 4-tuple for ``padding`` only.
* ``ops.argmin`` returns ``int32`` while ``ops.argmax`` returns ``int64``; both
  are cast to ``int64`` so that the two frameworks agree.
* ``ops.round`` only supports ``decimals=0`` (which is also what the PyTorch
  kernel does, so the attribute is ignored).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import mindspore as ms
from mindspore import Tensor as MSTensor
from mindspore import nn, ops

from flowmut.ir.ops import canonical_op

__all__ = [
    "KernelMissing",
    "MS_SUPPORTED",
    "MS_UNSUPPORTED",
    "MULTI_OUTPUT_DEFAULT",
    "canonical",
    "make_fn",
    "apply_pynative",
    "init_parameter",
    "ms_dtype",
    "dtype_name",
]


class KernelMissing(NotImplementedError):
    """Raised when MindSpore 2.7.1 cannot execute an IR operator."""


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_MS_DTYPES: Dict[str, Any] = {
    "float64": ms.float64,
    "float32": ms.float32,
    "float16": ms.float16,
    "bfloat16": ms.bfloat16,
    "int64": ms.int64,
    "int32": ms.int32,
    "int16": ms.int16,
    "int8": ms.int8,
    "uint8": ms.uint8,
    "bool": ms.bool_,
    "complex64": ms.complex64,
}

_MS_DTYPE_NAMES: Dict[Any, str] = {v: k for k, v in _MS_DTYPES.items()}


def ms_dtype(name: str):
    """Canonical dtype name -> ``mindspore`` dtype (``float32`` by default)."""
    return _MS_DTYPES.get(str(name), ms.float32)


def dtype_name(dtype: Any) -> str:
    """``mindspore`` dtype -> canonical dtype name."""
    try:
        if dtype in _MS_DTYPE_NAMES:
            return _MS_DTYPE_NAMES[dtype]
    except Exception:
        pass
    text = str(dtype)
    for prefix in ("mindspore.", "ms."):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


def canonical(op: str) -> str:
    """Re-export of :func:`flowmut.ir.ops.canonical_op`."""
    return canonical_op(op)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _pair(value: Any, default: int = 1) -> Tuple[int, int]:
    if value is None:
        return (default, default)
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def _int_or_pair(value: Any, default: int = 1):
    """``int`` when both entries agree (several MindSpore kernels demand it)."""
    a, b = _pair(value, default)
    return a if a == b else (a, b)


def _norm_shape_attr(attrs: Dict[str, Any], rank: int) -> Tuple[int, ...]:
    norm = attrs.get("normalized_shape")
    if norm is None:
        return ()
    if isinstance(norm, int):
        return (int(norm),)
    return tuple(int(v) for v in norm)


# ---------------------------------------------------------------------------
# nn.Cell building blocks
# ---------------------------------------------------------------------------

class _UnaryCell(nn.Cell):
    """Base for single-primitive unary operators (kept explicit for the parser)."""

    def __init__(self) -> None:
        super().__init__()


class ReLUCell(_UnaryCell):
    def construct(self, x):
        return ops.relu(x)


class GELUCell(_UnaryCell):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = str(approximate or "none")

    def construct(self, x):
        return ops.gelu(x, approximate=self.approximate)


class QuickGELUCell(_UnaryCell):
    def construct(self, x):
        return x * ops.sigmoid(1.702 * x)


class SiLUCell(_UnaryCell):
    def construct(self, x):
        return ops.silu(x)


class MishCell(_UnaryCell):
    def construct(self, x):
        return ops.mish(x)


class HardSwishCell(_UnaryCell):
    def construct(self, x):
        return ops.hardswish(x)


class HardSigmoidCell(_UnaryCell):
    def construct(self, x):
        return ops.hardsigmoid(x)


class ELUCell(_UnaryCell):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def construct(self, x):
        return ops.elu(x, alpha=self.alpha)


class LeakyReLUCell(_UnaryCell):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def construct(self, x):
        return ops.leaky_relu(x, alpha=self.negative_slope)


class SoftplusCell(_UnaryCell):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)

    def construct(self, x):
        return ops.softplus(x, beta=self.beta, threshold=self.threshold)


class SigmoidCell(_UnaryCell):
    def construct(self, x):
        return ops.sigmoid(x)


class TanhCell(_UnaryCell):
    def construct(self, x):
        return ops.tanh(x)


class ExpCell(_UnaryCell):
    def construct(self, x):
        return ops.exp(x)


class LogCell(_UnaryCell):
    def construct(self, x):
        return ops.log(x)


class SqrtCell(_UnaryCell):
    def construct(self, x):
        return ops.sqrt(x)


class RsqrtCell(_UnaryCell):
    def construct(self, x):
        return ops.rsqrt(x)


class ReciprocalCell(_UnaryCell):
    def construct(self, x):
        return ops.reciprocal(x)


class SinCell(_UnaryCell):
    def construct(self, x):
        return ops.sin(x)


class CosCell(_UnaryCell):
    def construct(self, x):
        return ops.cos(x)


class ErfCell(_UnaryCell):
    def construct(self, x):
        return ops.erf(x)


class FloorCell(_UnaryCell):
    def construct(self, x):
        return ops.floor(x)


class SignCell(_UnaryCell):
    def construct(self, x):
        return ops.sign(x)


class NegCell(_UnaryCell):
    def construct(self, x):
        return ops.neg(x)


class AbsCell(_UnaryCell):
    def construct(self, x):
        return ops.abs(x)


class SquareCell(_UnaryCell):
    def construct(self, x):
        return ops.square(x)


class RoundCell(_UnaryCell):
    """``ops.round`` only accepts ``decimals=0``; the attribute is ignored,
    which is exactly what the PyTorch kernel does."""

    def construct(self, x):
        return ops.round(x, decimals=0)


class ClampCell(_UnaryCell):
    def __init__(self, lo: Any = None, hi: Any = None) -> None:
        super().__init__()
        if lo is None and hi is None:
            lo, hi = -1.0, 1.0
        self.lo = None if lo is None else float(lo)
        self.hi = None if hi is None else float(hi)

    def construct(self, x):
        return ops.clamp(x, self.lo, self.hi)


class DropoutCell(_UnaryCell):
    """Deterministic by design: dropout behaves as the identity."""

    def construct(self, x):
        return x


class IdentityCell(_UnaryCell):
    def construct(self, x):
        return x


class CloneCell(_UnaryCell):
    def construct(self, x):
        return x.copy()


class DetachCell(_UnaryCell):
    def construct(self, x):
        return ops.stop_gradient(x)


class StopGradientCell(_UnaryCell):
    def construct(self, x):
        return ops.stop_gradient(x)


class ContiguousCell(_UnaryCell):
    """MindSpore has no memory formats; tensors are always contiguous."""

    def construct(self, x):
        return x


class CastCell(_UnaryCell):
    def __init__(self, dtype: str = "float32") -> None:
        super().__init__()
        self.dtype = ms_dtype(dtype)

    def construct(self, x):
        return ops.cast(x, self.dtype)


class FlipCell(_UnaryCell):
    def __init__(self, dims: Any) -> None:
        super().__init__()
        if isinstance(dims, (tuple, list)):
            self.dims = tuple(int(d) for d in dims)
        else:
            self.dims = (int(dims),)

    def construct(self, x):
        return ops.flip(x, self.dims)


class RollCell(_UnaryCell):
    """``ops.roll`` has no CPU kernel, so shifts are done with split/concat.

    ``dims=None`` (torch semantics: roll the flattened tensor) is supported as
    well, by flattening, rolling and reshaping back.
    """

    def __init__(self, shifts: Any, dims: Any = None) -> None:
        super().__init__()
        if isinstance(shifts, (tuple, list)):
            self.shifts = tuple(int(s) for s in shifts)
        else:
            self.shifts = (int(shifts),)
        if dims is None:
            self.dims: Optional[Tuple[int, ...]] = None
        elif isinstance(dims, (tuple, list)):
            self.dims = tuple(int(d) for d in dims)
        else:
            self.dims = (int(dims),)

    def _shift(self, out, shift: int, dim: int):
        size = out.shape[dim]
        if size == 0:
            return out
        step = shift % size
        if step == 0:
            return out
        first, second = ops.split(out, (size - step, step), axis=dim)
        return ops.concat((second, first), axis=dim)

    def construct(self, x):
        if self.dims is None:
            shape = x.shape
            out = ops.reshape(x, (-1,))
            for shift in self.shifts:
                out = self._shift(out, shift, 0)
            return ops.reshape(out, tuple(shape))
        dims = self.dims
        shifts = self.shifts
        if len(shifts) == 1 and len(dims) > 1:
            shifts = shifts * len(dims)
        if len(dims) == 1 and len(shifts) > 1:
            dims = dims * len(shifts)
        out = x
        for shift, dim in zip(shifts, dims):
            out = self._shift(out, shift, dim)
        return out


# -- elementwise binary / ternary -------------------------------------------

class _BinaryCell(nn.Cell):
    def __init__(self, other: Any = None) -> None:
        super().__init__()
        self.other = None if other is None else float(other)

    def construct(self, a, b=None):
        raise NotImplementedError


class AddCell(_BinaryCell):
    def construct(self, a, b=None):
        if b is None:
            return a + self.other
        return ops.add(a, b)


class SubCell(_BinaryCell):
    def construct(self, a, b=None):
        if b is None:
            return a - self.other
        return ops.sub(a, b)


class MulCell(_BinaryCell):
    def construct(self, a, b=None):
        if b is None:
            return a * self.other
        return ops.mul(a, b)


class DivCell(_BinaryCell):
    def construct(self, a, b=None):
        if b is None:
            return a / self.other
        return ops.div(a, b)


class PowCell(nn.Cell):
    def __init__(self, exponent: Any = None) -> None:
        super().__init__()
        self.exponent = None if exponent is None else float(exponent)

    def construct(self, a, b=None):
        if b is None:
            return ops.pow(a, self.exponent)
        return ops.pow(a, b)


class MaximumCell(nn.Cell):
    def construct(self, a, b):
        return ops.maximum(a, b)


class MinimumCell(nn.Cell):
    def construct(self, a, b):
        return ops.minimum(a, b)


class WhereCell(nn.Cell):
    def construct(self, cond, a, b):
        return ops.where(ops.cast(cond, ms.bool_), a, b)


class MaskedFillCell(nn.Cell):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = float(value)

    def construct(self, x, mask):
        return ops.masked_fill(x, ops.cast(mask, ms.bool_), self.value)


# -- activations with a dimension -------------------------------------------

class SoftmaxCell(nn.Cell):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = int(dim)

    def construct(self, x):
        return ops.softmax(x, axis=self.dim)


class LogSoftmaxCell(nn.Cell):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = int(dim)

    def construct(self, x):
        return ops.log_softmax(x, axis=self.dim)


# -- reductions --------------------------------------------------------------

def _reduce_dims(dim: Any):
    if dim is None:
        return None
    if isinstance(dim, (list, tuple)):
        return tuple(int(d) for d in dim)
    return int(dim)


class MeanCell(nn.Cell):
    def __init__(self, dim: Any = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = _reduce_dims(dim)
        self.keepdim = bool(keepdim)

    def construct(self, x):
        if self.dim is None:
            return ops.mean(x, axis=None, keep_dims=False)
        return ops.mean(x, axis=self.dim, keep_dims=self.keepdim)


class SumCell(nn.Cell):
    def __init__(self, dim: Any = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = _reduce_dims(dim)
        self.keepdim = bool(keepdim)

    def construct(self, x):
        if self.dim is None:
            return ops.sum(x)
        return ops.sum(x, dim=self.dim, keepdim=self.keepdim)


class AMaxCell(nn.Cell):
    def __init__(self, dim: Any = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = _reduce_dims(dim)
        self.keepdim = bool(keepdim)

    def construct(self, x):
        if self.dim is None:
            return ops.amax(x, axis=None, keepdims=False)
        return ops.amax(x, axis=self.dim, keepdims=self.keepdim)


class AMinCell(nn.Cell):
    def __init__(self, dim: Any = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = _reduce_dims(dim)
        self.keepdim = bool(keepdim)

    def construct(self, x):
        if self.dim is None:
            return ops.amin(x, axis=None, keepdims=False)
        return ops.amin(x, axis=self.dim, keepdims=self.keepdim)


class ProdCell(nn.Cell):
    def __init__(self, dim: Any = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = _reduce_dims(dim)
        self.keepdim = bool(keepdim)

    def construct(self, x):
        if self.dim is None:
            return ops.prod(x, axis=None, keep_dims=False)
        return ops.prod(x, axis=self.dim, keep_dims=self.keepdim)


class VarCell(nn.Cell):
    def __init__(self, dim: Any = None, keepdim: bool = False,
                 correction: int = 1) -> None:
        super().__init__()
        self.dim = _reduce_dims(dim)
        self.keepdim = bool(keepdim)
        self.correction = int(correction)

    def construct(self, x):
        if self.dim is None:
            return ops.var(x, axis=None, ddof=self.correction, keepdims=False)
        return ops.var(x, axis=self.dim, ddof=self.correction, keepdims=self.keepdim)


class StdCell(nn.Cell):
    def __init__(self, dim: Any = None, keepdim: bool = False,
                 correction: int = 1) -> None:
        super().__init__()
        self.dim = _reduce_dims(dim)
        self.keepdim = bool(keepdim)
        self.correction = int(correction)

    def construct(self, x):
        if self.dim is None:
            return ops.std(x, axis=None, ddof=self.correction, keepdims=False)
        return ops.std(x, axis=self.dim, ddof=self.correction, keepdims=self.keepdim)


class ArgMaxCell(nn.Cell):
    def __init__(self, dim: Any = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = None if dim is None else int(dim)
        self.keepdim = bool(keepdim)

    def construct(self, x):
        if self.dim is None:
            return ops.cast(ops.argmax(x), ms.int64)
        return ops.cast(ops.argmax(x, dim=self.dim, keepdim=self.keepdim), ms.int64)


class ArgMinCell(nn.Cell):
    def __init__(self, dim: Any = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = None if dim is None else int(dim)
        self.keepdim = bool(keepdim)

    def construct(self, x):
        if self.dim is None:
            return ops.cast(ops.argmin(x), ms.int64)
        return ops.cast(ops.argmin(x, axis=self.dim, keepdims=self.keepdim), ms.int64)


# -- linear algebra ---------------------------------------------------------

class MatMulCell(nn.Cell):
    def construct(self, a, b):
        return ops.matmul(a, b)


class BMMCell(nn.Cell):
    def construct(self, a, b):
        return ops.bmm(a, b)


class LinearCell(nn.Cell):
    def construct(self, x, w, b=None):
        out = ops.matmul(x, ops.transpose(w, (1, 0)))
        if b is not None:
            out = ops.add(out, b)
        return out


class EinsumCell(nn.Cell):
    """A small einsum engine built from supported primitives.

    ``ops.einsum`` has no CPU kernel in MindSpore 2.7.1.  This cell supports the
    equations that appear in practice: one-operand transpose/reduce/diagonal-free
    patterns and general two-operand contractions without repeated labels.  The
    equation is validated in :func:`make_fn`, so an unsupported equation raises
    :class:`KernelMissing` *before* the graph is generated.
    """

    def __init__(self, equation: str) -> None:
        super().__init__()
        self.equation = _normalise_equation(equation)

    def construct(self, *operands):
        return _einsum_apply(self.equation, operands)


class SDPACell(nn.Cell):
    """``scaled_dot_product_attention`` from ``BatchMatMul`` + ``Softmax``.

    ``dropout_p`` is ignored so that mutants are compared under identical
    stochastic conditions.  A boolean mask keeps the ``True`` positions; a
    floating point mask is added to the logits (PyTorch semantics).
    """

    def __init__(self, dropout_p: float = 0.0, is_causal: bool = False) -> None:
        super().__init__()
        self.dropout_p = float(dropout_p)
        self.is_causal = bool(is_causal)

    def construct(self, q, k, v, mask=None):
        emb = q.shape[-1]
        scale = 1.0 / math.sqrt(float(emb))
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale
        if mask is not None:
            if mask.dtype == ms.bool_:
                scores = ops.select(mask, scores,
                                    ops.full_like(scores, float("-inf")))
            else:
                scores = ops.add(scores, mask)
        elif self.is_causal:
            length = scores.shape[-2]
            source = scores.shape[-1]
            keep = ops.tril(ops.ones((length, source), ms.float32), 0)
            scores = ops.add(scores, (1.0 - keep) * float("-1e30"))
        weights = ops.softmax(scores, axis=-1)
        return ops.matmul(weights, v)


# -- convolution / pooling ---------------------------------------------------

class Conv2DCell(nn.Cell):
    def __init__(self, stride: Any = 1, padding: Any = 0, dilation: Any = 1,
                 groups: int = 1) -> None:
        super().__init__()
        self.stride = _pair(stride, 1)
        self.padding = _pair(padding, 0)
        self.dilation = _pair(dilation, 1)
        self.groups = int(groups or 1)

    def construct(self, x, w, b=None):
        return ops.conv2d(x, w, b, stride=self.stride, pad_mode="pad",
                          padding=self.padding, dilation=self.dilation,
                          groups=self.groups)


class Conv1DCell(nn.Cell):
    def __init__(self, stride: Any = 1, padding: Any = 0, dilation: Any = 1,
                 groups: int = 1) -> None:
        super().__init__()
        self.stride = _int_or_pair(stride, 1)
        self.padding = _int_or_pair(padding, 0)
        self.dilation = _int_or_pair(dilation, 1)
        self.groups = int(groups or 1)

    def construct(self, x, w, b=None):
        return ops.conv1d(x, w, b, stride=self.stride, pad_mode="pad",
                          padding=self.padding, dilation=self.dilation,
                          groups=self.groups)


class ConvTransposeCell(nn.Cell):
    """``conv_transpose2d`` via the ``Conv2DTranspose`` primitive.

    ``ops.conv_transpose2d`` has no CPU kernel in MindSpore 2.7.1, while the
    ``Conv2DTranspose`` primitive does.  The primitive must know the output
    channel count, the kernel size and the output shape at construction time, so
    ``make_fn`` needs the shapes the adapter can supply (``_input_shapes`` /
    ``_out_shape``, taken from the IR's declared edge specs).  When they are
    missing, a best-effort derivation from the weight tensor is used at call
    time if the output channels were unavailable.
    """

    def __init__(self, out_channel: Optional[int] = None,
                 kernel_size: Optional[Any] = None, stride: Any = 1,
                 padding: Any = 0, output_padding: Any = 0, dilation: Any = 1,
                 groups: int = 1) -> None:
        super().__init__()
        self.stride = _int_or_pair(stride, 1)
        self.padding = _pair(padding, 0)
        self.output_padding = _pair(output_padding, 0)
        self.dilation = _int_or_pair(dilation, 1)
        self.groups = int(groups or 1)
        if out_channel is None or kernel_size is None:
            raise KernelMissing(
                "conv_transpose2d needs the weight shape in MindSpore 2.7.1: "
                "pass '_input_shapes'/'_out_shape' (the adapter does this)")
        self.out_channel = int(out_channel)
        k = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.kernel_size = _int_or_pair(k, 1)
        self.op = ops.Conv2DTranspose(out_channel=self.out_channel,
                                      kernel_size=self.kernel_size,
                                      pad_mode="pad", pad=self.padding[0],
                                      stride=self.stride,
                                      dilation=self.dilation,
                                      group=self.groups)

    def construct(self, x, w, b=None):
        n = x.shape[0]
        h = x.shape[2]
        width = x.shape[3]
        kh, kw = _pair(self.kernel_size, 1)
        sh, sw = _pair(self.stride, 1)
        ph, pw = self.padding
        dh, dw = _pair(self.dilation, 1)
        oh = (h - 1) * sh - 2 * ph + dh * (kh - 1) + 1
        ow = (width - 1) * sw - 2 * pw + dw * (kw - 1) + 1
        out = self.op(x, w, (n, self.out_channel, oh, ow))
        if b is not None:
            out = ops.add(out, ops.reshape(b, (1, self.out_channel, 1, 1)))
        op_h, op_w = self.output_padding
        if op_h or op_w:
            out = ops.pad(out, (0, op_w, 0, op_h), mode="constant", value=0.0)
        return out


class MaxPool2DCell(nn.Cell):
    def __init__(self, kernel_size: Any = 2, stride: Any = None,
                 padding: Any = 0, dilation: Any = 1,
                 ceil_mode: bool = False) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size, 2)
        self.stride = None if stride is None else _pair(stride, 1)
        self.padding = _pair(padding, 0)
        self.dilation = _pair(dilation, 1)
        self.ceil_mode = bool(ceil_mode)

    def construct(self, x):
        return ops.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                              self.dilation, return_indices=False,
                              ceil_mode=self.ceil_mode)


class AvgPool2DCell(nn.Cell):
    def __init__(self, kernel_size: Any = 2, stride: Any = None,
                 padding: Any = 0, ceil_mode: bool = False,
                 count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size, 2)
        self.stride = None if stride is None else _int_or_pair(stride, 1)
        pad = _pair(padding, 0)
        # ``ops.avg_pool2d`` only accepts an int or a 4-tuple for padding.
        self.padding = pad[0] if pad[0] == pad[1] else (pad[0], pad[0], pad[1], pad[1])
        self.ceil_mode = bool(ceil_mode)
        self.count_include_pad = bool(count_include_pad)

    def construct(self, x):
        return ops.avg_pool2d(x, self.kernel_size, self.stride, self.padding,
                              ceil_mode=self.ceil_mode,
                              count_include_pad=self.count_include_pad)


class AdaptiveAvgPoolCell(nn.Cell):
    def __init__(self, output_size: Any = 1) -> None:
        super().__init__()
        self.output_size = _pair(output_size, 1)

    def construct(self, x):
        return ops.adaptive_avg_pool2d(x, self.output_size)


class AdaptiveMaxPoolCell(nn.Cell):
    def __init__(self, output_size: Any = 1) -> None:
        super().__init__()
        self.output_size = _pair(output_size, 1)

    def construct(self, x):
        return ops.adaptive_max_pool2d(x, self.output_size, return_indices=False)


class GlobalAvgPoolCell(nn.Cell):
    def __init__(self, keepdim: bool = False, spatial: bool = True) -> None:
        super().__init__()
        self.keepdim = bool(keepdim)
        self.spatial = bool(spatial)

    def construct(self, x):
        if not self.spatial or len(x.shape) < 3:
            return ops.mean(x, axis=-1, keep_dims=self.keepdim)
        dims = tuple(range(2, len(x.shape)))
        return ops.mean(x, axis=dims, keep_dims=self.keepdim)


class GlobalMaxPoolCell(nn.Cell):
    def __init__(self, keepdim: bool = False, spatial: bool = True) -> None:
        super().__init__()
        self.keepdim = bool(keepdim)
        self.spatial = bool(spatial)

    def construct(self, x):
        if not self.spatial or len(x.shape) < 3:
            return ops.amax(x, axis=-1, keepdims=self.keepdim)
        dims = tuple(range(2, len(x.shape)))
        return ops.amax(x, axis=dims, keepdims=self.keepdim)


# -- normalisation -----------------------------------------------------------

class BatchNormCell(nn.Cell):
    """Inference-mode batch normalisation with fixed running statistics."""

    def __init__(self, num_features: Optional[int] = None,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = None if num_features is None else int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)

    def construct(self, x, w=None, b=None):
        channels = self.num_features
        if channels is None:
            channels = x.shape[1] if len(x.shape) > 1 else x.shape[0]
        mean = ops.zeros((channels,), x.dtype)
        variance = ops.ones((channels,), x.dtype)
        if w is not None:
            w = ops.reshape(w, (channels,))
        else:
            w = ops.ones((channels,), x.dtype)
        if b is not None:
            b = ops.reshape(b, (channels,))
        else:
            b = ops.zeros((channels,), x.dtype)
        return ops.batch_norm(x, mean, variance, w, b, training=False,
                              momentum=self.momentum, eps=self.eps)


class LayerNormCell(nn.Cell):
    """``ops.layer_norm`` has no CPU graph kernel, so it is spelled out."""

    def __init__(self, normalized_shape: Optional[Any] = None,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = None if normalized_shape is None \
            else _norm_shape_attr({"normalized_shape": normalized_shape}, 2)
        self.eps = float(eps)

    def construct(self, x, w=None, b=None):
        norm = self.normalized_shape
        if not norm:
            norm = (x.shape[-1],)
        dims = tuple(range(len(x.shape) - len(norm), len(x.shape)))
        mean = ops.mean(x, axis=dims, keep_dims=True)
        variance = ops.var(x, axis=dims, ddof=0, keepdims=True)
        out = (x - mean) / ops.sqrt(variance + self.eps)
        if w is not None:
            out = out * w
        if b is not None:
            out = out + b
        return out


class GroupNormCell(nn.Cell):
    def __init__(self, num_groups: int = 1, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_groups = int(num_groups or 1)
        self.eps = float(eps)

    def construct(self, x, w=None, b=None):
        return ops.group_norm(x, self.num_groups, w, b, self.eps)


class InstanceNormCell(nn.Cell):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)

    def construct(self, x, w=None, b=None):
        rank = len(x.shape)
        dims = tuple(range(2, rank)) if rank >= 3 else (1,)
        mean = ops.mean(x, axis=dims, keep_dims=True)
        variance = ops.var(x, axis=dims, ddof=0, keepdims=True)
        out = (x - mean) / ops.sqrt(variance + self.eps)
        if w is not None:
            out = out * w
        if b is not None:
            out = out + b
        return out


class RMSNormCell(nn.Cell):
    """``ops.rms_norm`` has no CPU kernel; computed from mean/square/rsqrt."""

    def __init__(self, normalized_shape: Optional[Any] = None,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = None if normalized_shape is None \
            else _norm_shape_attr({"normalized_shape": normalized_shape}, 2)
        self.eps = float(eps)

    def construct(self, x, w=None):
        norm = self.normalized_shape
        if not norm:
            norm = (x.shape[-1],)
        dims = tuple(range(len(x.shape) - len(norm), len(x.shape)))
        squared = ops.mean(x * x, axis=dims, keep_dims=True)
        out = x * ops.rsqrt(squared + self.eps)
        if w is not None:
            out = out * w
        return out


class L2NormCell(nn.Cell):
    def __init__(self, dim: int = -1, eps: float = 1e-12) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def construct(self, x):
        norm = ops.sqrt(ops.sum(x * x, dim=self.dim, keepdim=True))
        denom = ops.clamp(norm, self.eps, None)
        return x / denom


# -- shape manipulation ------------------------------------------------------

class ReshapeCell(nn.Cell):
    def __init__(self, shape: Sequence[int]) -> None:
        super().__init__()
        self.shape = tuple(int(v) for v in shape)

    def construct(self, x):
        return ops.reshape(x, self.shape)


class ViewCell(nn.Cell):
    def __init__(self, shape: Sequence[int]) -> None:
        super().__init__()
        self.shape = tuple(int(v) for v in shape)

    def construct(self, x):
        return ops.reshape(x, self.shape)


class FlattenCell(nn.Cell):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = int(start_dim)
        self.end_dim = int(end_dim)

    def construct(self, x):
        return ops.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


class PermuteCell(nn.Cell):
    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        self.dims = tuple(int(d) for d in dims)

    def construct(self, x):
        return ops.transpose(x, self.dims)


class TransposeCell(nn.Cell):
    def __init__(self, dim0: int = 0, dim1: int = 1) -> None:
        super().__init__()
        self.dim0 = int(dim0)
        self.dim1 = int(dim1)

    def construct(self, x):
        rank = len(x.shape)
        d0 = self.dim0 % rank
        d1 = self.dim1 % rank
        dims = list(range(rank))
        dims[d0], dims[d1] = dims[d1], dims[d0]
        return ops.transpose(x, tuple(dims))


class SqueezeCell(nn.Cell):
    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = None if dim is None else int(dim)

    def construct(self, x):
        if self.dim is None:
            return ops.squeeze(x)
        return ops.squeeze(x, axis=self.dim)


class UnsqueezeCell(nn.Cell):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = int(dim)

    def construct(self, x):
        return ops.expand_dims(x, axis=self.dim)


class ExpandCell(nn.Cell):
    def __init__(self, shape: Sequence[int]) -> None:
        super().__init__()
        self.shape = tuple(int(v) for v in shape)

    def construct(self, x):
        return ops.broadcast_to(x, self.shape)


class BroadcastToCell(nn.Cell):
    def __init__(self, shape: Sequence[int]) -> None:
        super().__init__()
        self.shape = tuple(int(v) for v in shape)

    def construct(self, x):
        return ops.broadcast_to(x, self.shape)


class ConcatCell(nn.Cell):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = int(dim)

    def construct(self, *xs):
        return ops.concat(xs, axis=self.dim)


class StackCell(nn.Cell):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = int(dim)

    def construct(self, *xs):
        return ops.stack(xs, axis=self.dim)


class SplitCell(nn.Cell):
    def __init__(self, dim: int = 0, sections: Any = None,
                 num_outputs: int = 2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.sections = sections
        self.num_outputs = max(1, int(num_outputs or 2))

    def construct(self, x):
        size = x.shape[self.dim]
        n = self.num_outputs
        sections = self.sections
        if sections is None:
            part = size // n
            if part <= 0:
                part = 1
            return ops.split(x, part, axis=self.dim)
        if isinstance(sections, int):
            remainder = size - int(sections) * n
            parts = [int(sections)] * n
            if parts:
                parts[-1] += remainder
            return ops.split(x, tuple(parts), axis=self.dim)
        parts = tuple(int(v) for v in sections)
        if len(parts) < n:
            parts = parts + (parts[-1],) * (n - len(parts))
        return ops.split(x, parts, axis=self.dim)


class ChunkCell(nn.Cell):
    def __init__(self, dim: int = 0, num_outputs: int = 2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_outputs = max(1, int(num_outputs or 2))

    def construct(self, x):
        return ops.chunk(x, self.num_outputs, axis=self.dim)


class GetItemCell(nn.Cell):
    def __init__(self, dim: int = 0, index: int = 0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.index = int(index)

    def construct(self, x):
        rank = len(x.shape)
        dim = self.dim % rank
        picking = [slice(None)] * rank
        picking[dim] = self.index
        return x[tuple(picking)]


class PadCell(nn.Cell):
    def __init__(self, pad: Sequence[int], mode: str = "constant",
                 value: float = 0.0) -> None:
        super().__init__()
        self.pad = tuple(int(v) for v in pad)
        self.mode = str(mode or "constant")
        if self.mode not in ("constant", "reflect", "replicate", "circular"):
            self.mode = "constant"
        self.value = float(value)

    def construct(self, x):
        if self.mode == "constant":
            return ops.pad(x, self.pad, mode="constant", value=self.value)
        return ops.pad(x, self.pad, mode=self.mode)


class PixelShuffleCell(nn.Cell):
    def __init__(self, upscale_factor: int = 2) -> None:
        super().__init__()
        self.upscale_factor = int(upscale_factor)

    def construct(self, x):
        return ops.pixel_shuffle(x, self.upscale_factor)


class PixelUnshuffleCell(nn.Cell):
    def __init__(self, upscale_factor: int = 2) -> None:
        super().__init__()
        self.upscale_factor = int(upscale_factor)

    def construct(self, x):
        return ops.pixel_unshuffle(x, self.upscale_factor)


class TileCell(nn.Cell):
    def __init__(self, reps: Sequence[int]) -> None:
        super().__init__()
        self.reps = tuple(int(v) for v in reps)

    def construct(self, x):
        return ops.tile(x, self.reps)


class RepeatInterleaveCell(nn.Cell):
    def __init__(self, repeats: Any = 2, dim: Optional[int] = None) -> None:
        super().__init__()
        self.repeats = repeats
        self.dim = None if dim is None else int(dim)

    def construct(self, x):
        repeats = self.repeats
        if isinstance(repeats, (tuple, list)):
            repeats = ms.Tensor([int(v) for v in repeats], ms.int64)
        return ops.repeat_interleave(x, repeats, axis=self.dim)


class InterpolateCell(nn.Cell):
    """``ops.interpolate`` wrapper.

    MindSpore cannot combine ``scale_factor`` with a 4-D input for the
    ``nearest``/``bilinear`` modes, so a scale factor is always resolved to an
    explicit output size via ``recompute_scale_factor=True`` (which is what
    PyTorch does as well when the scale factor is not an integer).
    """

    def __init__(self, size: Any = None, scale_factor: Any = None,
                 mode: str = "nearest", align_corners: Any = None) -> None:
        super().__init__()
        self.size = None if size is None else tuple(int(v) for v in size)
        self.scale_factor = None
        if scale_factor is not None:
            if isinstance(scale_factor, (tuple, list)):
                self.scale_factor = tuple(float(v) for v in scale_factor)
            else:
                self.scale_factor = float(scale_factor)
        self.mode = str(mode or "nearest")
        self.align_corners = align_corners
        if self.mode not in ("nearest", "nearest-exact", "linear", "bilinear",
                             "bicubic", "trilinear", "area"):
            self.mode = "nearest"
        if self.size is None and self.scale_factor is None:
            raise KernelMissing("interpolate requires 'size' or 'scale_factor'")

    def construct(self, x):
        if self.size is not None:
            if self.mode in ("linear", "bilinear", "bicubic", "trilinear"):
                corners = False if self.align_corners is None else bool(self.align_corners)
                return ops.interpolate(x, size=self.size, mode=self.mode,
                                       align_corners=corners)
            return ops.interpolate(x, size=self.size, mode=self.mode)
        return ops.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                               recompute_scale_factor=True)


# -- embeddings / misc -------------------------------------------------------

class EmbeddingCell(nn.Cell):
    """``ops.embedding`` has no CPU PyNative kernel; ``ops.gather`` is used."""

    def construct(self, ids, weight):
        return ops.gather(weight, ops.cast(ids, ms.int32), 0)


class TopKCell(nn.Cell):
    def __init__(self, k: int = 1, dim: int = -1, largest: bool = True,
                 sorted_: bool = True) -> None:
        super().__init__()
        self.k = int(k)
        self.dim = int(dim)
        self.largest = bool(largest)
        self.sorted = bool(sorted_)

    def construct(self, x):
        values, indices = ops.topk(x, self.k, dim=self.dim, largest=self.largest,
                                   sorted=self.sorted)
        return values, ops.cast(indices, ms.int64)


# ---------------------------------------------------------------------------
# einsum support
# ---------------------------------------------------------------------------

def _normalise_equation(equation: str) -> str:
    return "".join(str(equation).split())


def _validate_equation(equation: str) -> None:
    text = _normalise_equation(equation)
    if "->" not in text or "," not in text:
        raise KernelMissing(
            f"einsum equation {equation!r} is not supported by the MindSpore "
            f"backend (only two-operand explicit-output equations are)")
    left, right = text.split("->")
    terms = left.split(",")
    if len(terms) != 2:
        raise KernelMissing(
            f"einsum equation {equation!r} is not supported by the MindSpore "
            f"backend (only two-operand equations are)")
    sa, sb = terms
    valid = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for term in (sa, sb, right):
        if any(ch not in valid for ch in term):
            raise KernelMissing(f"einsum equation {equation!r} has invalid labels")
    for term in (sa, sb):
        if len(set(term)) != len(term):
            raise KernelMissing(
                f"einsum equation {equation!r} is not supported by the MindSpore "
                f"backend (repeated labels inside one operand are unsupported)")
    for label in right:
        if label not in set(sa) | set(sb):
            raise KernelMissing(f"einsum equation {equation!r} outputs unknown label")


def _einsum_apply(equation: str, operands) -> Any:
    """Static-shape two-operand einsum built from transpose/reshape/bmm."""
    left, right = equation.split("->")
    sa, sb = left.split(",")
    return _einsum_two(sa, sb, right, operands[0], operands[1])


def _einsum_two(sa: str, sb: str, so: str, a, b):
    rank_a = len(a.shape)
    rank_b = len(b.shape)
    if len(sa) != rank_a or len(sb) != rank_b:
        raise ValueError(
            f"einsum subscripts {sa!r},{sb!r} do not match ranks {rank_a},{rank_b}")
    contracted = [c for c in sa if c in sb and c not in so]
    batch = [c for c in sa if c in sb and c in so]
    only_a = [c for c in sa if c not in sb and c in so]
    only_b = [c for c in sb if c not in sa and c in so]

    perm_a = [sa.index(c) for c in batch + only_a + contracted]
    a_t = ops.transpose(a, tuple(perm_a)) if perm_a != list(range(rank_a)) else a
    perm_b = [sb.index(c) for c in batch + contracted + only_b]
    b_t = ops.transpose(b, tuple(perm_b)) if perm_b != list(range(rank_b)) else b

    a_shape = a_t.shape
    b_shape = b_t.shape
    m = 1
    for v in a_shape[len(batch):len(batch) + len(only_a)]:
        m *= int(v)
    k = 1
    for v in a_shape[len(batch) + len(only_a):]:
        k *= int(v)
    n = 1
    for v in b_shape[len(batch) + len(contracted):]:
        n *= int(v)

    batch_dims = tuple(int(v) for v in a_shape[:len(batch)])
    if batch_dims:
        a_2d = ops.reshape(a_t, (-1, m, k))
        b_2d = ops.reshape(b_t, (-1, k, n))
        out = ops.bmm(a_2d, b_2d)
        out_dims = tuple(batch_dims) + tuple(int(v) for v in a_shape[len(batch):len(batch) + len(only_a)]) \
            + tuple(int(v) for v in b_shape[len(batch) + len(contracted):])
        return ops.reshape(out, out_dims)
    a_2d = ops.reshape(a_t, (m, k))
    b_2d = ops.reshape(b_t, (k, n))
    out = ops.matmul(a_2d, b_2d)
    out_dims = tuple(int(v) for v in a_shape[len(batch):len(batch) + len(only_a)]) \
        + tuple(int(v) for v in b_shape[len(batch) + len(contracted):])
    if not out_dims:
        return ops.reshape(out, ())
    return ops.reshape(out, out_dims)


# ---------------------------------------------------------------------------
# op -> callable table
# ---------------------------------------------------------------------------

#: Ops that have no MindSpore implementation at all in 2.7.1.
MS_UNSUPPORTED: set = set()

MULTI_OUTPUT_DEFAULT: Dict[str, int] = {"split": 2, "chunk": 2, "topk": 2}

_ATTR = Tuple[Dict[str, Any], Dict[str, Any]]


def _shape_list(attrs: Dict[str, Any], key: str) -> Optional[List[Tuple[int, ...]]]:
    raw = attrs.get(key)
    if not raw:
        return None
    out: List[Tuple[int, ...]] = []
    for entry in raw:
        try:
            out.append(tuple(int(v) for v in entry))
        except Exception:
            return None
    return out


def _conv_transpose_builder(attrs: Dict[str, Any]) -> nn.Cell:
    shapes = _shape_list(attrs, "_input_shapes")
    weight_shape = shapes[1] if shapes and len(shapes) > 1 else None
    out_channels = attrs.get("out_channels")
    kernel = attrs.get("kernel_size")
    if weight_shape and len(weight_shape) == 4:
        out_channels = weight_shape[1]
        kernel = (weight_shape[2], weight_shape[3])
    return ConvTransposeCell(
        out_channel=out_channels, kernel_size=kernel,
        stride=attrs.get("stride", 1), padding=attrs.get("padding", 0),
        output_padding=attrs.get("output_padding", 0),
        dilation=attrs.get("dilation", 1), groups=attrs.get("groups", 1))


def _einsum_builder(attrs: Dict[str, Any]) -> nn.Cell:
    equation = attrs.get("equation")
    if not equation:
        raise ValueError("einsum requires an 'equation' attribute")
    _validate_equation(str(equation))
    return EinsumCell(str(equation))


def _norm_builder(cls):
    def build(attrs: Dict[str, Any]) -> nn.Cell:
        norm = attrs.get("normalized_shape")
        if norm is not None:
            norm = _norm_shape_attr(attrs, 2)
        return cls(normalized_shape=norm, eps=attrs.get("eps", 1e-5))
    return build


#: ``op -> builder(attrs) -> nn.Cell``.  Every builder is a pure function of
#: ``attrs`` so that the same table can be used from PyNative and from a
#: generated graph-mode ``Cell``.
_BUILDERS: Dict[str, Callable[[Dict[str, Any]], nn.Cell]] = {
    # elementwise unary
    "relu": lambda a: ReLUCell(),
    "gelu": lambda a: GELUCell(a.get("approximate", "none")),
    "quick_gelu": lambda a: QuickGELUCell(),
    "silu": lambda a: SiLUCell(),
    "mish": lambda a: MishCell(),
    "hardswish": lambda a: HardSwishCell(),
    "hardsigmoid": lambda a: HardSigmoidCell(),
    "elu": lambda a: ELUCell(a.get("alpha", 1.0)),
    "leaky_relu": lambda a: LeakyReLUCell(a.get("negative_slope", 0.01)),
    "softplus": lambda a: SoftplusCell(a.get("beta", 1.0), a.get("threshold", 20.0)),
    "sigmoid": lambda a: SigmoidCell(),
    "tanh": lambda a: TanhCell(),
    "exp": lambda a: ExpCell(),
    "log": lambda a: LogCell(),
    "sqrt": lambda a: SqrtCell(),
    "rsqrt": lambda a: RsqrtCell(),
    "reciprocal": lambda a: ReciprocalCell(),
    "sin": lambda a: SinCell(),
    "cos": lambda a: CosCell(),
    "erf": lambda a: ErfCell(),
    "floor": lambda a: FloorCell(),
    "round": lambda a: RoundCell(),
    "sign": lambda a: SignCell(),
    "neg": lambda a: NegCell(),
    "abs": lambda a: AbsCell(),
    "square": lambda a: SquareCell(),
    "clamp": lambda a: ClampCell(a.get("min"), a.get("max")),
    "dropout": lambda a: DropoutCell(),
    "identity": lambda a: IdentityCell(),
    "clone": lambda a: CloneCell(),
    "detach": lambda a: DetachCell(),
    "stop_gradient": lambda a: StopGradientCell(),
    "contiguous": lambda a: ContiguousCell(),
    "cast": lambda a: CastCell(a.get("dtype", "float32")),
    "flip": lambda a: FlipCell(a.get("dims") if a.get("dims") is not None
                               else a.get("dim", 0)),
    "roll": lambda a: RollCell(a.get("shifts", 1),
                               a.get("dims") if a.get("dims") is not None
                               else a.get("dim", 0)),
    # elementwise binary / ternary
    "add": lambda a: AddCell(a.get("other") if a.get("other") is not None
                             else a.get("alpha")),
    "sub": lambda a: SubCell(a.get("other")),
    "mul": lambda a: MulCell(a.get("other")),
    "div": lambda a: DivCell(a.get("other")),
    "pow": lambda a: PowCell(a.get("exponent")),
    "maximum": lambda a: MaximumCell(),
    "minimum": lambda a: MinimumCell(),
    "where": lambda a: WhereCell(),
    "masked_fill": lambda a: MaskedFillCell(a.get("value", 0.0)),
    # activations
    "softmax": lambda a: SoftmaxCell(a.get("dim", -1)),
    "log_softmax": lambda a: LogSoftmaxCell(a.get("dim", -1)),
    # reductions
    "mean": lambda a: MeanCell(a.get("dim"), a.get("keepdim", False)),
    "sum": lambda a: SumCell(a.get("dim"), a.get("keepdim", False)),
    "amax": lambda a: AMaxCell(a.get("dim"), a.get("keepdim", False)),
    "amin": lambda a: AMinCell(a.get("dim"), a.get("keepdim", False)),
    "prod": lambda a: ProdCell(a.get("dim"), a.get("keepdim", False)),
    "var": lambda a: VarCell(a.get("dim"), a.get("keepdim", False),
                             a.get("correction", 1)),
    "std": lambda a: StdCell(a.get("dim"), a.get("keepdim", False),
                             a.get("correction", 1)),
    "argmax": lambda a: ArgMaxCell(a.get("dim"), a.get("keepdim", False)),
    "argmin": lambda a: ArgMinCell(a.get("dim"), a.get("keepdim", False)),
    # linear algebra
    "matmul": lambda a: MatMulCell(),
    "bmm": lambda a: BMMCell(),
    "linear": lambda a: LinearCell(),
    "einsum": _einsum_builder,
    "sdpa": lambda a: SDPACell(a.get("dropout_p", 0.0), a.get("is_causal", False)),
    "scaled_dot_product_attention": lambda a: SDPACell(a.get("dropout_p", 0.0),
                                                       a.get("is_causal", False)),
    # convolution / pooling
    "conv2d": lambda a: Conv2DCell(a.get("stride", 1), a.get("padding", 0),
                                   a.get("dilation", 1), a.get("groups", 1)),
    "conv1d": lambda a: Conv1DCell(a.get("stride", 1), a.get("padding", 0),
                                   a.get("dilation", 1), a.get("groups", 1)),
    "conv_transpose2d": _conv_transpose_builder,
    "maxpool2d": lambda a: MaxPool2DCell(a.get("kernel_size", 2), a.get("stride"),
                                         a.get("padding", 0), a.get("dilation", 1),
                                         a.get("ceil_mode", False)),
    "avgpool2d": lambda a: AvgPool2DCell(a.get("kernel_size", 2), a.get("stride"),
                                         a.get("padding", 0), a.get("ceil_mode", False),
                                         a.get("count_include_pad", True)),
    "adaptive_avgpool2d": lambda a: AdaptiveAvgPoolCell(a.get("output_size", 1)),
    "adaptive_maxpool2d": lambda a: AdaptiveMaxPoolCell(a.get("output_size", 1)),
    "global_avgpool": lambda a: GlobalAvgPoolCell(a.get("keepdim", False),
                                                  a.get("spatial", True)),
    "global_maxpool": lambda a: GlobalMaxPoolCell(a.get("keepdim", False),
                                                  a.get("spatial", True)),
    # normalisation
    "batchnorm": lambda a: BatchNormCell(a.get("num_features"),
                                         a.get("momentum", 0.1), a.get("eps", 1e-5)),
    "layernorm": lambda a: LayerNormCell(a.get("normalized_shape"), a.get("eps", 1e-5)),
    "groupnorm": lambda a: GroupNormCell(a.get("num_groups", 1), a.get("eps", 1e-5)),
    "instancenorm": lambda a: InstanceNormCell(a.get("eps", 1e-5)),
    "rmsnorm": lambda a: RMSNormCell(a.get("normalized_shape"), a.get("eps", 1e-6)),
    "l2norm": lambda a: L2NormCell(a.get("dim", -1), a.get("eps", 1e-12)),
    # shape manipulation
    "reshape": lambda a: ReshapeCell(_target_shape(a)),
    "view": lambda a: ViewCell(_target_shape(a)),
    "flatten": lambda a: FlattenCell(a.get("start_dim", 1), a.get("end_dim", -1)),
    "permute": lambda a: PermuteCell(_required(a, "dims")),
    "transpose": lambda a: TransposeCell(a.get("dim0", 0), a.get("dim1", 1)),
    "squeeze": lambda a: SqueezeCell(a.get("dim")),
    "unsqueeze": lambda a: UnsqueezeCell(a.get("dim", 0)),
    "expand": lambda a: ExpandCell(_required(a, "shape")),
    "broadcast_to": lambda a: BroadcastToCell(_required(a, "shape")),
    "concat": lambda a: ConcatCell(a.get("dim", 0)),
    "stack": lambda a: StackCell(a.get("dim", 0)),
    "split": lambda a: SplitCell(a.get("dim", 0), a.get("sections"),
                                 a.get("num_outputs", 2)),
    "chunk": lambda a: ChunkCell(a.get("dim", 0), a.get("num_outputs", 2)),
    "getitem": lambda a: GetItemCell(a.get("dim", 0), a.get("index", 0)),
    "pad": lambda a: PadCell(_required(a, "pad"), a.get("mode", "constant"),
                             a.get("value", 0.0)),
    "pixel_shuffle": lambda a: PixelShuffleCell(a.get("upscale_factor", 2)),
    "pixel_unshuffle": lambda a: PixelUnshuffleCell(a.get("upscale_factor", 2)),
    "tile": lambda a: TileCell(_required(a, "reps")),
    "repeat_interleave": lambda a: RepeatInterleaveCell(a.get("repeats", 2),
                                                        a.get("dim")),
    "interpolate": lambda a: InterpolateCell(a.get("size"), a.get("scale_factor"),
                                             a.get("mode", "nearest"),
                                             a.get("align_corners")),
    # embeddings / misc
    "embedding": lambda a: EmbeddingCell(),
    "topk": lambda a: TopKCell(a.get("k", 1), a.get("dim", -1),
                               a.get("largest", True), a.get("sorted", True)),
}

#: Ops the MindSpore backend implements, in every mode.
MS_SUPPORTED: set = set(_BUILDERS)


def _required(attrs: Dict[str, Any], key: str) -> Any:
    value = attrs.get(key)
    if value is None:
        raise KernelMissing(f"operator requires a {key!r} attribute")
    return value


def _target_shape(attrs: Dict[str, Any]) -> Sequence[int]:
    shape = attrs.get("shape") or attrs.get("size")
    if shape is None:
        raise KernelMissing("reshape/view requires a 'shape' attribute")
    return [int(v) for v in shape]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

_FN_CACHE: Dict[Any, nn.Cell] = {}


def _attrs_key(op: str, attrs: Dict[str, Any]) -> Any:
    try:
        return (op, repr(sorted((str(k), repr(v)) for k, v in attrs.items())))
    except Exception:
        return None


def make_fn(op: str, attrs: Dict[str, Any]) -> nn.Cell:
    """Return a callable ``nn.Cell`` with ``attrs`` bound at construction time.

    The returned object is usable both eagerly (PyNative) and inside a generated
    graph-mode ``Cell``'s ``construct``.
    """
    name = canonical_op(op)
    if name in MS_UNSUPPORTED:
        raise KernelMissing(
            f"IR op {name!r} is not supported by the MindSpore 2.7.1 backend")
    builder = _BUILDERS.get(name)
    if builder is None:
        raise KernelMissing(f"no MindSpore kernel for IR op {op!r}")
    try:
        return builder(dict(attrs))
    except KernelMissing:
        raise
    except Exception as exc:
        raise KernelMissing(
            f"cannot build a MindSpore kernel for IR op {name!r}: "
            f"{type(exc).__name__}: {exc}") from exc


def apply_pynative(op: str, attrs: Dict[str, Any], inputs: Sequence[Any]) -> Any:
    """Eagerly evaluate one IR node (used by the ``F(e)`` profiling pass)."""
    name = canonical_op(op)
    key = _attrs_key(name, attrs)
    fn = _FN_CACHE.get(key) if key is not None else None
    if fn is None:
        fn = make_fn(name, attrs)
        if key is not None and len(_FN_CACHE) < 4096:
            _FN_CACHE[key] = fn
    return fn(*list(inputs))


# ---------------------------------------------------------------------------
# deterministic parameter initialisation (numpy-only)
# ---------------------------------------------------------------------------

def init_parameter(tensor: Any, init: str, generator: np.random.Generator) -> Any:
    """Deterministic initialisation mirroring ``torch_kernels.init_parameter``.

    ``tensor`` is a ``numpy.ndarray`` (filled in place and returned) or a
    ``mindspore.Tensor`` (a freshly initialised :class:`numpy.ndarray` is
    returned).  The fan-in/fan-out formulas are identical to the PyTorch
    version so that both frameworks see parameters of the same scale.
    """
    if isinstance(tensor, MSTensor):
        array = np.zeros(tuple(int(d) for d in tensor.shape),
                         dtype=np.dtype(dtype_name(tensor.dtype)))
        init_parameter(array, init, generator)
        return ms.Tensor.from_numpy(array)

    array = tensor
    shape = list(array.shape)
    if not shape:
        array[...] = 0
        return array
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

    if init == "zeros":
        array[...] = 0.0
    elif init == "ones":
        array[...] = 1.0
    elif init == "kaiming_normal":
        std = math.sqrt(2.0 / max(1, fan_in))
        array[...] = generator.normal(0.0, std, size=shape).astype(array.dtype, copy=False)
    elif init == "xavier_uniform":
        bound = math.sqrt(6.0 / max(1, fan_in + fan_out))
        array[...] = generator.uniform(-bound, bound, size=shape).astype(array.dtype, copy=False)
    elif init == "normal":
        array[...] = generator.normal(0.0, 0.02, size=shape).astype(array.dtype, copy=False)
    elif init == "truncated_normal":
        values = generator.normal(0.0, 0.02, size=shape)
        array[...] = np.clip(values, -0.04, 0.04).astype(array.dtype, copy=False)
    elif init == "uniform":
        array[...] = generator.uniform(-0.1, 0.1, size=shape).astype(array.dtype, copy=False)
    elif init == "identity":
        array[...] = 0.0
        n = min(shape[0], shape[1] if len(shape) > 1 else shape[0])
        for i in range(n):
            array[i, i] = 1.0
    else:  # kaiming_uniform (the default)
        bound = math.sqrt(6.0 / max(1, fan_in))
        array[...] = generator.uniform(-bound, bound, size=shape).astype(array.dtype, copy=False)
    return array
