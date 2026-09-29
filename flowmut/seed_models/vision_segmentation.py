"""Semantic-segmentation seed models.

* **SegFormer-B1** -- hierarchical MiT encoder: overlap patch embeddings, then
  transformer blocks whose self-attention uses *spatial reduction*
  (``reshape``/``linear`` down to a shorter sequence, ``matmul``/``softmax``
  attention, ``linear`` back) followed by a Mix-FFN (depthwise 3x3 conv -> GELU
  -> linear -> linear).  A lightweight all-MLP decoder projects every scale to a
  common width, upsamples to 1/4 resolution, concatenates and predicts.
* **Mask2Former-Swin-T** -- a simplified Swin stage stack (shifted windows
  partitioned with ``reshape``/``permute`` and attention built from
  ``matmul``/``softmax``), a pixel decoder that fuses multi-scale features with
  ``interpolate`` + ``concat`` + conv, and a transformer decoder with masked
  cross-attention plus a per-query mask head.

Both default to a compact scale (documented hyper-parameters are kept under
``"full"``) and import no DL framework.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from flowmut.ir.graph import Graph, GraphBuilder
from flowmut.seed_models.registry import SeedModel, register_seed


def _shape_of(edge: str, b: GraphBuilder) -> Tuple[int, int, int, int]:
    s = b.graph.edges[edge].spec.shape
    return int(s[0]), int(s[1]), int(s[2]), int(s[3])



def _as_symbolic_batch(b: GraphBuilder, edge: str) -> str:
    """Re-emit ``edge`` with a symbolic batch dimension (``-1``).

    ``concat`` refuses operands whose batch dimensions disagree, so features that
    still carry a concrete batch are rebased onto ``-1`` before multi-scale
    fusion.
    """
    spec = b.graph.edges[edge].spec
    if spec is None or not spec.shape or int(spec.shape[0]) < 0:
        return edge
    shape = (-1,) + tuple(int(v) for v in spec.shape[1:])
    node = b.graph.producer_of(edge)
    # ``identity`` keeps the tensors bit-identical while letting the declared
    # spec carry the symbolic batch dimension.
    return b.identity(edge, _shape=shape, scope=f"{node.scope}.poly_batch")


def _symbolic_batch_graph(b: GraphBuilder) -> Graph:
    """Finalise ``b`` with a symbolic (``-1``) batch dimension.

    The driver profiles every seed with more than one batch size, so the batch
    axis is declared as ``-1`` throughout: every shape is inferred with ``-1`` as
    the leading dimension and the kernels then use the real batch size at
    execution time.  Only the image batch axis becomes symbolic -- the spatial
    axes are concrete, and no 4-D model here has a spatial or channel size that
    could be confused with it.
    """
    original = GraphBuilder.input

    def symbolic_input(self, name, shape, *args, **kwargs):
        dims = tuple(shape)
        if dims:
            dims = (-1,) + tuple(dims[1:])
        return original(self, name, dims, *args, **kwargs)

    GraphBuilder.input = symbolic_input
    try:
        graph = b.build()
    finally:
        GraphBuilder.input = original
    return graph


def _cba(b: GraphBuilder, x: str, out_channels: int, kernel_size: int = 3,
         stride: int = 1, groups: int = 1, scope: str = "conv") -> str:
    """Conv + BatchNorm + SiLU composite used by the segmentation decoders."""
    tag = f"k{kernel_size}_s{stride}_g{groups}"
    return b.conv_block(x, out_channels, kernel_size=kernel_size, stride=stride,
                        groups=groups, act="silu", norm="batchnorm",
                        scope=f"{scope}.{tag}")

def _conv(b: GraphBuilder, x: str, out_channels: int, kernel_size: int, scope: str,
          stride: int = 1, padding: Optional[int] = None, groups: int = 1,
          act: str = "", norm: str = "", bias: bool = False) -> str:
    """``conv_block`` wrapper that makes the kernel geometry part of the scope.

    The framework adapters key parameters by ``(op, attrs, scope)``; without the
    kernel/stride tag two differently-shaped layers that happen to share a scope
    would collide on one parameter tensor.
    """
    if padding is None:
        # same-style only for stride 1; a strided layer pads 0 (a k=s downsample
        # must really halve the grid) unless the caller says otherwise.
        padding = kernel_size // 2 if stride == 1 else 0
    tag = f"k{kernel_size}_s{stride}_g{groups}_p{padding}"
    return b.conv_block(x, out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, groups=groups, act=act, norm=norm,
                        bias=bias, scope=f"{scope}.{tag}")



# ---------------------------------------------------------------------------
# SegFormer-B1
# ---------------------------------------------------------------------------
# Documented (SegFormer-B1, Xie et al. 2021): MiT-B1 encoder with depths
# (2, 2, 2, 2), dims (64, 128, 320, 512), heads (1, 2, 5, 8), SR ratios
# (8, 4, 2, 1), mlp_ratio 4, overlap patch embeddings k=7 s=4 (stage 0) then
# k=3 s=2; all-MLP decoder with embedding dim 256 and a 150-class ADE20K head.
# Documented benchmark: 512x512 float32 NCHW, 13,715,798 parameters.
SEGFORMER_B1_CONFIGS: Dict[str, Dict[str, Any]] = {
    "compact": dict(
        depths=(1, 1, 1, 1),
        dims=(16, 32, 64, 128),
        heads=(1, 2, 4, 8),
        sr_ratios=(4, 2, 2, 1),
        patch_sizes=(7, 3, 3, 3),
        patch_strides=(4, 2, 2, 2),
        mlp_ratio=4.0,
        decoder_dim=64,
        num_classes=150,
    ),
    "full": dict(
        depths=(2, 2, 2, 2),
        dims=(64, 128, 320, 512),
        heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        patch_sizes=(7, 3, 3, 3),
        patch_strides=(4, 2, 2, 2),
        mlp_ratio=4.0,
        decoder_dim=256,
        num_classes=150,
    ),
}
SEGFORMER_B1 = SEGFORMER_B1_CONFIGS["compact"]


def _overlap_patch_embed(b: GraphBuilder, x: str, dim: int, kernel: int, stride: int,
                         scope: str) -> Tuple[str, Tuple[int, int]]:
    """Overlapping patch embedding: conv -> flatten to (n, seq, c) -> LayerNorm.

    Accepts either an NCHW image/grid or a ``(n, seq, c)`` token sequence, and
    returns the edge *and* the 2-D grid size it was flattened from so callers
    can rebuild the spatial layout for Mix-FFN / attention bookkeeping.
    """
    shape = b.graph.edges[x].spec.shape
    if len(shape) == 3:
        n, seq, c = (int(v) for v in shape)
        side = int(round(seq ** 0.5))
        x = b.permute(x, dims=(0, 2, 1), scope=f"{scope}.to_map")
        x = b.reshape(x, shape=(-1, c, side, side), scope=f"{scope}.grid")
    with b.block(scope):
        y = _conv(b, x, dim, kernel_size=kernel, stride=stride,
                  padding=kernel // 2, act="", norm="", scope="proj")
        nn, cc, hh, ww = _shape_of(y, b)
        y = b.reshape(y, shape=(-1, cc, hh * ww), scope="flatten")
        y = b.permute(y, dims=(0, 2, 1), scope="to_seq")
        return b.layernorm(y, normalized_shape=(dim,), eps=1e-6, scope="norm"), (hh, ww)


def _sr_attention(b: GraphBuilder, x: str, dim: int, num_heads: int, sr_ratio: int,
                  out_hw: Tuple[int, int], scope: str) -> str:
    """Efficient self-attention with spatial reduction (SegFormer Eq. 2)."""
    n, seq, c = (int(v) for v in b.graph.edges[x].spec.shape)
    hh, ww = out_hw
    hd = dim // num_heads
    red_h, red_w = max(1, hh // sr_ratio), max(1, ww // sr_ratio)
    red_seq = red_h * red_w
    with b.block(scope):
        q = b.linear(x, dim, scope="q")
        # -- spatial reduction: fold the 2-D grid back in and convolve -----
        kv = b.permute(x, dims=(0, 2, 1), scope="kv_to_map")
        kv = b.reshape(kv, shape=(-1, c, hh, ww), scope="kv_grid")
        kv = _conv(b, kv, c, kernel_size=sr_ratio, stride=sr_ratio, act="",
                   norm="", scope="sr_conv")
        kv = b.reshape(kv, shape=(-1, c, red_seq), scope="kv_flat")
        kv = b.permute(kv, dims=(0, 2, 1), scope="kv_seq")
        kv = b.layernorm(kv, normalized_shape=(c,), eps=1e-6, scope="kv_norm")
        k = b.linear(kv, dim, scope="k")
        v = b.linear(kv, dim, scope="v")

        q = b.reshape(q, shape=(-1, seq, num_heads, hd), scope="q_heads")
        k = b.reshape(k, shape=(-1, red_seq, num_heads, hd), scope="k_heads")
        v = b.reshape(v, shape=(-1, red_seq, num_heads, hd), scope="v_heads")
        q = b.permute(q, dims=(0, 2, 1, 3), scope="q_t")
        k = b.permute(k, dims=(0, 2, 3, 1), scope="k_t")
        v = b.permute(v, dims=(0, 2, 1, 3), scope="v_t")
        logits = b.matmul(q, k, scope="logits")
        logits = b.mul(logits, other=hd ** -0.5, scope="scale")
        probs = b.softmax(logits, dim=-1, scope="probs")
        ctx = b.matmul(probs, v, scope="ctx")
        ctx = b.permute(ctx, dims=(0, 2, 1, 3), scope="merge_heads")
        ctx = b.reshape(ctx, shape=(-1, seq, dim), scope="ctx_flat")
        return b.linear(ctx, dim, scope="out_proj")


def _mix_ffn(b: GraphBuilder, x: str, dim: int, mlp_ratio: float,
             out_hw: Tuple[int, int], scope: str) -> str:
    """Mix-FFN: depthwise 3x3 conv -> GELU -> linear -> linear."""
    n, seq, c = (int(v) for v in b.graph.edges[x].spec.shape)
    hh, ww = out_hw
    hidden = max(1, int(round(dim * mlp_ratio)))
    with b.block(scope):
        y = b.permute(x, dims=(0, 2, 1), scope="to_map")
        y = b.reshape(y, shape=(-1, c, hh, ww), scope="grid")
        y = _conv(b, y, dim, kernel_size=3, padding=1, groups=dim, act="gelu",
                         norm="", scope="dwconv")
        y = b.reshape(y, shape=(-1, dim, seq), scope="flat")
        y = b.permute(y, dims=(0, 2, 1), scope="to_seq")
        y = b.linear(y, hidden, act="gelu", scope="fc1")
        return b.linear(y, dim, scope="fc2")


def _mit_block(b: GraphBuilder, x: str, dim: int, num_heads: int, sr_ratio: int,
               mlp_ratio: float, out_hw: Tuple[int, int], scope: str) -> str:
    """One MiT transformer block (pre-norm attention + pre-norm Mix-FFN)."""
    with b.block(scope):
        h = b.layernorm(x, normalized_shape=(dim,), eps=1e-6, scope="attn_norm")
        h = _sr_attention(b, h, dim, num_heads, sr_ratio, out_hw, "attn")
        x = b.add(x, h, scope="attn_residual")
        y = b.layernorm(x, normalized_shape=(dim,), eps=1e-6, scope="ffn_norm")
        y = _mix_ffn(b, y, dim, mlp_ratio, out_hw, "ffn")
        return b.add(x, y, scope="ffn_residual")


def build_segformer_b1(cfg: Dict[str, Any] = SEGFORMER_B1) -> Any:
    """Build the (compact by default) SegFormer-B1 graph."""
    b = GraphBuilder("segformer_b1")
    depths = tuple(cfg["depths"])
    dims = tuple(cfg["dims"])
    heads = tuple(cfg["heads"])
    sr = tuple(cfg["sr_ratios"])
    ks = tuple(cfg["patch_sizes"])
    strides = tuple(cfg["patch_strides"])
    x = b.input("images", (1, 3, 512, 512))

    feats: List[str] = []
    h = x
    for si in range(len(dims)):
        h, (side_h, side_w) = _overlap_patch_embed(
            b, h, dims[si], int(ks[si]), int(strides[si]),
            f"encoder.stage.{si}.patch_embed")
        for bi in range(int(depths[si])):
            h = _mit_block(b, h, dims[si], int(heads[si]), int(sr[si]),
                           float(cfg["mlp_ratio"]), (side_h, side_w),
                           f"encoder.stage.{si}.block.{bi}")
        feats.append(h)

    # -- all-MLP decoder --------------------------------------------------
    dec = int(cfg["decoder_dim"])
    with b.block("decoder"):
        proj = [b.linear(f, dec, scope=f"linear_c.{i}") for i, f in enumerate(feats)]
        maps = []
        for i, p in enumerate(proj):
            n, seq, c = (int(v) for v in b.graph.edges[p].spec.shape)
            hw = int(round(seq ** 0.5))
            m = b.permute(p, dims=(0, 2, 1), scope=f"to_map.{i}")
            maps.append(b.reshape(m, shape=(-1, c, hw, hw), scope=f"grid.{i}"))
        size = (x_size(b) // 4, x_size(b) // 4)
        maps = [b.interpolate(m, size=size, mode="bilinear", align_corners=False,
                              scope=f"upsample.{i}") for i, m in enumerate(maps)]
        fused = b.concat(*maps, dim=1, scope="concat")
        fused = _conv(b, fused, dec, kernel_size=1, act="gelu", norm="",
                      scope="fuse")
        logits = _conv(b, fused, int(cfg["num_classes"]), kernel_size=1, act="",
                       norm="", scope="classifier")
        logits = b.interpolate(logits, size=(x_size(b), x_size(b)), mode="bilinear",
                               align_corners=False, scope="to_input")
    b.output([logits], names=["seg_logits"])
    return _symbolic_batch_graph(b)


def x_size(b: GraphBuilder) -> int:
    """Spatial size of the graph's first input (square inputs only)."""
    s = b.graph.edges[b.graph.inputs[0]].spec.shape
    return int(s[-1])


# ---------------------------------------------------------------------------
# Mask2Former-Swin-T
# ---------------------------------------------------------------------------
# Documented (Mask2Former-Swin-T, Cheng et al. 2022): Swin-Tiny backbone
# (depths (2, 2, 6, 2), dims (96, 192, 384, 768), 4x4 patch embedding), MSDeformAttn
# pixel decoder, 9 transformer decoder layers with masked cross-attention and
# 100 object queries over 150 ADE20K classes.
# Documented benchmark: 512x512 float32 NCHW, 47,441,169 parameters.
MASK2FORMER_SWIN_T_CONFIGS: Dict[str, Dict[str, Any]] = {
    "compact": dict(
        depths=(1, 1, 2),
        dims=(24, 48, 96),
        heads=(3, 6, 12),
        window=4,
        patch_stride=4,
        hidden_dim=32,
        num_heads=4,
        num_queries=16,
        num_layers=2,
        num_classes=150,
        mask_dim=32,
        decoder_levels=2,
    ),
    "full": dict(
        depths=(2, 2, 6, 2),
        dims=(96, 192, 384, 768),
        heads=(3, 6, 12, 24),
        window=7,
        patch_stride=4,
        hidden_dim=256,
        num_heads=8,
        num_queries=100,
        num_layers=9,
        num_classes=150,
        mask_dim=256,
        decoder_levels=4,
    ),
}
MASK2FORMER_SWIN_T = MASK2FORMER_SWIN_T_CONFIGS["compact"]


def _cyclic_shift(b: GraphBuilder, x: str, shift: int, scope: str) -> str:
    """Cyclic ``(-shift, -shift)`` roll of an NCHW tensor (shifted windows)."""
    if shift == 0:
        return x
    return b.roll(x, shifts=(-shift, -shift), dims=(2, 3), scope=scope)


def _window_partition(b: GraphBuilder, x: str, win: int, scope: str) -> str:
    """``(n, c, h, w) -> (n*nw_h*nw_w, win*win, c)`` via reshape/permute."""
    n, c, h, w = _shape_of(x, b)
    nw_h, nw_w = h // win, w // win
    y = b.reshape(x, shape=(-1, c, nw_h, win, nw_w, win), scope=f"{scope}.grid")
    y = b.permute(y, dims=(0, 2, 4, 3, 5, 1), scope=f"{scope}.windows")
    return b.reshape(y, shape=(-1, win * win, c), scope=f"{scope}.tokens")


def _window_reverse(b: GraphBuilder, x: str, win: int,
                    shape: Tuple[int, int, int, int], scope: str) -> str:
    """``(n*nw_h*nw_w, win*win, c) -> (n, c, h, w)``; inverse of partition."""
    n, c, h, w = shape
    nw_h, nw_w = h // win, w // win
    y = b.reshape(x, shape=(-1, nw_h, nw_w, win, win, c), scope=f"{scope}.windows")
    y = b.permute(y, dims=(0, 5, 1, 3, 2, 4), scope=f"{scope}.grid")
    return b.reshape(y, shape=(-1, c, h, w), scope=f"{scope}.merge")


def _shifted_window_attention(b: GraphBuilder, x: str, dim: int, num_heads: int,
                              win: int, shift: int, scope: str) -> str:
    """Shifted-window self-attention explicitly built from matmul/softmax."""
    n, c, h, w = _shape_of(x, b)
    y = _cyclic_shift(b, x, shift, f"{scope}.shift")
    ph, pw = h + (-h) % win, w + (-w) % win
    if ph != h or pw != w:
        y = b.pad(y, pad=(0, pw - w, 0, ph - h), mode="constant", value=0.0,
                  _shape=(-1, c, ph, pw), scope=f"{scope}.pad")
    hd = dim // num_heads
    seq = win * win
    with b.block(scope):
        y = _window_partition(b, y, win, "partition")
        qkv = b.linear(y, 3 * dim, scope="qkv")
        qkv = b.reshape(qkv, shape=(-1, seq, 3, dim), scope="unfuse_qkv")
        q, k, v = b.split(qkv, dim=2, num_outputs=3, _num_outputs=3, scope="qkv_parts")
        q = b.reshape(q, shape=(-1, seq, num_heads, hd), scope="q_heads")
        k = b.reshape(k, shape=(-1, seq, num_heads, hd), scope="k_heads")
        v = b.reshape(v, shape=(-1, seq, num_heads, hd), scope="v_heads")
        q = b.permute(q, dims=(0, 2, 1, 3), scope="q_t")
        k = b.permute(k, dims=(0, 2, 3, 1), scope="k_t")
        v = b.permute(v, dims=(0, 2, 1, 3), scope="v_t")
        logits = b.matmul(q, k, scope="logits")
        logits = b.mul(logits, other=hd ** -0.5, scope="scale")
        probs = b.softmax(logits, dim=-1, scope="probs")
        ctx = b.matmul(probs, v, scope="ctx")
        ctx = b.permute(ctx, dims=(0, 2, 1, 3), scope="merge_heads")
        ctx = b.reshape(ctx, shape=(-1, seq, dim), scope="ctx_flat")
        ctx = b.linear(ctx, dim, scope="proj")
        ctx = _window_reverse(b, ctx, win, (n, dim, ph, pw), "reverse")
    return _cyclic_shift(b, ctx, -shift, f"{scope}.unshift")


def _swin_stage_block(b: GraphBuilder, x: str, dim: int, num_heads: int, win: int,
                      shift: int, mlp_ratio: float, scope: str) -> str:
    """Swin block branch: (shifted) window attention then a GELU MLP."""
    with b.block(scope):
        y = b.permute(x, dims=(0, 2, 3, 1), scope="attn_to_nhwc")
        y = b.layernorm(y, normalized_shape=(dim,), eps=1e-5, scope="attn_norm")
        y = b.permute(y, dims=(0, 3, 1, 2), scope="attn_to_nchw")
        y = _shifted_window_attention(b, y, dim, num_heads, win, shift, "attn")
        x = b.add(x, y, scope="attn_residual")
        z = b.permute(x, dims=(0, 2, 3, 1), scope="mlp_to_nhwc")
        z = b.layernorm(z, normalized_shape=(dim,), eps=1e-5, scope="mlp_norm")
        hidden = max(1, int(round(dim * mlp_ratio)))
        z = b.linear(z, hidden, act="gelu", scope="mlp_fc1")
        z = b.linear(z, dim, scope="mlp_fc2")
        return b.permute(z, dims=(0, 3, 1, 2), scope="mlp_to_nchw")


def _patch_merging(b: GraphBuilder, x: str, out_dim: int, scope: str) -> str:
    """Patch merging: concat 2x2 patch tiles, LayerNorm, 1x1 projection."""
    n, c, h, w = _shape_of(x, b)
    with b.block(scope):
        y = b.reshape(x, shape=(-1, c, h // 2, 2, w // 2, 2), scope="grid")
        # (n, c, h/2, 2, w/2, 2) -> (n, c, 2, 2, h/2, w/2): the two tile axes move
        # next to the channels, so flattening them yields the 4c-channel
        # half-resolution feature map of Swin patch merging.
        y = b.permute(y, dims=(0, 1, 3, 5, 2, 4), scope="concat_tiles")
        y = b.reshape(y, shape=(-1, 4 * c, h // 2, w // 2), scope="merge_tiles")
        y = b.permute(y, dims=(0, 2, 3, 1), scope="to_nhwc")
        y = b.layernorm(y, normalized_shape=(4 * c,), eps=1e-5, scope="norm")
        y = b.permute(y, dims=(0, 3, 1, 2), scope="to_nchw")
        return _conv(b, y, out_dim, kernel_size=1, act="", norm="", scope="reduction")


def _multi_head_attention(b: GraphBuilder, q_in: str, kv_in: str, dim: int,
                          num_heads: int, scope: str) -> str:
    """Attention over ``(n, seq, dim)`` built from ``matmul``/``softmax``."""
    n, seq_q = (int(v) for v in b.graph.edges[q_in].spec.shape[:2])
    seq_kv = int(b.graph.edges[kv_in].spec.shape[1])
    hd = dim // num_heads
    with b.block(scope):
        q = b.linear(q_in, dim, scope="q_proj")
        k = b.linear(kv_in, dim, scope="k_proj")
        v = b.linear(kv_in, dim, scope="v_proj")
        q = b.reshape(q, shape=(-1, seq_q, num_heads, hd), scope="q_heads")
        k = b.reshape(k, shape=(-1, seq_kv, num_heads, hd), scope="k_heads")
        v = b.reshape(v, shape=(-1, seq_kv, num_heads, hd), scope="v_heads")
        q = b.permute(q, dims=(0, 2, 1, 3), scope="q_t")
        k = b.permute(k, dims=(0, 2, 3, 1), scope="k_t")
        v = b.permute(v, dims=(0, 2, 1, 3), scope="v_t")
        logits = b.matmul(q, k, scope="logits")
        logits = b.mul(logits, other=hd ** -0.5, scope="scale")
        probs = b.softmax(logits, dim=-1, scope="probs")
        ctx = b.matmul(probs, v, scope="ctx")
        ctx = b.permute(ctx, dims=(0, 2, 1, 3), scope="merge_heads")
        ctx = b.reshape(ctx, shape=(-1, seq_q, dim), scope="ctx_flat")
        return b.linear(ctx, dim, scope="out_proj")


def _masked_cross_attention(b: GraphBuilder, q_in: str, memory: str, mask: str,
                            dim: int, num_heads: int, scope: str) -> str:
    """Cross-attention whose logits are gated by a per-query mask (masked attention)."""
    n, seq_q = (int(v) for v in b.graph.edges[q_in].spec.shape[:2])
    seq_kv = int(b.graph.edges[memory].spec.shape[1])
    hd = dim // num_heads
    with b.block(scope):
        q = b.linear(q_in, dim, scope="q_proj")
        k = b.linear(memory, dim, scope="k_proj")
        v = b.linear(memory, dim, scope="v_proj")
        q = b.reshape(q, shape=(-1, seq_q, num_heads, hd), scope="q_heads")
        k = b.reshape(k, shape=(-1, seq_kv, num_heads, hd), scope="k_heads")
        v = b.reshape(v, shape=(-1, seq_kv, num_heads, hd), scope="v_heads")
        q = b.permute(q, dims=(0, 2, 1, 3), scope="q_t")
        k = b.permute(k, dims=(0, 2, 3, 1), scope="k_t")
        v = b.permute(v, dims=(0, 2, 1, 3), scope="v_t")
        logits = b.matmul(q, k, scope="logits")
        logits = b.mul(logits, other=hd ** -0.5, scope="scale")
        bias = b.mul(mask, other=-1e4, scope="mask_bias")
        bias = b.unsqueeze(bias, dim=1, scope="mask_heads")
        logits = b.add(logits, bias, scope="masked_logits")
        probs = b.softmax(logits, dim=-1, scope="probs")
        ctx = b.matmul(probs, v, scope="ctx")
        ctx = b.permute(ctx, dims=(0, 2, 1, 3), scope="merge_heads")
        ctx = b.reshape(ctx, shape=(-1, seq_q, dim), scope="ctx_flat")
        return b.linear(ctx, dim, scope="out_proj")


def _masked_decoder_layer(b: GraphBuilder, x: str, memory: str, mask: str, dim: int,
                          num_heads: int, scope: str) -> str:
    """Masked-attention transformer decoder layer (self-attn, masked cross-attn, FFN)."""
    with b.block(scope):
        h = b.layernorm(x, normalized_shape=(dim,), eps=1e-5, scope="self_norm")
        h = _multi_head_attention(b, h, h, dim, num_heads, "self_attn")
        x = b.add(h, x, scope="self_residual")

        h = b.layernorm(x, normalized_shape=(dim,), eps=1e-5, scope="cross_norm")
        h = _masked_cross_attention(b, h, memory, mask, dim, num_heads, "cross_attn")
        x = b.add(h, x, scope="cross_residual")

        y = b.layernorm(x, normalized_shape=(dim,), eps=1e-5, scope="ffn_norm")
        y = b.linear(y, 2 * dim, act="gelu", scope="ffn.0")
        y = b.linear(y, dim, scope="ffn.1")
        return b.add(y, x, scope="ffn_residual")


def build_mask2former_swin_t(cfg: Dict[str, Any] = MASK2FORMER_SWIN_T) -> Any:
    """Build the (compact by default) Mask2Former-Swin-T graph."""
    b = GraphBuilder("mask2former_swin_t")
    depths = tuple(cfg["depths"])
    dims = tuple(cfg["dims"])
    heads = tuple(cfg["heads"])
    win = int(cfg["window"])
    stride = int(cfg["patch_stride"])
    dim = int(cfg["hidden_dim"])
    nh = int(cfg["num_heads"])
    nq = int(cfg["num_queries"])
    mask_dim = int(cfg["mask_dim"])
    x = b.input("images", (1, 3, 512, 512))

    # -- Swin-style backbone ---------------------------------------------
    with b.block("backbone"):
        h = _conv(b, x, dims[0], kernel_size=stride, stride=stride, padding=0,
                  act="", norm="", scope="patch_embed")
        h = b.layernorm(
            b.permute(h, dims=(0, 2, 3, 1), scope="pe_to_nhwc"),
            normalized_shape=(dims[0],), eps=1e-5, scope="pe_norm")
        h = b.permute(h, dims=(0, 3, 1, 2), scope="pe_to_nchw")
        feats: List[str] = []
        for si, (depth, d, nhead) in enumerate(zip(depths, dims, heads)):
            if si:
                h = _patch_merging(b, h, d, f"patch_merging.{si - 1}")
            for bi in range(int(depth)):
                shift = 0 if bi % 2 == 0 else win // 2
                h = b.residual(
                    h,
                    partial(_swin_stage_block, b, dim=d, num_heads=int(nhead), win=win,
                            shift=shift, mlp_ratio=4.0, scope=f"blocks.{bi}"),
                    scope=f"stages.{si}.blocks.{bi}")
            feats.append(h)

    # -- pixel decoder: lateral projections + multi-scale fusion ----------
    # Every decoder feature carries the symbolic batch; ``concat`` requires the
    # operands to agree on it, so normalise the backbone outputs first.
    feats = [_as_symbolic_batch(b, f) for f in feats]
    feats = feats[-int(cfg["decoder_levels"]):]
    with b.block("pixel_decoder"):
        lateral = [_cba(b, f, dim, scope=f"lateral.{i}") for i, f in enumerate(feats)]
        # fuse from the coarsest scale upwards with interpolate + concat
        fused = lateral[-1]
        for i in range(len(lateral) - 2, -1, -1):
            size = _shape_of(lateral[i], b)[2:]
            up = b.interpolate(fused, size=size, mode="bilinear", align_corners=False,
                               scope=f"upsample.{i}")
            fused = b.concat(lateral[i], up, dim=1, scope=f"concat.{i}")
            fused = _cba(b, fused, dim, kernel_size=3, scope=f"fuse.{i}")
        # three multi-scale memories for the decoder
        n, c, hh, ww = _shape_of(fused, b)
        maps = [fused]
        for i in range(2):
            size = (max(1, hh // (2 ** (i + 1))), max(1, ww // (2 ** (i + 1))))
            maps.append(_conv(b, b.interpolate(fused, size=size, mode="bilinear",
                                                   align_corners=False,
                                                   scope=f"down_up.{i}"),
                                     dim, kernel_size=3, stride=2, padding=1,
                                     scope=f"down.{i}"))
        flat = []
        for i, m in enumerate(maps):
            mn, mc, mh, mw = _shape_of(m, b)
            fm = b.permute(b.reshape(m, shape=(-1, mc, mh * mw), scope=f"flat.{i}"),
                           dims=(0, 2, 1), scope=f"to_seq.{i}")
            flat.append(fm)
        memory = b.concat(*flat, dim=1, scope="memory")
        memory = b.linear(memory, dim, scope="memory_proj")
        # pixel features used by the mask head, in the same token layout as the
        # memory the decoder cross-attends to
        mask_feature = b.linear(memory, mask_dim, scope="mask_feature")

    # -- transformer decoder with masked attention -----------------------
    with b.block("decoder"):
        w_init = b.param((nq, dim), init="xavier_uniform", name="query_embed.weight")
        # the batch dimension must be concrete here (expand has no -1); the
        # declared spec is rewritten to -1 by ``_symbolic_batch_graph``
        queries = b.expand(w_init, shape=(1, nq, dim), scope="queries")
        z = queries
        for i in range(int(cfg["num_layers"])):
            # masked attention: the mask of layer i gates the cross-attention of
            # layer i+1, so it is predicted over the same image tokens as the
            # memory keys/values.
            mask_logits = b.linear(z, mask_dim, scope=f"mask_embed.{i}")
            mask = b.sigmoid(mask_logits, scope=f"mask.{i}")
            mask = b.matmul(mask, b.permute(mask_feature, dims=(0, 2, 1),
                                            scope=f"mask_feature_t.{i}"),
                            scope=f"mask_pred.{i}")
            z = _masked_decoder_layer(b, z, memory, mask, dim, nh, f"layer.{i}")
        z = b.layernorm(z, normalized_shape=(dim,), eps=1e-5, scope="norm")
        pred_logits = b.linear(z, int(cfg["num_classes"]) + 1, scope="class_embed")
        mask_embed = b.linear(z, mask_dim, scope="mask_embed")
        # per-query masks: dot(mask_embed, pixel features) -> (n, nq, num_tokens)
        pred_masks = b.matmul(
            mask_embed,
            b.permute(mask_feature, dims=(0, 2, 1), scope="mask_feature_t"),
            scope="pred_masks")

    b.output([pred_masks, pred_logits], names=["pred_masks", "pred_logits"])
    return _symbolic_batch_graph(b)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def seed_segformer_b1() -> SeedModel:
    return SeedModel(
        key="segformer_b1",
        task="Semantic Segmentation",
        builder=build_segformer_b1,
        input_shapes={"images": (1, 3, 512, 512)},
        input_dtypes={"images": "float32"},
        param_count=13715798,
        input_size="512x512",
        reference="SegFormer (Xie et al. 2021)",
    )


def seed_mask2former_swin_t() -> SeedModel:
    return SeedModel(
        key="mask2former_swin_t",
        task="Semantic Segmentation",
        builder=build_mask2former_swin_t,
        input_shapes={"images": (1, 3, 512, 512)},
        input_dtypes={"images": "float32"},
        param_count=47441169,
        input_size="512x512",
        reference="Mask2Former (Cheng et al. 2022)",
    )


MODELS = [
    seed_segformer_b1(),
    seed_mask2former_swin_t(),
]

for _m in MODELS:
    register_seed(_m)
