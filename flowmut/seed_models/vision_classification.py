"""Image-classification seed models.

Three compact-scale variants of modern classification backbones, each defined
once in the framework-neutral IR:

* **ConvNeXt V2-Tiny** -- patchify stem, four stages of ConvNeXt V2 blocks
  (depthwise 7x7 -> channels-last LayerNorm -> 4x pointwise expand -> GELU ->
  GRN -> 1x1 project), strided-conv downsampling, global pool + linear head.
* **Swin V2-Tiny** -- patch embedding, four stages of shifted-window attention
  built explicitly from ``matmul``/``softmax`` (cosine attention with a clamped
  logit scale and relative position bias, as in Swin V2), patch merging between
  stages, LayerNorm + linear head.
* **EfficientNetV2-S** -- stem conv, six stages of FusedMBConv/MBConv blocks
  with squeeze-excitation, head conv, global pool + linear classifier.

The documented (paper-scale) hyper-parameters are recorded in the per-model
``*_CONFIGS`` dicts under ``"full"``; the default ``"compact"`` variant shrinks
depths and widths so that the smoke tests execute on CPU.  ``SeedModel`` always
reports the *documented* model's parameter count from the paper's benchmark
table.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple

from flowmut.ir.graph import Graph, GraphBuilder
from flowmut.seed_models.registry import SeedModel, register_seed

# ---------------------------------------------------------------------------
# ConvNeXt V2-Tiny
# ---------------------------------------------------------------------------
# Documented (ConvNeXt V2-Tiny, Woo et al. 2023):
#   depths = (3, 3, 9, 3), dims = (96, 192, 384, 768), expand ratio = 4,
#   depthwise 7x7, LayerNorm (channels-last, eps 1e-6) before the block and
#   after the depthwise conv, GRN after the expand GELU, layer-scale 1e-6.
# Documented benchmark: 224x224 float32 NCHW, 28,635,496 parameters.
CONVNEXT_V2_TINY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "compact": dict(
        depths=(1, 1, 2, 1),
        dims=(24, 48, 96, 192),
        patch_stride=4,
        expand_ratio=4.0,
        num_classes=1000,
    ),
    "full": dict(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        patch_stride=4,
        expand_ratio=4.0,
        num_classes=1000,
    ),
}
CONVNEXT_V2_TINY = CONVNEXT_V2_TINY_CONFIGS["compact"]


def _conv(b: GraphBuilder, x: str, out_channels: int, kernel_size: int, scope: str,
          stride: int = 1, padding: Optional[int] = None, groups: int = 1,
          act: str = "", norm: str = "", bias: bool = False) -> str:
    """``conv_block`` wrapper that makes the kernel/stride part of the scope.

    The parameter bank keys tensors by ``(op, attrs, scope)``, so two layers that
    share a scope but not a shape would collide; embedding the kernel geometry in
    the scope keeps every parameter node distinct.
    """
    if padding is None:
        # same-style only for stride 1; a strided layer pads 0 (a k=s downsample
        # must really halve the grid) unless the caller says otherwise.
        padding = kernel_size // 2 if stride == 1 else 0
    tag = f"k{kernel_size}_s{stride}_g{groups}_p{padding}"
    return b.conv_block(x, out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, groups=groups, act=act, norm=norm,
                        bias=bias, scope=f"{scope}.{tag}")


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


def _channel_norm(b: GraphBuilder, x: str, scope: str) -> str:
    """LayerNorm over the channel dimension of an NCHW tensor.

    ConvNeXt normalises channels-last, so the tensor is permuted to NHWC, the
    trailing (channel) axis is normalised, and the result is permuted back.
    """
    h = b.permute(x, dims=(0, 2, 3, 1), scope=f"{scope}.to_nhwc")
    c = int(b.graph.edges[h].spec.shape[-1])
    h = b.layernorm(h, normalized_shape=(c,), eps=1e-6, scope=f"{scope}.norm")
    return b.permute(h, dims=(0, 3, 1, 2), scope=f"{scope}.to_nchw")


def _grn(b: GraphBuilder, x: str, scope: str) -> str:
    """Global Response Normalisation (ConvNeXt V2).

    ``X * (gamma * N(X) + 1)`` with ``N(X) = ||X_i|| / mean_j ||X_j||`` computed
    per channel over the spatial dimensions.  ``gamma`` is a real parameter
    initialised to zero by the authors, so the residual stream is untouched at
    initialisation.
    """
    h = b.permute(x, dims=(0, 2, 3, 1), scope=f"{scope}.to_nhwc")      # NHWC
    c = int(b.graph.edges[h].spec.shape[-1])
    gx = b.mul(h, h, scope=f"{scope}.square")
    agg = b.mean(gx, dim=(1, 2), keepdim=True, scope=f"{scope}.spatial_mean")
    norm = b.sqrt(agg, scope=f"{scope}.l2")
    div = b.add(norm, other=1e-6, scope=f"{scope}.eps")
    nx = b.div(norm, div, scope=f"{scope}.normalise")
    gamma = b.param((1, 1, 1, c), init="zeros", name=f"{scope}.gamma")
    scaled = b.mul(nx, gamma, scope=f"{scope}.scale")
    gate = b.add(scaled, other=1.0, scope=f"{scope}.gate")
    out = b.mul(h, gate, scope=f"{scope}.excite")
    return b.permute(out, dims=(0, 3, 1, 2), scope=f"{scope}.to_nchw")


def _convnext_v2_block(b: GraphBuilder, x: str, dim: int, expand_ratio: float,
                       layer_scale: float, scope: str) -> str:
    """One ConvNeXt V2 block; returns the layer-scaled branch activation."""
    hidden = max(1, int(round(dim * expand_ratio)))
    with b.block(scope):
        h = _conv(b, x, dim, kernel_size=7, padding=3, groups=dim,
                         act="", norm="", scope="dwconv")
        h = _channel_norm(b, h, "norm")
        h = _conv(b, h, hidden, kernel_size=1, act="", norm="", scope="pwconv1")
        h = b.gelu(h, scope="act")
        h = _grn(b, h, "grn")
        h = _conv(b, h, dim, kernel_size=1, act="", norm="", scope="pwconv2")
        if layer_scale != 1.0:
            h = b.mul(h, other=layer_scale, scope="layer_scale")
    return h


def build_convnext_v2_tiny(cfg: Dict[str, Any] = CONVNEXT_V2_TINY) -> Any:
    """Build the (compact by default) ConvNeXt V2-Tiny graph."""
    depths: Tuple[int, ...] = tuple(cfg["depths"])
    dims: Tuple[int, ...] = tuple(cfg["dims"])
    b = GraphBuilder("convnext_v2_tiny")
    x = b.input("images", (1, 3, 224, 224))

    with b.block("stem"):
        h = _conv(b, x, dims[0], kernel_size=4, stride=4, padding=0, act="",
                  norm="", scope="patchify")
        h = _channel_norm(b, h, "norm")

    scale = 1e-6
    for si, (depth, dim) in enumerate(zip(depths, dims)):
        for bi in range(depth):
            block = partial(_convnext_v2_block, b, dim=dim,
                            expand_ratio=float(cfg["expand_ratio"]),
                            layer_scale=scale, scope=f"blocks.{bi}")
            h = b.residual(h, block, scope=f"stages.{si}.blocks.{bi}")
        if si < len(dims) - 1:
            out_dim = dims[si + 1]
            with b.block(f"downsample{si}"):
                h = _channel_norm(b, h, "norm")
                h = _conv(b, h, out_dim, kernel_size=2, stride=2, padding=0, act="",
                          norm="", scope="conv")
            scale = min(scale * 10.0, 1e-3)

    h = b.global_avgpool(h, keepdim=False, scope="head.pool")
    h = b.layernorm(h, normalized_shape=(int(dims[-1]),), eps=1e-6, scope="head.norm")
    logits = b.linear(h, int(cfg["num_classes"]), scope="head.fc")
    b.output([logits], names=["logits"])
    return _symbolic_batch_graph(b)


# ---------------------------------------------------------------------------
# Swin Transformer V2-Tiny
# ---------------------------------------------------------------------------
# Documented (Swin V2-Tiny, Liu et al. 2022):
#   depths = (2, 2, 6, 2), dims = (96, 192, 384, 768), heads = (3, 6, 12, 24),
#   window = 8 (V2), mlp_ratio = 4, cosine attention with logit_scale clamped at
#   100, layer-scale 1e-5, patch merging between stages.
# Documented benchmark: 256x256 float32 NCHW, 28,347,154 parameters.
SWIN_V2_TINY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "compact": dict(
        depths=(1, 1, 2, 1),
        dims=(24, 48, 96, 192),
        heads=(3, 6, 12, 24),
        window=4,
        patch_stride=4,
        mlp_ratio=4.0,
        num_classes=1000,
    ),
    "full": dict(
        depths=(2, 2, 6, 2),
        dims=(96, 192, 384, 768),
        heads=(3, 6, 12, 24),
        window=8,
        patch_stride=4,
        mlp_ratio=4.0,
        num_classes=1000,
    ),
}
SWIN_V2_TINY = SWIN_V2_TINY_CONFIGS["compact"]


def _shape_of(b: GraphBuilder, edge: str) -> Tuple[int, int, int, int]:
    s = b.graph.edges[edge].spec.shape
    return int(s[0]), int(s[1]), int(s[2]), int(s[3])



def _window_partition(b: GraphBuilder, x: str, win: int, scope: str) -> str:
    """``(n, c, h, w) -> (n*nw_h*nw_w, win*win, c)`` via reshape/permute only."""
    n, c, h, w = _shape_of(b, x)
    nw_h, nw_w = h // win, w // win
    y = b.reshape(x, shape=(-1, c, nw_h, win, nw_w, win), scope=f"{scope}.grid")
    y = b.permute(y, dims=(0, 2, 4, 3, 5, 1), scope=f"{scope}.windows")
    return b.reshape(y, shape=(-1, win * win, c), scope=f"{scope}.tokens")


def _window_reverse(b: GraphBuilder, x: str, win: int, shape: Tuple[int, int, int, int],
                    scope: str) -> str:
    """``(n*nw_h*nw_w, win*win, c) -> (n, c, h, w)``; inverse of partition."""
    n, c, h, w = shape
    nw_h, nw_w = h // win, w // win
    y = b.reshape(x, shape=(-1, nw_h, nw_w, win, win, c), scope=f"{scope}.windows")
    y = b.permute(y, dims=(0, 5, 1, 3, 2, 4), scope=f"{scope}.grid")
    return b.reshape(y, shape=(-1, c, h, w), scope=f"{scope}.merge")


def _cyclic_shift(b: GraphBuilder, x: str, shift: int, scope: str) -> str:
    """Cyclic ``(-shift, -shift)`` roll used by shifted-window attention."""
    if shift == 0:
        return x
    return b.roll(x, shifts=(-shift, -shift), dims=(2, 3), scope=scope)


def _swin_attention(b: GraphBuilder, x: str, dim: int, num_heads: int,
                    logit_scale: float, scope: str) -> str:
    """Swin V2 window attention with cosine logits and a relative position bias.

    ``x`` is ``(nwin, seq, c)``.  Attention is built from ``matmul``/``softmax``
    on purpose (no ``sdpa``) so the data flow is fully exposed to mutation.
    """
    shape = tuple(int(v) for v in b.graph.edges[x].spec.shape)
    nwin, seq, c = shape[0], shape[1], shape[2]
    hd = c // num_heads          # head dim; must divide the embedding dim
    with b.block(scope):
        qkv = b.linear(x, 3 * c, scope="qkv")
        # split the fused projection on its "3" axis, keeping (nwin, seq, c)
        qkv = b.reshape(qkv, shape=(-1, seq, 3, c), scope="unfuse_qkv")
        q, k, v = b.split(qkv, dim=2, num_outputs=3, _num_outputs=3, scope="qkv_parts")

        # -- cosine attention: normalise q and k over the embedding -------
        q = b.l2norm(q, dim=-1, eps=1e-12, scope="q_norm")
        k = b.l2norm(k, dim=-1, eps=1e-12, scope="k_norm")
        scale = min(logit_scale, 100.0) / (hd ** 0.5)
        q = b.reshape(q, shape=(-1, seq, num_heads, hd), scope="q_heads")
        k = b.reshape(k, shape=(-1, seq, num_heads, hd), scope="k_heads")
        v = b.reshape(v, shape=(-1, seq, num_heads, hd), scope="v_heads")
        q = b.permute(q, dims=(0, 2, 1, 3), scope="q_t")
        k = b.permute(k, dims=(0, 2, 3, 1), scope="k_t")
        v = b.permute(v, dims=(0, 2, 1, 3), scope="v_t")
        logits = b.matmul(q, k, scope="attn_logits")
        logits = b.mul(logits, other=scale, scope="logit_scale")

        # -- relative position bias, shared across windows ----------------
        bias = b.param((num_heads, seq, seq), init="zeros", name="relative_bias")
        bias = b.unsqueeze(bias, dim=0, scope="bias_batch")
        logits = b.add(logits, bias, scope="logits_bias")

        probs = b.softmax(logits, dim=-1, scope="attn_probs")
        ctx = b.matmul(probs, v, scope="attn_ctx")
        ctx = b.permute(ctx, dims=(0, 2, 1, 3), scope="merge_heads")
        ctx = b.reshape(ctx, shape=(-1, seq, c), scope="attn_out")
        return b.linear(ctx, c, scope="proj")


def _swin_block(b: GraphBuilder, x: str, dim: int, num_heads: int, win: int,
                shift: int, mlp_ratio: float, layer_scale: float, scope: str) -> str:
    """Branch of one Swin V2 block (window attention + Mix-FFN).

    Both sub-layers are pre-norm and their outputs are layer-scaled; the
    caller's ``b.residual`` adds the skip connection together with the
    stochastic-depth/token-mixing path.
    """
    with b.block(scope):
        y = b.permute(x, dims=(0, 2, 3, 1), scope="attn_to_nhwc")
        y = b.layernorm(y, normalized_shape=(dim,), eps=1e-5, scope="attn_norm")
        y = b.permute(y, dims=(0, 3, 1, 2), scope="attn_to_nchw")
        # -- shifted windows: roll the feature map, and roll it back after ---
        y = _cyclic_shift(b, y, shift, "shift")
        cur = _shape_of(b, y)
        pdims = (cur[0], cur[1], cur[2] + (-cur[2]) % win,
                 cur[3] + (-cur[3]) % win)
        if pdims[2] != cur[2] or pdims[3] != cur[3]:
            y = b.pad(y, pad=(0, pdims[3] - cur[3], 0, pdims[2] - cur[2]),
                      mode="constant", value=0.0, _shape=pdims, scope="pad")
        y = _window_partition(b, y, win, "partition")
        y = _swin_attention(b, y, dim, num_heads, 14.0, "attn")
        y = _window_reverse(b, y, win, pdims, "reverse")
        y = _cyclic_shift(b, y, -shift, "unshift")
        if layer_scale != 1.0:
            y = b.mul(y, other=layer_scale, scope="attn_layer_scale")
        x = b.add(x, y, scope="attn_residual")

        # -- Mix-FFN: depthwise 3x3 -> GELU -> linear -> linear -----------
        z = b.permute(x, dims=(0, 2, 3, 1), scope="mlp_to_nhwc")
        z = b.layernorm(z, normalized_shape=(dim,), eps=1e-5, scope="mlp_norm")
        z = b.permute(z, dims=(0, 3, 1, 2), scope="mlp_to_nchw")
        z = _conv(b, z, dim, kernel_size=3, padding=1, groups=dim, act="gelu",
                         norm="", scope="mlp_dwconv")
        hidden = max(1, int(round(dim * mlp_ratio)))
        z = b.permute(z, dims=(0, 2, 3, 1), scope="mlp_nhwc")
        z = b.linear(z, hidden, act="gelu", scope="mlp_fc1")
        z = b.linear(z, dim, scope="mlp_fc2")
        z = b.permute(z, dims=(0, 3, 1, 2), scope="mlp_nchw")
        if layer_scale != 1.0:
            z = b.mul(z, other=layer_scale, scope="mlp_layer_scale")
        return z


def _swin_patch_merging(b: GraphBuilder, x: str, out_dim: int, scope: str) -> str:
    """Patch merging: concat 2x2 neighbouring patches, project, LayerNorm."""
    n, c, h, w = _shape_of(b, x)
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


def build_swin_v2_tiny(cfg: Dict[str, Any] = SWIN_V2_TINY) -> Any:
    """Build the (compact by default) Swin V2-Tiny graph."""
    b = GraphBuilder("swin_v2_tiny")
    depths: Tuple[int, ...] = tuple(cfg["depths"])
    dims: Tuple[int, ...] = tuple(cfg["dims"])
    heads: Tuple[int, ...] = tuple(cfg["heads"])
    win: int = int(cfg["window"])
    stride: int = int(cfg["patch_stride"])
    x = b.input("images", (1, 3, 256, 256))

    with b.block("patch_embed"):
        h = _conv(b, x, dims[0], kernel_size=stride, stride=stride, padding=0,
                  act="", norm="", scope="proj")

    scale = 1e-5
    for si, (depth, dim, nh) in enumerate(zip(depths, dims, heads)):
        if si:
            h = _swin_patch_merging(b, h, dim, f"patch_merging{si - 1}")
        for bi in range(depth):
            shift = 0 if bi % 2 == 0 else win // 2
            h = b.residual(
                h,
                partial(_swin_block, b, dim=dim, num_heads=nh, win=win, shift=shift,
                        mlp_ratio=float(cfg["mlp_ratio"]), layer_scale=scale,
                        scope=f"blocks.{bi}"),
                scope=f"stages.{si}.blocks.{bi}")
        scale = min(scale * 10.0, 1e-3)

    h = b.permute(h, dims=(0, 2, 3, 1), scope="norm_to_nhwc")
    h = b.layernorm(h, normalized_shape=(int(dims[-1]),), eps=1e-5, scope="norm")
    h = b.global_avgpool(h, keepdim=False, scope="head.pool")
    logits = b.linear(h, int(cfg["num_classes"]), scope="head.fc")
    b.output([logits], names=["logits"])
    return _symbolic_batch_graph(b)


# ---------------------------------------------------------------------------
# EfficientNetV2-S
# ---------------------------------------------------------------------------
# Documented (EfficientNetV2-S, Tan & Le 2021):
#   stem 3x3 stride 2 -> 24; stages (kernel, stride, expand, width, blocks, SE):
#     (3,1,1,24,2,no) (3,2,4,48,4,no) (3,2,4,64,4,no)
#     (3,2,4,128,6,yes) (3,1,6,160,9,yes) (3,2,6,256,15,yes)
#   head conv 1x1 -> 1280, SE ratio 0.25, SiLU activations.
# Documented benchmark: 384x384 float32 NCHW, 21,458,488 parameters.
EFFICIENTNETV2_S_CONFIGS: Dict[str, Dict[str, Any]] = {
    "compact": dict(
        stem_width=8,
        stage_cfg=(
            # (kernel, stride, expand, out_channels, blocks, se, fused)
            (3, 1, 1.0, 8, 1, False, True),
            (3, 2, 4.0, 16, 1, False, True),
            (3, 2, 4.0, 24, 1, False, True),
            (3, 2, 4.0, 48, 1, True, False),
            (3, 1, 6.0, 64, 1, True, False),
            (3, 2, 6.0, 96, 1, True, False),
        ),
        head_width=256,
        se_ratio=0.25,
        num_classes=1000,
    ),
    "full": dict(
        stem_width=24,
        stage_cfg=(
            (3, 1, 1.0, 24, 2, False, True),
            (3, 2, 4.0, 48, 4, False, True),
            (3, 2, 4.0, 64, 4, False, True),
            (3, 2, 4.0, 128, 6, True, False),
            (3, 1, 6.0, 160, 9, True, False),
            (3, 2, 6.0, 256, 15, True, False),
        ),
        head_width=1280,
        se_ratio=0.25,
        num_classes=1000,
    ),
}
EFFICIENTNETV2_S = EFFICIENTNETV2_S_CONFIGS["compact"]


def _squeeze_excite(b: GraphBuilder, x: str, channels: int, se_ratio: float,
                    scope: str) -> str:
    """SE: global pool -> reduce 1x1 -> SiLU -> expand 1x1 -> sigmoid -> mul."""
    reduced = max(1, int(round(channels * se_ratio)))
    with b.block(scope):
        s = b.global_avgpool(x, keepdim=True, scope="pool")
        s = _conv(b, s, reduced, kernel_size=1, act="silu", norm="", bias=True,
                         scope="reduce")
        s = _conv(b, s, channels, kernel_size=1, act="", norm="", bias=True,
                         scope="expand")
        s = b.sigmoid(s, scope="gate")
        return b.mul(x, s, scope="excite")


def _fused_mbconv(b: GraphBuilder, x: str, out_channels: int, kernel_size: int,
                  stride: int, expand_ratio: float, se_ratio: float, se: bool,
                  scope: str) -> str:
    """FusedMBConv: kxk expand conv (BN+SiLU) -> optional SE -> 1x1 project."""
    in_channels = int(b.graph.edges[x].spec.shape[1])
    mid = max(1, int(round(in_channels * expand_ratio)))
    with b.block(scope):
        h = _conv(b, x, mid, kernel_size=kernel_size, stride=stride,
                  act="silu", norm="batchnorm", scope="expand")
        if se and stride == 1:
            h = _squeeze_excite(b, h, mid, se_ratio, "se")
        return _conv(b, h, out_channels, kernel_size=1, act="", norm="batchnorm",
                     scope="project")


def _mbconv(b: GraphBuilder, x: str, out_channels: int, kernel_size: int,
            stride: int, expand_ratio: float, se_ratio: float, se: bool,
            scope: str) -> str:
    """MBConv: 1x1 expand -> depthwise kxk -> optional SE -> 1x1 project."""
    in_channels = int(b.graph.edges[x].spec.shape[1])
    mid = max(1, int(round(in_channels * expand_ratio)))
    with b.block(scope):
        h = _conv(b, x, mid, kernel_size=1, act="silu", norm="batchnorm",
                  scope="expand")
        h = _conv(b, h, mid, kernel_size=kernel_size, stride=stride, groups=mid,
                  act="silu", norm="batchnorm", scope="dwconv")
        if se and stride == 1:
            h = _squeeze_excite(b, h, mid, se_ratio, "se")
        return _conv(b, h, out_channels, kernel_size=1, act="", norm="batchnorm",
                     scope="project")


def build_efficientnetv2_s(cfg: Dict[str, Any] = EFFICIENTNETV2_S) -> Any:
    """Build the (compact by default) EfficientNetV2-S graph."""
    b = GraphBuilder("efficientnetv2_s")
    x = b.input("images", (1, 3, 384, 384))

    with b.block("stem"):
        h = _conv(b, x, int(cfg["stem_width"]), kernel_size=3, stride=2,
                  padding=1, act="silu", norm="batchnorm", scope="conv")

    for si, (k, stride, expand, out_c, blocks, se, fused) in enumerate(cfg["stage_cfg"]):
        for bi in range(blocks):
            s = int(stride) if bi == 0 else 1
            block_fn = _fused_mbconv if fused else _mbconv
            call = partial(block_fn, b, out_channels=int(out_c), kernel_size=int(k),
                           stride=s, expand_ratio=float(expand),
                           se_ratio=float(cfg["se_ratio"]), se=bool(se),
                           scope=f"blocks.{bi}")
            if s == 1 and int(b.graph.edges[h].spec.shape[1]) == int(out_c):
                h = b.residual(h, call, scope=f"stages.{si}.blocks.{bi}")
            else:
                h = call(h)

    with b.block("head"):
        h = _conv(b, h, int(cfg["head_width"]), kernel_size=1, act="silu",
                  norm="batchnorm", scope="conv")
        h = b.global_avgpool(h, keepdim=False, scope="pool")
    logits = b.linear(h, int(cfg["num_classes"]), scope="head.fc")
    b.output([logits], names=["logits"])
    return _symbolic_batch_graph(b)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def seed_convnext_v2_tiny() -> SeedModel:
    return SeedModel(
        key="convnext_v2_tiny",
        task="Image Classification",
        builder=build_convnext_v2_tiny,
        input_shapes={"images": (1, 3, 224, 224)},
        input_dtypes={"images": "float32"},
        param_count=28635496,
        input_size="224x224",
        reference="ConvNeXt V2 (Woo et al. 2023)",
    )


def seed_swin_v2_tiny() -> SeedModel:
    return SeedModel(
        key="swin_v2_tiny",
        task="Image Classification",
        builder=build_swin_v2_tiny,
        input_shapes={"images": (1, 3, 256, 256)},
        input_dtypes={"images": "float32"},
        param_count=28347154,
        input_size="256x256",
        reference="Swin Transformer V2 (Liu et al. 2022)",
    )


def seed_efficientnetv2_s() -> SeedModel:
    return SeedModel(
        key="efficientnetv2_s",
        task="Image Classification",
        builder=build_efficientnetv2_s,
        input_shapes={"images": (1, 3, 384, 384)},
        input_dtypes={"images": "float32"},
        param_count=21458488,
        input_size="384x384",
        reference="EfficientNetV2 (Tan & Le 2021)",
    )


MODELS = [
    seed_convnext_v2_tiny(),
    seed_swin_v2_tiny(),
    seed_efficientnetv2_s(),
]

for _m in MODELS:
    register_seed(_m)
