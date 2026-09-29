"""Object-detection seed models.

* **YOLOv10-S** -- CSP-style convolutional backbone, SPPF (successive
  max-pools + ``concat``), a PAN/FPN neck with nearest-neighbour ``interpolate``
  upsampling and ``concat`` fusion, and three anchor-free detection heads
  emitting box regression, classification logits and a dense score map.
* **RT-DETRv2-S** -- ResNet-style CNN backbone, a hybrid encoder that fuses
  multi-scale features and applies attention built from ``matmul``/``softmax``,
  and a transformer decoder whose layers perform self-attention over the object
  queries followed by cross-attention over the fused memory, ending in
  classification and box-regression heads.

Both are written at a compact scale by default (documented hyper-parameters are
recorded under ``"full"``), and neither imports a DL framework.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from flowmut.ir.graph import Graph, GraphBuilder
from flowmut.seed_models.registry import SeedModel, register_seed


def _shape_of(edge: str, b: GraphBuilder) -> Tuple[int, int, int, int]:
    s = b.graph.edges[edge].spec.shape
    return int(s[0]), int(s[1]), int(s[2]), int(s[3])



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
    """Conv + BatchNorm + SiLU, the YOLO/RT-DETR-style composite."""
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
# YOLOv10-S
# ---------------------------------------------------------------------------
# Documented (YOLOv10-S, Wang et al. 2024): width multiple 0.50, max channels
# 1024, depths (1, 2, 2, 1) for the C2f/SPPF stages, backbone widths
# (64, 128, 256, 512, 1024), PAN neck widths (256, 512, 512), three anchor-free
# heads with a shared conv trunk and box/class/score branches.
# Documented benchmark: 640x640 float32 NCHW, 8,128,272 parameters.
YOLOV10_S_CONFIGS: Dict[str, Dict[str, Any]] = {
    "compact": dict(
        stem_width=16,
        backbone_widths=(32, 64, 96, 128),
        backbone_blocks=(1, 2, 2, 1),
        neck_widths=(64, 96, 128),
        head_width=48,
        num_classes=80,
        anchors=2,
    ),
    "full": dict(
        stem_width=64,
        backbone_widths=(128, 256, 512, 1024),
        backbone_blocks=(3, 6, 6, 3),
        neck_widths=(256, 512, 512),
        head_width=256,
        num_classes=80,
        anchors=3,
    ),
}
YOLOV10_S = YOLOV10_S_CONFIGS["compact"]


def _csp_stage(b: GraphBuilder, x: str, out_channels: int, blocks: int,
               scope: str) -> str:
    """CSP-style stage: split the projections and fuse them after a bottleneck."""
    mid = max(1, out_channels // 2)
    with b.block(scope):
        left = _cba(b, x, mid, kernel_size=1, scope="conv1")
        right = _cba(b, x, mid, kernel_size=1, scope="conv2")
        for i in range(blocks):
            right = _cba(b, right, mid, kernel_size=3, scope=f"bottleneck.{i}")
        return _cba(b, b.concat(left, right, dim=1, scope="concat"), out_channels,
                    kernel_size=1, scope="fuse")


def _sppf(b: GraphBuilder, x: str, out_channels: int, scope: str) -> str:
    """Spatial Pyramid Pooling - Fast: three successive 5x5 max-pools, concatenated."""
    with b.block(scope):
        h = _cba(b, x, out_channels // 2, kernel_size=1, scope="reduce")
        p1 = b.maxpool2d(h, kernel_size=5, stride=1, padding=2, scope="pool1")
        p2 = b.maxpool2d(p1, kernel_size=5, stride=1, padding=2, scope="pool2")
        p3 = b.maxpool2d(p2, kernel_size=5, stride=1, padding=2, scope="pool3")
        fused = b.concat(h, p1, p2, p3, dim=1, scope="concat")
        return _cba(b, fused, out_channels, kernel_size=1, scope="fuse")


def _detection_head(b: GraphBuilder, feat: str, head_width: int, anchors: int,
                    num_classes: int, scope: str) -> Tuple[str, str, str]:
    """Shared-conv anchor-free head for one fused pyramid level.

    Returns ``(boxes, scores, cls_logits)`` as ``(n, anchors*k, h*w)`` tensors so
    the three levels merge into one dense anchor-free prediction set.
    """
    with b.block(scope):
        trunk = _cba(b, feat, head_width, kernel_size=3, scope="trunk.0")
        trunk = _cba(b, trunk, head_width, kernel_size=3, scope="trunk.1")
        boxes = _conv(b, trunk, anchors * 4, kernel_size=1, act="", norm="",
                      bias=True, scope="box")
        scores = _conv(b, trunk, anchors * 1, kernel_size=1, act="", norm="",
                       bias=True, scope="score")
        cls = _conv(b, trunk, anchors * num_classes, kernel_size=1, act="",
                    norm="", bias=True, scope="cls")
        _, _, hh, ww = _shape_of(boxes, b)
        boxes = b.reshape(boxes, shape=(-1, anchors * 4, hh * ww), scope="box_flat")
        scores = b.reshape(scores, shape=(-1, anchors, hh * ww), scope="score_flat")
        cls = b.reshape(cls, shape=(-1, anchors * num_classes, hh * ww),
                        scope="cls_flat")
        return boxes, scores, cls


def _fuse_levels(b: GraphBuilder, levels, size: Tuple[int, int], scope: str) -> str:
    """Resample every pyramid level to ``size`` and fuse them with ``concat``.

    The pyramid levels live at different strides, so each is interpolated to the
    level's grid before the channel-wise concatenation that forms the
    multi-scale detection features (the neck's cross-level fusion).
    """
    aligned = [b.interpolate(lv, size=size, mode="nearest",
                             scope=f"{scope}.resize.{i}")
               for i, lv in enumerate(levels)]
    return b.concat(*aligned, dim=1, scope=f"{scope}.concat")


def build_yolov10_s(cfg: Dict[str, Any] = YOLOV10_S) -> Any:
    """Build the (compact by default) YOLOv10-S graph."""
    b = GraphBuilder("yolov10_s")
    widths = tuple(cfg["backbone_widths"])
    nblocks = tuple(cfg["backbone_blocks"])
    neck = tuple(cfg["neck_widths"])
    anchors = int(cfg["anchors"])
    num_classes = int(cfg["num_classes"])
    x = b.input("images", (1, 3, 640, 640))

    with b.block("backbone"):
        h = _cba(b, x, int(cfg["stem_width"]), kernel_size=3, stride=2, scope="stem")
        h = _cba(b, h, int(cfg["stem_width"]) * 2, kernel_size=3, stride=2, scope="stem.1")
        c3 = _csp_stage(b, h, widths[0], nblocks[0], "stage.0")          # /8
        c4 = _csp_stage(b, _cba(b, c3, widths[1], stride=2, scope="down.1"),
                        widths[1], nblocks[1], "stage.1")                # /16
        c5 = _csp_stage(b, _cba(b, c4, widths[2], stride=2, scope="down.2"),
                        widths[2], nblocks[2], "stage.2")                # /32
        p5 = _sppf(b, c5, widths[3], "sppf")

    # -- PAN/FPN neck -----------------------------------------------------
    with b.block("neck"):
        n5 = _cba(b, p5, neck[2], kernel_size=1, scope="lateral.2")
        n4 = _cba(b, c4, neck[1], kernel_size=1, scope="lateral.1")
        n3 = _cba(b, c3, neck[0], kernel_size=1, scope="lateral.0")

        size4 = _shape_of(n4, b)[2:]
        size3 = _shape_of(n3, b)[2:]
        up5 = b.interpolate(n5, size=size4, mode="nearest", scope="upsample.2")
        n4 = _csp_stage(b, b.concat(n4, up5, dim=1, scope="fuse.1"), neck[1], 1,
                        "topdown.1")

        up4 = b.interpolate(n4, size=size3, mode="nearest", scope="upsample.1")
        n3 = _csp_stage(b, b.concat(n3, up4, dim=1, scope="fuse.0"), neck[0], 1,
                        "topdown.0")

        d3 = _cba(b, n3, neck[0], kernel_size=3, stride=2, scope="downsample.0")
        n4 = _csp_stage(b, b.concat(n4, d3, dim=1, scope="fuse.3"), neck[1], 1,
                        "bottomup.1")
        d4 = _cba(b, n4, neck[1], kernel_size=3, stride=2, scope="downsample.1")
        n5 = _csp_stage(b, b.concat(n5, d4, dim=1, scope="fuse.4"), neck[2], 1,
                        "bottomup.2")

    # -- three anchor-free heads -----------------------------------------
    # Each level sees the other levels resampled onto its own grid, then one
    # shared head predicts boxes, objectness scores and class logits.
    head_width = int(cfg["head_width"])
    levels = (n3, n4, n5)
    sizes = tuple(_shape_of(lv, b)[2:] for lv in levels)
    boxes: List[str] = []
    scores: List[str] = []
    clss: List[str] = []
    for i, size in enumerate(sizes):
        fused = _fuse_levels(b, levels, size, f"heads.{i}.fuse")
        bx, sc, cl = _detection_head(b, fused, head_width, anchors, num_classes,
                                     f"heads.{i}")
        boxes.append(bx)
        scores.append(sc)
        clss.append(cl)

    # Each level is (n, anchors*k, h*w) with its own h*w, so the levels are
    # concatenated along the spatial axis into one dense anchor-free prediction
    # set per output.
    b.output([b.concat(*boxes, dim=2, scope="merge_boxes"),
              b.concat(*scores, dim=2, scope="merge_scores"),
              b.concat(*clss, dim=2, scope="merge_cls")],
             names=["boxes", "scores", "cls"])
    return _symbolic_batch_graph(b)


# ---------------------------------------------------------------------------
# RT-DETRv2-S
# ---------------------------------------------------------------------------
# Documented (RT-DETRv2-S, Lv et al. 2024): ResNet-18 style CNN backbone,
# hybrid encoder (AIFI intra-scale attention + cross-scale feature fusion with
# CCFF), 6 transformer decoder layers, 300 object queries, 256-d embeddings,
# 8 heads.  Documented benchmark: 640x640 float32 NCHW, 20,226,500 parameters.
RTDETRV2_S_CONFIGS: Dict[str, Dict[str, Any]] = {
    "compact": dict(
        backbone_widths=(16, 32, 64, 128),
        backbone_blocks=(1, 1, 1, 1),
        hidden_dim=32,
        num_heads=4,
        num_queries=16,
        num_layers=2,
        num_classes=80,
        dim_feedforward=64,
    ),
    "full": dict(
        backbone_widths=(64, 128, 256, 512),
        backbone_blocks=(2, 2, 2, 2),
        hidden_dim=256,
        num_heads=8,
        num_queries=300,
        num_layers=6,
        num_classes=80,
        dim_feedforward=1024,
    ),
}
RTDETRV2_S = RTDETRV2_S_CONFIGS["compact"]


def _basic_block(b: GraphBuilder, x: str, out_channels: int, stride: int,
                 scope: str) -> str:
    """ResNet BasicBlock: two 3x3 convs with BN, residual, SiLU."""

    def branch(edge: str) -> str:
        y = _cba(b, edge, out_channels, kernel_size=3, stride=stride, scope="conv1")
        y = _cba(b, y, out_channels, kernel_size=3, scope="conv2")
        return y

    with b.block(scope):
        residual = b.residual(x, branch, scope="residual")
        return b.silu(residual, scope="act")


def _multi_head_attention(b: GraphBuilder, q_in: str, kv_in: str, dim: int,
                          num_heads: int, scope: str) -> str:
    """Attention over ``(n, seq, dim)`` built from ``matmul``/``softmax``.

    ``q_in`` and ``kv_in`` may be the same edge (self-attention) or different
    edges (cross-attention).
    """
    n, seq_q, _ = (int(v) for v in b.graph.edges[q_in].spec.shape)
    seq_kv = int(b.graph.edges[kv_in].spec.shape[1])
    hd = dim // num_heads
    scale = hd ** -0.5
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
        logits = b.mul(logits, other=scale, scope="scale")
        probs = b.softmax(logits, dim=-1, scope="probs")
        ctx = b.matmul(probs, v, scope="ctx")
        ctx = b.permute(ctx, dims=(0, 2, 1, 3), scope="merge_heads")
        ctx = b.reshape(ctx, shape=(-1, seq_q, dim), scope="ctx_flat")
        return b.linear(ctx, dim, scope="out_proj")


def _transformer_layer(b: GraphBuilder, x: str, memory: str, dim: int, num_heads: int,
                       ffn_dim: int, scope: str) -> str:
    """One DETR decoder layer: self-attention, cross-attention, feed-forward."""
    with b.block(scope):
        h = b.layernorm(x, normalized_shape=(dim,), eps=1e-5, scope="self_norm")
        h = _multi_head_attention(b, h, h, dim, num_heads, "self_attn")
        x = b.add(h, x, scope="self_residual")

        m = b.layernorm(memory, normalized_shape=(dim,), eps=1e-5, scope="memory_norm")
        q = b.layernorm(x, normalized_shape=(dim,), eps=1e-5, scope="cross_norm")
        h = _multi_head_attention(b, q, m, dim, num_heads, "cross_attn")
        x = b.add(h, x, scope="cross_residual")

        y = b.layernorm(x, normalized_shape=(dim,), eps=1e-5, scope="ffn_norm")
        y = b.linear(y, ffn_dim, act="gelu", scope="ffn.0")
        y = b.linear(y, dim, scope="ffn.1")
        return b.add(y, x, scope="ffn_residual")


def build_rtdetrv2_s(cfg: Dict[str, Any] = RTDETRV2_S) -> Any:
    """Build the (compact by default) RT-DETRv2-S graph."""
    b = GraphBuilder("rtdetrv2_s")
    widths = tuple(cfg["backbone_widths"])
    nblocks = tuple(cfg["backbone_blocks"])
    dim = int(cfg["hidden_dim"])
    heads = int(cfg["num_heads"])
    nq = int(cfg["num_queries"])
    x = b.input("images", (1, 3, 640, 640))

    with b.block("backbone"):
        h = _cba(b, x, widths[0] // 2, kernel_size=3, stride=2, scope="stem.0")
        h = _cba(b, h, widths[0], kernel_size=3, stride=2, scope="stem.1")
        for i, (w, n) in enumerate(zip(widths, nblocks)):
            for j in range(n):
                h = _basic_block(b, h, w, stride=1, scope=f"stage.{i}.block.{j}")
            if i + 1 < len(widths):
                h = _cba(b, h, widths[i + 1], kernel_size=3, stride=2,
                         scope=f"stage.{i}.downsample")
        # ``h`` now carries the coarsest feature map (1/32).
        c3 = _cba(b, h, dim, kernel_size=3, scope="encoder.lateral.2")

    # -- hybrid encoder: intra-scale attention over the flattened features --
    with b.block("encoder"):
        n, c, hh, ww = _shape_of(c3, b)
        tokens = b.reshape(c3, shape=(-1, c, hh * ww), scope="flatten")
        tokens = b.permute(tokens, dims=(0, 2, 1), scope="to_seq")
        attn = _multi_head_attention(b, tokens, tokens, dim, heads, "aifi")
        tokens = b.add(attn, tokens, scope="aifi_residual")
        tokens = b.layernorm(tokens, normalized_shape=(dim,), eps=1e-5, scope="norm")
        memory = b.linear(tokens, dim, scope="memory_proj")

        # -- cross-scale fusion back in image space ------------------------
        feat = b.reshape(memory, shape=(-1, dim, hh, ww), scope="to_map")
        feat = _cba(b, feat, dim, kernel_size=3, scope="fusion")
        memory = b.reshape(feat, shape=(-1, dim, hh * ww), scope="to_tokens")
        memory = b.permute(memory, dims=(0, 2, 1), scope="memory_seq")

    # -- decoder ----------------------------------------------------------
    with b.block("decoder"):
        w_init = b.param((nq, dim), init="xavier_uniform", name="query_embed.weight")
        # the batch dimension must be concrete here (expand has no -1); the
        # declared spec is rewritten to -1 by ``_symbolic_batch_graph``
        queries = b.expand(w_init, shape=(1, nq, dim), scope="queries")
        z = queries
        for i in range(int(cfg["num_layers"])):
            z = _transformer_layer(b, z, memory, dim, heads,
                                   int(cfg["dim_feedforward"]), f"layer.{i}")
        z = b.layernorm(z, normalized_shape=(dim,), eps=1e-5, scope="norm")
        logits = b.linear(z, int(cfg["num_classes"]), scope="class_embed")
        boxes = b.linear(z, 4, act="sigmoid", scope="bbox_embed")

    b.output([logits, boxes], names=["logits", "boxes"])
    return _symbolic_batch_graph(b)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def seed_yolov10_s() -> SeedModel:
    return SeedModel(
        key="yolov10_s",
        task="Object Detection",
        builder=build_yolov10_s,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtypes={"images": "float32"},
        param_count=8128272,
        input_size="640x640",
        reference="YOLOv10 (Wang et al. 2024)",
    )


def seed_rtdetrv2_s() -> SeedModel:
    return SeedModel(
        key="rtdetrv2_s",
        task="Object Detection",
        builder=build_rtdetrv2_s,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtypes={"images": "float32"},
        param_count=20226500,
        input_size="640x640",
        reference="RT-DETRv2 (Lv et al. 2024)",
    )


MODELS = [
    seed_yolov10_s(),
    seed_rtdetrv2_s(),
]

for _m in MODELS:
    register_seed(_m)
