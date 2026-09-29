"""Vision seed models for anomaly detection, pose estimation and scene text.

* **EfficientAD-S** (industrial anomaly detection) -- a small teacher/student
  convolutional feature extractor with ``avgpool2d`` downsampling, a residual
  normalising-flow correction on the student branch, a convolutional autoencoder
  branch, and an anomaly-map head that fuses the student/teacher distance map
  with the autoencoder reconstruction error and upsamples back to the input
  resolution.
* **RTMPose-m** (2-D keypoint detection) -- a CSPNeXt-style backbone (stem conv,
  CSP stages whose main branch is a stack of residual bottlenecks, strided-conv
  downsampling) and a SimCC head that produces x- and y-coordinate logits with
  separate linear layers, plus an interpolated heatmap branch.
* **PARSeq** (scene text recognition) -- a ViT-style patch-embedding encoder
  (conv patchify -> ``flatten``/``permute`` -> pre-norm transformer blocks with
  hand-written multi-head self-attention) and a positional-query decoder whose
  learned queries first self-attend and then cross-attend over the encoder
  memory before a linear character-vocabulary head.

Every attention is built explicitly from ``linear``/``reshape``/``permute``/
``matmul``/``softmax`` (never ``sdpa``) so the data flow stays fully exposed.

Documented (paper-scale) hyper-parameters live in :data:`CONFIGS` under
``"full"``; the default ``"compact"`` variant shrinks the input resolution,
width and depth so that smoke tests execute on CPU.  ``SeedModel`` always
reports the *documented* model's parameter count from the paper's benchmark
table.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from flowmut.ir.graph import GraphBuilder
from flowmut.seed_models.registry import SeedModel, register_seed

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
# EfficientAD-S (Batzner et al. 2024):
#   teacher and student (PDN) CNNs with 4x4 / 3x3 convolutions and 2x2 average
#   pooling, a normalising flow on the student features, a convolutional
#   autoencoder branch, and a single-channel anomaly map; 8,057,856 parameters
#   at the documented 256x256 input.
#
# RTMPose-m (Jiang et al. 2023):
#   CSPNeXt-m backbone (stem 32, stages 64/128/256/512, 2 residual bottlenecks
#   per stage), SimCC head with split ratio 2.0 and 17 keypoints, 13,587,611
#   parameters at the documented 256x192 input.
#
# PARSeq (Bautista & Atienza 2022):
#   ViT-style encoder (patch 8x8, dim 384, 12 layers, 6 heads), 4-layer
#   positional-query decoder (25 learned queries) with cross-attention, and a
#   linear head over the 94-character vocabulary; 23,832,671 parameters at the
#   documented 32x128 input.
CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "compact": {
        "efficientad_s": dict(
            image=(1, 3, 64, 64),
            channels=(16, 32, 64),
            ae_channels=(16, 32, 64),
        ),
        "rtmpose_m": dict(
            image=(1, 3, 64, 48),
            stem_width=16,
            stage_channels=(16, 32, 64),
            num_keypoints=2,
            simcc_split=2.0,
            heatmap=True,
        ),
        "parseq": dict(
            image=(1, 3, 32, 128),
            patch_size=8,
            hidden=64,
            encoder_layers=2,
            decoder_layers=2,
            num_heads=4,
            mlp_hidden=128,
            num_queries=8,
            vocab_size=128,
            query_pos=True,
        ),
    },
    "full": {
        "efficientad_s": dict(
            image=(1, 3, 256, 256),
            channels=(64, 128, 256),
            ae_channels=(64, 128, 256),
        ),
        "rtmpose_m": dict(
            image=(1, 3, 256, 192),
            stem_width=32,
            stage_channels=(64, 128, 256, 512),
            num_keypoints=17,
            simcc_split=2.0,
            heatmap=True,
        ),
        "parseq": dict(
            image=(1, 3, 32, 128),
            patch_size=8,
            hidden=384,
            encoder_layers=12,
            decoder_layers=4,
            num_heads=6,
            mlp_hidden=1536,
            num_queries=25,
            vocab_size=94,
            query_pos=True,
        ),
    },
}

DEFAULT_CONFIG = "compact"

EFFICIENTAD_S = CONFIGS[DEFAULT_CONFIG]["efficientad_s"]
RTMPOSE_M = CONFIGS[DEFAULT_CONFIG]["rtmpose_m"]
PARSEQ = CONFIGS[DEFAULT_CONFIG]["parseq"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _shape_of(b: GraphBuilder, edge: str) -> Tuple[int, ...]:
    return tuple(int(v) for v in b.graph.edges[edge].spec.shape)


def _channels_of(b: GraphBuilder, edge: str) -> int:
    return int(_shape_of(b, edge)[1])


def _conv_transpose(b: GraphBuilder, x: str, out_channels: int, stride: int = 2,
                    kernel_size: int = 3, scope: str = "deconv") -> str:
    """Strided transposed convolution + batch norm + ReLU (doubles the map)."""
    in_channels = _channels_of(b, x)
    w = b.param((in_channels, out_channels, kernel_size, kernel_size),
                init="kaiming_uniform", name=f"{scope}.weight")
    y = b.conv_transpose2d(x, w, stride=stride, padding=kernel_size // 2,
                           output_padding=stride - 1, groups=1, _scope=scope)
    y = b.batchnorm(y, num_features=out_channels, _scope=f"{scope}.norm")
    return b.relu(y, _scope=f"{scope}.act")


def _split_heads(b: GraphBuilder, x: str, batch: int, seq: int, heads: int,
                 head_dim: int, scope: str) -> str:
    h = b.reshape(x, shape=(batch, seq, heads, head_dim), _scope=f"{scope}.reshape")
    return b.permute(h, dims=(0, 2, 1, 3), _scope=f"{scope}.permute")


def _merge_heads(b: GraphBuilder, x: str, batch: int, seq: int, dim: int,
                 scope: str) -> str:
    h = b.permute(x, dims=(0, 2, 1, 3), _scope=f"{scope}.permute")
    return b.reshape(h, shape=(batch, seq, dim), _scope=f"{scope}.reshape")


def _multi_head_attention(b: GraphBuilder, q_in: str, kv_in: str, dim: int,
                          heads: int, scope: str) -> str:
    """Explicit scaled dot-product attention over two rank-3 token sequences."""
    head_dim = dim // heads
    bq, sq, _ = _shape_of(b, q_in)
    bk, sk, _ = _shape_of(b, kv_in)
    with b.block(scope):
        q = b.linear(q_in, dim, scope="q")
        k = b.linear(kv_in, dim, scope="k")
        v = b.linear(kv_in, dim, scope="v")
        qh = _split_heads(b, q, bq, sq, heads, head_dim, "q")
        kh = _split_heads(b, k, bk, sk, heads, head_dim, "k")
        vh = _split_heads(b, v, bk, sk, heads, head_dim, "v")
        kt = b.permute(kh, dims=(0, 1, 3, 2), _scope="k.transpose")
        logits = b.matmul(qh, kt, _scope="logits")
        logits = b.mul(logits, other=1.0 / (head_dim ** 0.5), _scope="scale")
        probs = b.softmax(logits, dim=-1, _scope="probs")
        ctx = b.matmul(probs, vh, _scope="ctx")
        ctx = _merge_heads(b, ctx, bq, sq, dim, "ctx")
        return b.linear(ctx, dim, scope="out")


# ---------------------------------------------------------------------------
# EfficientAD-S
# ---------------------------------------------------------------------------
# Documented (EfficientAD-S, Batzner et al. 2024): teacher/student (PDN) CNNs
# with 3x3 convolutions and 2x2 average-pool downsampling, a normalising flow on
# the student features, a convolutional autoencoder branch and a one-channel
# anomaly map; 256x256 input, 8,057,856 parameters.

def _efficientad_extractor(b: GraphBuilder, x: str, channels: Tuple[int, ...],
                           scope: str) -> str:
    """Student/teacher feature extractor: conv blocks with avg-pool downsampling."""
    with b.block(scope):
        h = b.conv_block(x, channels[0], kernel_size=3, stride=2, act="relu",
                         norm="batchnorm", scope="stem")
        for i, c in enumerate(channels):
            h = b.conv_block(h, c, kernel_size=3, stride=1, act="relu",
                             norm="batchnorm", scope=f"stages.{i}.conv")
            if i < len(channels) - 1:
                h = b.avgpool2d(h, kernel_size=2, stride=2, _scope=f"stages.{i}.pool")
                h = b.conv_block(h, c, kernel_size=3, stride=1, act="relu",
                                 norm="batchnorm", scope=f"stages.{i}.refine")
        return h


def _efficientad_autoencoder(b: GraphBuilder, x: str, channels: Tuple[int, ...],
                             out_channels: int, scope: str) -> str:
    """Convolutional bottleneck autoencoder reconstructing the input frame."""
    with b.block(scope):
        h = x
        for i, c in enumerate(channels):
            h = b.conv_block(h, c, kernel_size=3, stride=2, act="relu",
                             norm="batchnorm", scope=f"encoder.{i}")
        for i, c in enumerate(reversed(channels)):
            out_c = channels[len(channels) - 2 - i] if i < len(channels) - 1 else out_channels
            h = _conv_transpose(b, h, int(out_c), stride=2, kernel_size=3,
                                scope=f"decoder.{i}")
        return b.conv_block(h, out_channels, kernel_size=3, stride=1, act="",
                            norm="", scope="refine")


def build_efficientad_s(cfg: Dict[str, Any] = EFFICIENTAD_S) -> Any:
    """Build the (compact by default) EfficientAD-S graph."""
    image: Tuple[int, ...] = tuple(int(v) for v in cfg["image"])
    channels: Tuple[int, ...] = tuple(int(v) for v in cfg["channels"])
    ae_channels: Tuple[int, ...] = tuple(int(v) for v in cfg["ae_channels"])
    b = GraphBuilder("efficientad_s")
    x = b.input("image", image, dtype="float32")

    # Teacher: a frozen (stop-gradient) copy of the feature extractor.
    with b.block("teacher"):
        teacher = _efficientad_extractor(b, x, channels, "extractor")
        teacher = b.stop_gradient(teacher, _scope="freeze")

    # Student + normalising flow on the student features.
    with b.block("student"):
        student = _efficientad_extractor(b, x, channels, "extractor")
        flow = b.conv_block(student, channels[-1], kernel_size=1, stride=1,
                            act="tanh", norm="", scope="flow.0")
        flow = b.conv_block(flow, channels[-1], kernel_size=1, stride=1, act="",
                            norm="", scope="flow.2")
        student = b.add(student, flow, _scope="flow.residual")

    diff = b.sub(student, teacher, _scope="distance.diff")
    diff = b.abs(diff, _scope="distance.abs")
    dist = b.mean(diff, dim=1, keepdim=True, _scope="distance.channel_mean")
    _, _, height, width = _shape_of(b, x)
    dist_up = b.interpolate(dist, size=(height, width), mode="nearest",
                            _scope="distance.upsample")

    # Autoencoder branch: per-pixel reconstruction error.
    recon = _efficientad_autoencoder(b, x, ae_channels, image[1], "autoencoder")
    err = b.abs(b.sub(x, recon, _scope="reconstruction.diff"), _scope="reconstruction.abs")
    err = b.mean(err, dim=1, keepdim=True, _scope="reconstruction.channel_mean")

    anomaly = b.add(b.mul(err, other=0.5, _scope="anomaly.pixel_weight"), dist_up,
                    _scope="anomaly.map")
    score = b.mean(anomaly, dim=(2, 3), keepdim=True, _scope="score.mean")
    score = b.flatten(score, start_dim=1, _scope="score.flatten")
    b.output([anomaly, score], names=["anomaly_map", "score"])
    return b.build()


# ---------------------------------------------------------------------------
# RTMPose-m
# ---------------------------------------------------------------------------

def _bottleneck(b: GraphBuilder, x: str, out_channels: int, stride: int = 1,
                scope: str = "bottleneck") -> str:
    """Residual bottleneck: 1x1 -> 3x3 -> 1x1 with a strided shortcut."""
    in_channels = _channels_of(b, x)
    mid = max(1, out_channels // 2)
    with b.block(scope):
        y = b.conv_block(x, mid, kernel_size=1, stride=1, act="silu",
                         norm="batchnorm", scope="conv1")
        y = b.conv_block(y, mid, kernel_size=3, stride=stride, act="silu",
                         norm="batchnorm", scope="conv2")
        y = b.conv_block(y, out_channels, kernel_size=1, stride=1, act="",
                         norm="batchnorm", scope="conv3")
        skip = x
        if in_channels != out_channels or stride != 1:
            skip = b.conv_block(x, out_channels, kernel_size=1, stride=stride,
                                act="", norm="batchnorm", scope="shortcut")
        return b.add(skip, y, _scope="add")


def _csp_stage(b: GraphBuilder, x: str, out_channels: int, num_blocks: int,
               scope: str) -> str:
    """CSPNeXt stage: split channels, run bottlenecks on one half, concat, project."""
    in_channels = _channels_of(b, x)
    if in_channels % 2:
        raise ValueError("CSP stage needs an even channel count")
    half = in_channels // 2
    with b.block(scope):
        side, main = b.split(x, dim=1, num_outputs=2, _num_outputs=2, _scope="split")
        side = b.conv_block(side, half, kernel_size=1, stride=1, act="silu",
                            norm="batchnorm", scope="side")
        main = b.conv_block(main, half, kernel_size=1, stride=1, act="silu",
                            norm="batchnorm", scope="main.0")
        for i in range(num_blocks):
            main = _bottleneck(b, main, half, stride=1, scope=f"main.{i + 1}")
        merged = b.concat(side, main, dim=1, _scope="concat")
        return b.conv_block(merged, out_channels, kernel_size=1, stride=1, act="silu",
                            norm="batchnorm", scope="project")


def build_rtmpose_m(cfg: Dict[str, Any] = RTMPOSE_M) -> Any:
    """Build the (compact by default) RTMPose-m graph."""
    image: Tuple[int, ...] = tuple(int(v) for v in cfg["image"])
    channels: Tuple[int, ...] = tuple(int(v) for v in cfg["stage_channels"])
    num_keypoints = int(cfg["num_keypoints"])
    split = float(cfg["simcc_split"])
    batch = int(image[0])
    b = GraphBuilder("rtmpose_m")
    x = b.input("image", image, dtype="float32")

    with b.block("backbone"):
        h = b.conv_block(x, int(cfg["stem_width"]), kernel_size=3, stride=2,
                         act="silu", norm="batchnorm", scope="stem")
        for i, c in enumerate(channels):
            h = _csp_stage(b, h, c, num_blocks=2, scope=f"stages.{i}")
            if i < len(channels) - 1:
                h = b.conv_block(h, channels[i + 1], kernel_size=3, stride=2,
                                 act="silu", norm="batchnorm", scope=f"transition.{i}")
        feat = b.conv_block(h, channels[-1], kernel_size=1, stride=1, act="silu",
                            norm="batchnorm", scope="final")

    _, feat_c, feat_h, feat_w = _shape_of(b, feat)
    x_bins = max(1, int(round(feat_w * split)))
    y_bins = max(1, int(round(feat_h * split)))

    # SimCC x head: global average over the spatial map, then a linear classifier
    # over the (keypoint, x-bin) logits.
    with b.block("head.simcc_x"):
        gx = b.mean(feat, dim=(2, 3), keepdim=True, _scope="pool")
        gx = b.flatten(gx, start_dim=1, _scope="flatten")
        lx = b.linear(gx, num_keypoints * x_bins, scope="fc")
        simcc_x = b.reshape(lx, shape=(batch, num_keypoints, x_bins), _scope="logits")

    # SimCC y head: global max over the spatial map (a different reduction from
    # the x head, so the two branches carry different data flow).
    with b.block("head.simcc_y"):
        gy = b.amax(feat, dim=(2, 3), keepdim=True, _scope="pool")
        gy = b.flatten(gy, start_dim=1, _scope="flatten")
        ly = b.linear(gy, num_keypoints * y_bins, scope="fc")
        simcc_y = b.reshape(ly, shape=(batch, num_keypoints, y_bins), _scope="logits")

    outputs: List[str] = [simcc_x, simcc_y]
    names: List[str] = ["simcc_x", "simcc_y"]
    if cfg.get("heatmap", True):
        with b.block("head.heatmap"):
            hm = b.conv_block(feat, num_keypoints, kernel_size=1, stride=1, act="",
                              norm="", scope="conv")
            hm = b.interpolate(hm, size=(image[2], image[3]), mode="bilinear",
                               _scope="upsample")
        outputs.append(hm)
        names.append("heatmap")
    b.output(outputs, names=names)
    return b.build()


# ---------------------------------------------------------------------------
# PARSeq
# ---------------------------------------------------------------------------

def _parseq_encoder_block(b: GraphBuilder, x: str, dim: int, heads: int,
                          mlp_hidden: int, scope: str) -> str:
    """Pre-norm ViT block: self-attention residual + MLP residual."""
    with b.block(scope):
        normed = b.layernorm(x, normalized_shape=(dim,), eps=1e-6, _scope="attn.norm")
        x = b.add(x, _multi_head_attention(b, normed, normed, dim, heads, "attn"),
                  _scope="attn_residual")
        normed = b.layernorm(x, normalized_shape=(dim,), eps=1e-6, _scope="mlp.norm")
        h = b.linear(normed, mlp_hidden, act="gelu", scope="mlp.fc1")
        h = b.linear(h, dim, scope="mlp.fc2")
        return b.add(x, h, _scope="mlp_residual")


def _parseq_decoder_block(b: GraphBuilder, x: str, memory: str, dim: int, heads: int,
                          mlp_hidden: int, scope: str) -> str:
    """Pre-norm decoder block: self-attention -> cross-attention -> MLP.

    The learned queries attend to each other non-causally (PARSeq decodes all
    characters in parallel); only the cross-attention reads the encoder memory.
    """
    with b.block(scope):
        normed = b.layernorm(x, normalized_shape=(dim,), eps=1e-6, _scope="self.norm")
        x = b.add(x, _multi_head_attention(b, normed, normed, dim, heads, "self"),
                  _scope="self_residual")
        normed = b.layernorm(x, normalized_shape=(dim,), eps=1e-6, _scope="cross.norm")
        x = b.add(x, _multi_head_attention(b, normed, memory, dim, heads, "cross"),
                  _scope="cross_residual")
        normed = b.layernorm(x, normalized_shape=(dim,), eps=1e-6, _scope="mlp.norm")
        h = b.linear(normed, mlp_hidden, act="gelu", scope="mlp.fc1")
        h = b.linear(h, dim, scope="mlp.fc2")
        return b.add(x, h, _scope="mlp_residual")


def build_parseq(cfg: Dict[str, Any] = PARSEQ) -> Any:
    """Build the (compact by default) PARSeq graph."""
    image: Tuple[int, ...] = tuple(int(v) for v in cfg["image"])
    patch = int(cfg["patch_size"])
    dim = int(cfg["hidden"])
    heads = int(cfg["num_heads"])
    mlp_hidden = int(cfg["mlp_hidden"])
    batch = int(image[0])
    b = GraphBuilder("parseq")
    x = b.input("image", image, dtype="float32")

    # -- ViT-style patch embedding ----------------------------------------
    with b.block("patch_embed"):
        p = b.conv_block(x, dim, kernel_size=patch, stride=patch, padding=0, act="",
                         norm="", scope="proj")
        p = b.flatten(p, start_dim=2, _scope="flatten")
        p = b.permute(p, dims=(0, 2, 1), _scope="tokens")
        _, tokens, _ = _shape_of(b, p)
        pos = b.param((1, tokens, dim), init="truncated_normal", name="pos_embed")
        memory = b.add(p, pos, _scope="add_positions")

    # -- encoder ----------------------------------------------------------
    for i in range(int(cfg["encoder_layers"])):
        memory = _parseq_encoder_block(b, memory, dim, heads, mlp_hidden,
                                       f"encoder.layers.{i}")
    memory = b.layernorm(memory, normalized_shape=(dim,), eps=1e-6, _scope="encoder.norm")

    # -- positional-query decoder -----------------------------------------
    with b.block("decoder"):
        queries = b.param((1, int(cfg["num_queries"]), dim), init="truncated_normal",
                          name="query_embed")
        if cfg.get("query_pos", True):
            query_pos = b.param((1, int(cfg["num_queries"]), dim), init="normal",
                                name="query_pos")
            queries = b.add(queries, query_pos, _scope="add_query_positions")
        for i in range(int(cfg["decoder_layers"])):
            queries = _parseq_decoder_block(b, queries, memory, dim, heads, mlp_hidden,
                                            f"layers.{i}")
        queries = b.layernorm(queries, normalized_shape=(dim,), eps=1e-6, _scope="norm")
    logits = b.linear(queries, int(cfg["vocab_size"]), scope="head.fc")
    b.output([logits], names=["logits"])
    return b.build()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def seed_efficientad_s() -> SeedModel:
    cfg = EFFICIENTAD_S
    image = tuple(int(v) for v in cfg["image"])
    return SeedModel(
        key="efficientad_s",
        task="Industrial Anomaly Detection",
        builder=build_efficientad_s,
        input_shapes={"image": image},
        input_dtypes={"image": "float32"},
        param_count=8057856,
        input_size="256x256",
        reference="EfficientAD (Batzner et al. 2024)",
        notes=f"compact graph: {image[2]}x{image[3]} input, channels {tuple(cfg['channels'])}; "
              f"paper: 256x256 input, 8,057,856 params",
    )


def seed_rtmpose_m() -> SeedModel:
    cfg = RTMPOSE_M
    image = tuple(int(v) for v in cfg["image"])
    return SeedModel(
        key="rtmpose_m",
        task="Keypoint Detection",
        builder=build_rtmpose_m,
        input_shapes={"image": image},
        input_dtypes={"image": "float32"},
        param_count=13587611,
        input_size="256x192",
        reference="RTMPose (Jiang et al. 2023)",
        notes=f"compact graph: {image[2]}x{image[3]} input, stages "
              f"{tuple(cfg['stage_channels'])}, {cfg['num_keypoints']} keypoints; "
              f"paper: 256x192 input, 17 keypoints, 13,587,611 params",
    )


def seed_parseq() -> SeedModel:
    cfg = PARSEQ
    image = tuple(int(v) for v in cfg["image"])
    return SeedModel(
        key="parseq",
        task="Scene Text Recognition",
        builder=build_parseq,
        input_shapes={"image": image},
        input_dtypes={"image": "float32"},
        param_count=23832671,
        input_size="32x128",
        reference="PARSeq (Bautista & Atienza 2022)",
        notes=f"compact graph: {image[2]}x{image[3]} input, hidden {cfg['hidden']}, "
              f"{cfg['encoder_layers']} encoder / {cfg['decoder_layers']} decoder layers, "
              f"{cfg['num_queries']} queries; paper: 94-char vocabulary, 23,832,671 params",
    )


MODELS = [
    seed_efficientad_s(),
    seed_rtmpose_m(),
    seed_parseq(),
]

for _m in MODELS:
    register_seed(_m)
