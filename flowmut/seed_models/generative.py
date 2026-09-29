"""Generative seed models: DiT-S/2 and SwinIR-M (x4).

* **DiT-S/2** (class-conditional image generation) -- a 2x2 strided-conv
  patchifier, a positional embedding, a conditioning vector that sums a
  timestep MLP with a class-label embedding, a stack of DiT blocks with
  adaLN-Zero modulation (one linear projection split into six shift/scale/gate
  vectors, each applied with ``mul``/``add`` and the residual branches gated by
  ``mul``), hand-written multi-head self-attention, a final LayerNorm and a
  linear layer that predicts ``patch_size^2 * out_channels`` values per token,
  unpatchified back to the image grid.
* **SwinIR-M (x4)** (image restoration / super-resolution) -- a shallow feature
  conv, a stack of residual Swin Transformer blocks whose window attention is
  built explicitly from ``reshape``/``permute``/``matmul``/``softmax`` over
  non-overlapping windows (plus a learned relative position bias), a conv layer
  with a long skip ``add``, and a 4x ``pixel_shuffle`` upsampler with a final
  conv back to the image.

Documented (paper-scale) hyper-parameters live in :data:`CONFIGS` under
``"full"``; the default ``"compact"`` variant shrinks the input resolution,
width and depth so that smoke tests execute on CPU.  ``SeedModel`` always
reports the *documented* model's parameter count from the paper's benchmark
table.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from flowmut.ir.graph import GraphBuilder
from flowmut.seed_models.registry import SeedModel, register_seed

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
# DiT-S/2 (Peebles & Xie 2023):
#   patch size 2, hidden 384, 12 DiT blocks, 6 heads, MLP ratio 4, adaLN-Zero
#   modulation, in/out channels 3, 256x256 input, 32,963,360 parameters.
#
# SwinIR-M (x4) (Liang et al. 2021):
#   embed dim 180, 6 residual Swin Transformer blocks (6 heads, window 8,
#   MLP ratio 2), 4x pixel-shuffle upsampler, 256x256 LR input, 11,900,199
#   parameters.
CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "compact": {
        "dit_s_2": dict(
            image=(1, 3, 32, 32),
            patch_size=2,
            hidden=64,
            num_layers=2,
            num_heads=4,
            mlp_hidden=256,
            out_channels=3,
            num_classes=64,
        ),
        "swinir_m_x4": dict(
            image=(1, 3, 32, 32),
            embed_dim=32,
            num_blocks=2,
            num_stls=1,
            num_heads=2,
            window=4,
            mlp_hidden=64,
            upscale=4,
            out_channels=3,
        ),
    },
    "full": {
        "dit_s_2": dict(
            image=(1, 3, 256, 256),
            patch_size=2,
            hidden=384,
            num_layers=12,
            num_heads=6,
            mlp_hidden=1536,
            out_channels=3,
            num_classes=1000,
        ),
        "swinir_m_x4": dict(
            image=(1, 3, 256, 256),
            embed_dim=180,
            num_blocks=6,
            num_stls=2,
            num_heads=6,
            window=8,
            mlp_hidden=360,
            upscale=4,
            out_channels=3,
        ),
    },
}

DEFAULT_CONFIG = "compact"

DIT_S_2 = CONFIGS[DEFAULT_CONFIG]["dit_s_2"]
SWINIR_M_X4 = CONFIGS[DEFAULT_CONFIG]["swinir_m_x4"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _shape_of(b: GraphBuilder, edge: str) -> Tuple[int, ...]:
    return tuple(int(v) for v in b.graph.edges[edge].spec.shape)


def _channels_of(b: GraphBuilder, edge: str) -> int:
    return int(_shape_of(b, edge)[1])


def _split_heads(b: GraphBuilder, x: str, batch: int, seq: int, heads: int,
                 head_dim: int, scope: str) -> str:
    h = b.reshape(x, shape=(batch, seq, heads, head_dim), _scope=f"{scope}.reshape")
    return b.permute(h, dims=(0, 2, 1, 3), _scope=f"{scope}.permute")


def _merge_heads(b: GraphBuilder, x: str, batch: int, seq: int, dim: int,
                 scope: str) -> str:
    h = b.permute(x, dims=(0, 2, 1, 3), _scope=f"{scope}.permute")
    return b.reshape(h, shape=(batch, seq, dim), _scope=f"{scope}.reshape")


def _multi_head_attention(b: GraphBuilder, x: str, dim: int, heads: int,
                          scope: str) -> str:
    """Self-attention over a rank-3 ``(batch, tokens, dim)`` sequence."""
    head_dim = dim // heads
    batch, tokens, _ = _shape_of(b, x)
    with b.block(scope):
        q = b.linear(x, dim, scope="q")
        k = b.linear(x, dim, scope="k")
        v = b.linear(x, dim, scope="v")
        qh = _split_heads(b, q, batch, tokens, heads, head_dim, "q")
        kh = _split_heads(b, k, batch, tokens, heads, head_dim, "k")
        vh = _split_heads(b, v, batch, tokens, heads, head_dim, "v")
        kt = b.permute(kh, dims=(0, 1, 3, 2), _scope="k.transpose")
        logits = b.matmul(qh, kt, _scope="logits")
        logits = b.mul(logits, other=1.0 / (head_dim ** 0.5), _scope="scale")
        probs = b.softmax(logits, dim=-1, _scope="probs")
        ctx = b.matmul(probs, vh, _scope="ctx")
        ctx = _merge_heads(b, ctx, batch, tokens, dim, "ctx")
        return b.linear(ctx, dim, scope="out")


# ---------------------------------------------------------------------------
# DiT-S/2
# ---------------------------------------------------------------------------

def _dit_block(b: GraphBuilder, x: str, cond: str, cfg: Dict[str, Any],
               scope: str) -> str:
    """One DiT block with adaLN-Zero modulation and gated residual branches."""
    dim = int(cfg["hidden"])
    heads = int(cfg["num_heads"])
    mlp_hidden = int(cfg["mlp_hidden"])
    with b.block(scope):
        # adaLN-Zero: one projection per block emits six modulation vectors.
        # (The authors zero-initialise this projection; the IR records the
        # structure and leaves the initialiser to the framework adapter.)
        mod = b.linear(cond, 6 * dim, scope="adaLN")
        parts = b.split(mod, dim=-1, num_outputs=6, _num_outputs=6, _scope="modulation")
        shift_msa = b.unsqueeze(parts[0], dim=1, _scope="shift_msa")
        scale_msa = b.unsqueeze(parts[1], dim=1, _scope="scale_msa")
        gate_msa = b.unsqueeze(parts[2], dim=1, _scope="gate_msa")
        shift_mlp = b.unsqueeze(parts[3], dim=1, _scope="shift_mlp")
        scale_mlp = b.unsqueeze(parts[4], dim=1, _scope="scale_mlp")
        gate_mlp = b.unsqueeze(parts[5], dim=1, _scope="gate_mlp")

        normed = b.layernorm(x, normalized_shape=(dim,), eps=1e-6, _scope="attn.norm")
        normed = b.mul(normed, scale_msa, _scope="attn.scale")
        normed = b.add(normed, shift_msa, _scope="attn.modulate")
        attn = _multi_head_attention(b, normed, dim, heads, "attn")
        x = b.add(x, b.mul(attn, gate_msa, _scope="attn.gate"), _scope="attn_residual")

        normed = b.layernorm(x, normalized_shape=(dim,), eps=1e-6, _scope="mlp.norm")
        normed = b.add(b.mul(normed, scale_mlp, _scope="mlp.scale"), shift_mlp,
                       _scope="mlp.modulate")
        h = b.gelu(b.linear(normed, mlp_hidden, scope="mlp.fc1"), _scope="mlp.act")
        h = b.linear(h, dim, scope="mlp.fc2")
        x = b.add(x, b.mul(h, gate_mlp, _scope="mlp.gate"), _scope="mlp_residual")
    return x


def build_dit_s_2(cfg: Dict[str, Any] = DIT_S_2) -> Any:
    """Build the (compact by default) DiT-S/2 graph."""
    image: Tuple[int, ...] = tuple(int(v) for v in cfg["image"])
    patch = int(cfg["patch_size"])
    dim = int(cfg["hidden"])
    heads = int(cfg["num_heads"])
    out_channels = int(cfg["out_channels"])
    batch = int(image[0])
    grid_h, grid_w = image[2] // patch, image[3] // patch
    tokens = grid_h * grid_w
    b = GraphBuilder("dit_s_2")
    x = b.input("image", image, dtype="float32")
    timestep = b.input("timestep", (batch, 1), dtype="float32")
    class_labels = b.input("class_labels", (batch,), dtype="int64")

    # -- patchify + positional embedding ----------------------------------
    with b.block("patch_embed"):
        p = b.conv_block(x, dim, kernel_size=patch, stride=patch, padding=0, act="",
                         norm="", scope="proj")
        p = b.flatten(p, start_dim=2, _scope="flatten")
        p = b.permute(p, dims=(0, 2, 1), _scope="tokens")
        pos = b.param((1, tokens, dim), init="truncated_normal", name="pos_embed")
        h = b.add(p, pos, _scope="add_positions")

    # -- timestep + class conditioning ------------------------------------
    with b.block("cond_embed"):
        cond = b.linear(timestep, dim, scope="mlp.0")
        cond = b.silu(cond, _scope="mlp.1")
        cond = b.linear(cond, dim, scope="mlp.2")
        class_table = b.param((int(cfg["num_classes"]), dim), init="normal",
                              name="class_embed")
        class_embed = b.embedding(class_labels, class_table, dtype="float32",
                                  _scope="class_lookup")
        cond = b.add(cond, class_embed, _scope="add_class")

    for i in range(int(cfg["num_layers"])):
        h = _dit_block(b, h, cond, cfg, f"blocks.{i}")

    # -- final adaLN + linear + unpatchify --------------------------------
    with b.block("final_layer"):
        mod = b.linear(cond, 2 * dim, scope="adaLN")
        parts = b.split(mod, dim=-1, num_outputs=2, _num_outputs=2, _scope="modulation")
        shift = b.unsqueeze(parts[0], dim=1, _scope="shift")
        scale = b.unsqueeze(parts[1], dim=1, _scope="scale")
        h = b.layernorm(h, normalized_shape=(dim,), eps=1e-6, _scope="norm")
        h = b.add(b.mul(h, scale, _scope="scale_norm"), shift, _scope="modulate")
        h = b.linear(h, patch * patch * out_channels, scope="linear")
        h = b.reshape(h, shape=(batch, grid_h, grid_w, patch, patch, out_channels),
                      _scope="unpatchify.grid")
        h = b.permute(h, dims=(0, 5, 1, 3, 2, 4), _scope="unpatchify.merge")
        sample = b.reshape(h, shape=(batch, out_channels, grid_h * patch, grid_w * patch),
                           _scope="unpatchify.image")
    b.output([sample], names=["sample"])
    return b.build()


# ---------------------------------------------------------------------------
# SwinIR-M (x4)
# ---------------------------------------------------------------------------

def _channel_norm(b: GraphBuilder, x: str, scope: str) -> str:
    """LayerNorm over the channel axis of an NCHW tensor (channels-last)."""
    h = b.permute(x, dims=(0, 2, 3, 1), _scope=f"{scope}.to_nhwc")
    c = int(_shape_of(b, h)[-1])
    h = b.layernorm(h, normalized_shape=(c,), eps=1e-5, _scope=f"{scope}.norm")
    return b.permute(h, dims=(0, 3, 1, 2), _scope=f"{scope}.to_nchw")


def _window_attention(b: GraphBuilder, x: str, dim: int, heads: int, window: int,
                      scope: str) -> str:
    """Window multi-head self-attention (``(B, C, H, W)`` in, same shape out).

    The feature map is folded into ``nH*nW`` non-overlapping ``window x window``
    tiles and attention is computed inside each tile only.
    """
    batch, channels, height, width = _shape_of(b, x)
    grid_h, grid_w = height // window, width // window
    tiles = window * window
    nwin = batch * grid_h * grid_w
    head_dim = dim // heads
    with b.block(scope):
        h = b.reshape(x, shape=(batch, channels, grid_h, window, grid_w, window),
                      _scope="grid")
        h = b.permute(h, dims=(0, 2, 4, 3, 5, 1), _scope="windows")
        h = b.reshape(h, shape=(nwin, tiles, channels), _scope="tokens")
        q = b.linear(h, dim, scope="q")
        k = b.linear(h, dim, scope="k")
        v = b.linear(h, dim, scope="v")
        qh = _split_heads(b, q, nwin, tiles, heads, head_dim, "q")
        kh = _split_heads(b, k, nwin, tiles, heads, head_dim, "k")
        vh = _split_heads(b, v, nwin, tiles, heads, head_dim, "v")
        kt = b.permute(kh, dims=(0, 1, 3, 2), _scope="k.transpose")
        logits = b.matmul(qh, kt, _scope="logits")
        logits = b.mul(logits, other=1.0 / (head_dim ** 0.5), _scope="scale")
        # Learned relative position bias, shared by every window/tile.
        bias = b.param((1, heads, tiles, tiles), init="zeros", name=f"{scope}.rel_bias")
        logits = b.add(logits, bias, _scope="bias")
        probs = b.softmax(logits, dim=-1, _scope="probs")
        ctx = b.matmul(probs, vh, _scope="ctx")
        ctx = _merge_heads(b, ctx, nwin, tiles, dim, "ctx")
        ctx = b.linear(ctx, dim, scope="out")
        ctx = b.reshape(ctx, shape=(batch, grid_h, grid_w, window, window, dim),
                        _scope="unwindows")
        ctx = b.permute(ctx, dims=(0, 5, 1, 3, 2, 4), _scope="grid")
        return b.reshape(ctx, shape=(batch, channels, height, width), _scope="merge")


def _swinir_stl(b: GraphBuilder, x: str, cfg: Dict[str, Any], scope: str) -> str:
    """Swin Transformer layer: window-attention residual + channel MLP residual."""
    dim = int(cfg["embed_dim"])
    heads = int(cfg["num_heads"])
    window = int(cfg["window"])
    mlp_hidden = int(cfg["mlp_hidden"])
    with b.block(scope):
        normed = _channel_norm(b, x, "attn")
        x = b.add(x, _window_attention(b, normed, dim, heads, window, "attn"),
                  _scope="attn_residual")
        normed = b.permute(x, dims=(0, 2, 3, 1), _scope="mlp.to_nhwc")
        normed = b.layernorm(normed, normalized_shape=(dim,), eps=1e-5, _scope="mlp.norm")
        h = b.linear(normed, mlp_hidden, act="gelu", scope="mlp.fc1")
        h = b.linear(h, dim, scope="mlp.fc2")
        h = b.permute(h, dims=(0, 3, 1, 2), _scope="mlp.to_nchw")
        return b.add(x, h, _scope="mlp_residual")


def _swinir_rstb(b: GraphBuilder, x: str, cfg: Dict[str, Any], scope: str) -> str:
    """Residual Swin Transformer block: STLs followed by a conv residual."""
    dim = int(cfg["embed_dim"])
    with b.block(scope):
        h = x
        for i in range(int(cfg["num_stls"])):
            h = _swinir_stl(b, h, cfg, f"stl.{i}")
        h = b.conv_block(h, dim, kernel_size=3, stride=1, act="", norm="",
                         scope="conv")
        return b.add(x, h, _scope="conv_residual")


def build_swinir_m_x4(cfg: Dict[str, Any] = SWINIR_M_X4) -> Any:
    """Build the (compact by default) SwinIR-M x4 graph."""
    image: Tuple[int, ...] = tuple(int(v) for v in cfg["image"])
    dim = int(cfg["embed_dim"])
    upscale = int(cfg["upscale"])
    out_channels = int(cfg["out_channels"])
    window = int(cfg["window"])
    if image[2] % window or image[3] % window:
        raise ValueError("the input resolution must be divisible by the window size")
    b = GraphBuilder("swinir_m_x4")
    x = b.input("image", image, dtype="float32")

    shallow = b.conv_block(x, dim, kernel_size=3, stride=1, act="", norm="",
                           scope="conv_first")
    h = shallow
    for i in range(int(cfg["num_blocks"])):
        h = _swinir_rstb(b, h, cfg, f"layers.{i}")
    h = b.conv_block(h, dim, kernel_size=3, stride=1, act="", norm="",
                     scope="conv_after_body")
    h = b.add(shallow, h, _scope="long_skip")

    # 4x pixel-shuffle upsampler: project to dim * upscale^2 channels, shuffle,
    # then refine back to the image.
    h = b.conv_block(h, dim * upscale * upscale, kernel_size=3, stride=1, act="",
                     norm="", scope="upsample.project")
    h = b.pixel_shuffle(h, upscale_factor=upscale, _scope="upsample.shuffle")
    h = b.conv_block(h, dim, kernel_size=3, stride=1, act="gelu", norm="",
                     scope="upsample.refine")
    sr = b.conv_block(h, out_channels, kernel_size=3, stride=1, act="", norm="",
                      scope="conv_last")
    b.output([sr], names=["sr_image"])
    return b.build()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def seed_dit_s_2() -> SeedModel:
    cfg = DIT_S_2
    image = tuple(int(v) for v in cfg["image"])
    return SeedModel(
        key="dit_s_2",
        task="Image Generation",
        builder=build_dit_s_2,
        input_shapes={"image": image, "timestep": (1, 1), "class_labels": (1,)},
        input_dtypes={"image": "float32", "timestep": "float32",
                      "class_labels": "int64"},
        input_ranges={"class_labels": (0, int(cfg["num_classes"]))},
        param_count=32963360,
        input_size="256x256",
        reference="DiT (Peebles & Xie 2023)",
        notes=f"compact graph: {image[2]}x{image[3]} input, patch {cfg['patch_size']}, "
              f"hidden {cfg['hidden']}, {cfg['num_layers']} DiT blocks; "
              f"paper: 256x256 input, 32,963,360 params",
    )


def seed_swinir_m_x4() -> SeedModel:
    cfg = SWINIR_M_X4
    image = tuple(int(v) for v in cfg["image"])
    return SeedModel(
        key="swinir_m_x4",
        task="Image Restoration",
        builder=build_swinir_m_x4,
        input_shapes={"image": image},
        input_dtypes={"image": "float32"},
        param_count=11900199,
        input_size="256x256",
        reference="SwinIR (Liang et al. 2021)",
        notes=f"compact graph: {image[2]}x{image[3]} LR input, embed dim {cfg['embed_dim']}, "
              f"{cfg['num_blocks']} RSTB, window {cfg['window']}, x{cfg['upscale']} upsampler; "
              f"paper: 256x256 LR input, 11,900,199 params",
    )


MODELS = [
    seed_dit_s_2(),
    seed_swinir_m_x4(),
]

for _m in MODELS:
    register_seed(_m)
