"""Text seed models: ModernBERT-base and Qwen2.5-0.5B.

Two text architectures, each defined once in the framework-neutral IR:

* **ModernBERT-base** (text classification) -- token embedding + learned
  positional embedding, a stack of pre-norm transformer encoder layers
  (LayerNorm -> explicitly built multi-head self-attention -> output projection,
  residual) alternating local (windowed) and global attention, a GeGLU MLP with
  its own residual, a final LayerNorm, mean pooling and a linear classifier.
* **Qwen2.5-0.5B** (text generation) -- a decoder-only transformer language
  model: token embedding, additive positional embedding (the paper's RoPE is
  approximated here because the IR has no rotary operator), ``num_layers`` decoder
  blocks with RMSNorm, grouped-query attention (``num_heads`` query heads against
  ``num_kv_heads`` key/value heads, with the head expansion written out
  explicitly), a SwiGLU MLP, and a final RMSNorm + vocabulary projection.

The attention of both models is built by hand from ``linear`` / ``reshape`` /
``permute`` / ``matmul`` / ``softmax`` / ``matmul`` (never ``sdpa``) so that the
data flow is fully exposed to the mutation operators.

Documented (paper-scale) hyper-parameters live in :data:`CONFIGS` under
``"full"``; the default ``"compact"`` variant shrinks the vocabulary, the
sequence length, the width and the depth so that smoke tests execute on CPU.
``SeedModel`` always reports the *documented* model's parameter count from the
paper's benchmark table.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional

from flowmut.ir.graph import GraphBuilder
from flowmut.seed_models.registry import SeedModel, register_seed

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
# ModernBERT-base (Warner et al. 2024):
#   vocab 50368, hidden 768, 22 layers, 12 heads, GeGLU intermediate 1152,
#   max sequence 8192 (512 used here), global attention every third layer and
#   local sliding-window attention (128) elsewhere, no bias in the linear layers.
#   Documented benchmark: 1x512 tokens, 149,606,402 parameters.
#
# Qwen2.5-0.5B (Qwen Team 2024):
#   vocab 151936, hidden 896, 24 layers, 14 query heads / 2 KV heads
#   (head dim 64), SwiGLU intermediate 4864, RMSNorm eps 1e-6, RoPE theta 1e6,
#   tied input/output embeddings, max sequence 32768 (512 used here).
#   Documented benchmark: 1x512 tokens, 494,032,768 parameters.
CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "compact": {
        "modernbert": dict(
            vocab_size=2048,
            max_seq_len=128,
            hidden=64,
            num_layers=2,
            num_heads=4,
            mlp_hidden=128,       # GeGLU gate/up width (2 x mlp_hidden activations)
            local_window=32,      # sliding-window size of the local layers
            local_every=2,        # every local_every-th layer is global
            num_classes=2,
        ),
        "qwen2_5": dict(
            vocab_size=2048,
            max_seq_len=128,
            hidden=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate=128,
        ),
    },
    "full": {
        "modernbert": dict(
            vocab_size=50368,
            max_seq_len=512,
            hidden=768,
            num_layers=22,
            num_heads=12,
            mlp_hidden=1152,
            local_window=128,
            local_every=3,
            num_classes=2,
        ),
        "qwen2_5": dict(
            vocab_size=151936,
            max_seq_len=512,
            hidden=896,
            num_layers=24,
            num_heads=14,
            num_kv_heads=2,
            intermediate=4864,
        ),
    },
}

DEFAULT_CONFIG = "compact"

MODERNBERT = CONFIGS[DEFAULT_CONFIG]["modernbert"]
QWEN2_5 = CONFIGS[DEFAULT_CONFIG]["qwen2_5"]


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _padding_indicator(b: GraphBuilder, attention_mask: str, scope: str) -> str:
    """``1 - attention_mask`` reshaped to ``(B, 1, 1, L)`` for ``masked_fill``.

    The graph carries no constants, so the "one" used by the inversion is a
    (broadcastable) parameter initialised to ones.
    """
    one = b.param((1,), init="ones", name=f"{scope}.one")
    inv = b.sub(one, attention_mask, _scope=f"{scope}.invert")
    inv = b.unsqueeze(inv, dim=1, _scope=f"{scope}.rows")
    return b.unsqueeze(inv, dim=1, _scope=f"{scope}.keys")


def _future_mask(b: GraphBuilder, seq_len: int, scope: str = "causal_mask") -> str:
    """Upper-triangular ``(1, 1, L, L)`` future mask built from pad + concat.

    ``U_1 = [[0]]`` and ``U_2n = [[U_n, 1], [0, U_n]]``, so each doubling step
    only needs ``tile``/``concat``.  ``seq_len`` must be a power of two.
    """
    if seq_len <= 0 or (seq_len & (seq_len - 1)):
        raise ValueError(f"causal mask construction needs a power-of-two length, got {seq_len}")
    with b.block(scope):
        ones = b.param((1, 1, 1, 1), init="ones", name=f"{scope}.ones")
        zeros = b.param((1, 1, 1, 1), init="zeros", name=f"{scope}.zeros")
        cur = b.param((1, 1, 1, 1), init="zeros", name=f"{scope}.base")
        size = 1
        while size < seq_len:
            on = b.tile(ones, reps=(1, 1, size, size), _scope=f"{scope}.ones{size}")
            ze = b.tile(zeros, reps=(1, 1, size, size), _scope=f"{scope}.zeros{size}")
            top = b.concat(cur, on, dim=-1, _scope=f"{scope}.top{size}")
            bot = b.concat(ze, cur, dim=-1, _scope=f"{scope}.bottom{size}")
            cur = b.concat(top, bot, dim=-2, _scope=f"{scope}.merge{size}")
            size *= 2
    return cur


def _split_heads(b: GraphBuilder, x: str, batch: int, seq: int, heads: int,
                 head_dim: int, scope: str) -> str:
    h = b.reshape(x, shape=(batch, seq, heads, head_dim), _scope=f"{scope}.reshape")
    return b.permute(h, dims=(0, 2, 1, 3), _scope=f"{scope}.permute")


def _merge_heads(b: GraphBuilder, x: str, batch: int, seq: int, dim: int,
                 scope: str) -> str:
    h = b.permute(x, dims=(0, 2, 1, 3), _scope=f"{scope}.permute")
    return b.reshape(h, shape=(batch, seq, dim), _scope=f"{scope}.reshape")


# ---------------------------------------------------------------------------
# ModernBERT-base
# ---------------------------------------------------------------------------

def _modernbert_attention(b: GraphBuilder, x: str, mask: str, cfg: Dict[str, Any],
                          batch: int, seq: int, scope: str,
                          window: Optional[int] = None) -> str:
    """Multi-head self-attention, global (4-D) or windowed (5-D).

    Global attention reshapes to ``(B, H, L, E)``; local attention additionally
    folds the sequence into ``seq // window`` non-overlapping windows of
    ``window`` tokens so that attention only ever mixes tokens inside a window.
    """
    dim = int(cfg["hidden"])
    heads = int(cfg["num_heads"])
    head_dim = dim // heads
    with b.block(scope):
        q = b.linear(x, dim, scope="q")
        k = b.linear(x, dim, scope="k")
        v = b.linear(x, dim, scope="v")
        if window:
            nwin = seq // window
            # Fold the windows into the batch axis so that attention mixes only
            # tokens inside one window: (B, L, D) -> (B*nwin, H, window, E).
            q = b.reshape(q, shape=(batch, nwin, window, heads, head_dim), _scope="q.grid")
            q = b.permute(q, dims=(0, 1, 3, 2, 4), _scope="q.heads")
            q = b.reshape(q, shape=(batch * nwin, heads, window, head_dim), _scope="q.window")
            k = b.reshape(k, shape=(batch, nwin, window, heads, head_dim), _scope="k.grid")
            k = b.permute(k, dims=(0, 1, 3, 2, 4), _scope="k.heads")
            k = b.reshape(k, shape=(batch * nwin, heads, window, head_dim), _scope="k.window")
            v = b.reshape(v, shape=(batch, nwin, window, heads, head_dim), _scope="v.grid")
            v = b.permute(v, dims=(0, 1, 3, 2, 4), _scope="v.heads")
            v = b.reshape(v, shape=(batch * nwin, heads, window, head_dim), _scope="v.window")
            kt = b.permute(k, dims=(0, 1, 3, 2), _scope="k.transpose")
            logits = b.matmul(q, kt, _scope="logits")
            logits = b.mul(logits, other=1.0 / (head_dim ** 0.5), _scope="scale")
            win_mask = b.reshape(mask, shape=(batch, 1, nwin, window), _scope="mask.windows")
            win_mask = b.reshape(win_mask, shape=(batch * nwin, 1, window), _scope="mask.batch")
            win_mask = b.unsqueeze(win_mask, dim=1, _scope="mask.heads")
            logits = b.masked_fill(logits, win_mask, value=-1e9, _scope="mask")
            probs = b.softmax(logits, dim=-1, _scope="probs")
            ctx = b.matmul(probs, v, _scope="ctx")
            ctx = b.permute(ctx, dims=(0, 2, 1, 3), _scope="ctx.merge")
            ctx = b.reshape(ctx, shape=(batch, nwin, window, heads, head_dim),
                            _scope="ctx.windows")
            ctx = b.reshape(ctx, shape=(batch, seq, dim), _scope="ctx.tokens")
        else:
            qh = _split_heads(b, q, batch, seq, heads, head_dim, "q")
            kh = _split_heads(b, k, batch, seq, heads, head_dim, "k")
            vh = _split_heads(b, v, batch, seq, heads, head_dim, "v")
            kt = b.permute(kh, dims=(0, 1, 3, 2), _scope="k.transpose")
            logits = b.matmul(qh, kt, _scope="logits")
            logits = b.mul(logits, other=1.0 / (head_dim ** 0.5), _scope="scale")
            logits = b.masked_fill(logits, mask, value=-1e9, _scope="mask")
            probs = b.softmax(logits, dim=-1, _scope="probs")
            ctx = b.matmul(probs, vh, _scope="ctx")
            ctx = _merge_heads(b, ctx, batch, seq, dim, "ctx")
        return b.linear(ctx, dim, scope="out")


def _modernbert_attn_branch(b: GraphBuilder, x: str, cfg: Dict[str, Any], mask: str,
                            batch: int, seq: int, window: Optional[int],
                            scope: str) -> str:
    normed = b.layernorm(x, normalized_shape=(int(cfg["hidden"]),), eps=1e-5,
                         _scope=f"{scope}.norm")
    return _modernbert_attention(b, normed, mask, cfg, batch, seq, scope, window=window)


def _modernbert_mlp_branch(b: GraphBuilder, x: str, cfg: Dict[str, Any],
                           scope: str) -> str:
    """GeGLU feed-forward: ``down(gelu(gate(x)) * up(x))``."""
    dim = int(cfg["hidden"])
    hidden = int(cfg["mlp_hidden"])
    with b.block(scope):
        normed = b.layernorm(x, normalized_shape=(dim,), eps=1e-5, _scope="norm")
        gate = b.gelu(b.linear(normed, hidden, scope="gate"), _scope="gate.act")
        up = b.linear(normed, hidden, scope="up")
        return b.linear(b.mul(gate, up, _scope="mul"), dim, scope="down")


def build_modernbert(cfg: Dict[str, Any] = MODERNBERT) -> Any:
    """Build the (compact by default) ModernBERT-base graph."""
    vocab = int(cfg["vocab_size"])
    seq = int(cfg["max_seq_len"])
    dim = int(cfg["hidden"])
    batch = 1
    b = GraphBuilder("modernbert_base")
    ids = b.input("input_ids", (batch, seq), dtype="int64")
    attention_mask = b.input("attention_mask", (batch, seq), dtype="float32")

    with b.block("embeddings"):
        tok = b.embedding(ids, b.param((vocab, dim), init="normal", name="tok_emb"),
                          dtype="float32", _scope="token")
        pos = b.param((1, seq, dim), init="truncated_normal", name="pos_emb")
        h = b.add(tok, pos, _scope="add_positions")
        h = b.layernorm(h, normalized_shape=(dim,), eps=1e-5, _scope="norm")

    mask = _padding_indicator(b, attention_mask, "attention_mask")

    local_every = int(cfg["local_every"])
    window = int(cfg["local_window"])
    for i in range(int(cfg["num_layers"])):
        # ModernBERT interleaves global attention with sliding-window attention;
        # the last layer of every group of ``local_every`` is global.
        win = window if (i % local_every) != (local_every - 1) else None
        h = b.residual(
            h,
            partial(_modernbert_attn_branch, b, cfg=cfg, mask=mask, batch=batch,
                    seq=seq, window=win, scope=f"layers.{i}.attn"),
            scope=f"layers.{i}.attn_residual",
        )
        h = b.residual(
            h,
            partial(_modernbert_mlp_branch, b, cfg=cfg, scope=f"layers.{i}.mlp"),
            scope=f"layers.{i}.mlp_residual",
        )

    with b.block("head"):
        h = b.layernorm(h, normalized_shape=(dim,), eps=1e-5, _scope="norm")
        pooled = b.mean(h, dim=1, keepdim=False, _scope="pool")
        logits = b.linear(pooled, int(cfg["num_classes"]), scope="classifier")
    b.output([logits], names=["logits"])
    return b.build()


# ---------------------------------------------------------------------------
# Qwen2.5-0.5B
# ---------------------------------------------------------------------------

def _qwen_attention(b: GraphBuilder, x: str, cfg: Dict[str, Any], mask: str,
                    batch: int, seq: int, scope: str) -> str:
    """Grouped-query attention with an explicit query/key/value head expansion."""
    dim = int(cfg["hidden"])
    heads = int(cfg["num_heads"])
    kv_heads = int(cfg["num_kv_heads"])
    head_dim = dim // heads
    kv_dim = kv_heads * head_dim
    groups = heads // kv_heads
    with b.block(scope):
        q = b.linear(x, dim, scope="q")
        k = b.linear(x, kv_dim, scope="k")
        v = b.linear(x, kv_dim, scope="v")
        qh = _split_heads(b, q, batch, seq, heads, head_dim, "q")
        kh = _split_heads(b, k, batch, seq, kv_heads, head_dim, "k")
        vh = _split_heads(b, v, batch, seq, kv_heads, head_dim, "v")
        if groups > 1:
            # (B, KV, L, E) -> (B, KV, 1, L, E) -> (B, KV, groups, L, E)
            #                -> (B, H, L, E) with each KV head repeated `groups` times.
            kh = b.unsqueeze(kh, dim=2, _scope="k.group")
            kh = b.expand(kh, shape=(batch, kv_heads, groups, seq, head_dim),
                          _scope="k.expand")
            kh = b.reshape(kh, shape=(batch, heads, seq, head_dim), _scope="k.repeat")
            vh = b.unsqueeze(vh, dim=2, _scope="v.group")
            vh = b.expand(vh, shape=(batch, kv_heads, groups, seq, head_dim),
                          _scope="v.expand")
            vh = b.reshape(vh, shape=(batch, heads, seq, head_dim), _scope="v.repeat")
        kt = b.permute(kh, dims=(0, 1, 3, 2), _scope="k.transpose")
        logits = b.matmul(qh, kt, _scope="logits")
        logits = b.mul(logits, other=1.0 / (head_dim ** 0.5), _scope="scale")
        logits = b.masked_fill(logits, mask, value=-1e9, _scope="mask")
        probs = b.softmax(logits, dim=-1, _scope="probs")
        ctx = b.matmul(probs, vh, _scope="ctx")
        ctx = _merge_heads(b, ctx, batch, seq, dim, "ctx")
        return b.linear(ctx, dim, scope="out")


def _qwen_attn_branch(b: GraphBuilder, x: str, cfg: Dict[str, Any], mask: str,
                      batch: int, seq: int, scope: str) -> str:
    normed = b.rmsnorm(x, normalized_shape=(int(cfg["hidden"]),), eps=1e-6,
                       _scope=f"{scope}.norm")
    return _qwen_attention(b, normed, cfg, mask, batch, seq, scope)


def _qwen_mlp_branch(b: GraphBuilder, x: str, cfg: Dict[str, Any], scope: str) -> str:
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``."""
    dim = int(cfg["hidden"])
    inter = int(cfg["intermediate"])
    with b.block(scope):
        normed = b.rmsnorm(x, normalized_shape=(dim,), eps=1e-6, _scope="norm")
        gate = b.silu(b.linear(normed, inter, scope="gate"), _scope="gate.act")
        up = b.linear(normed, inter, scope="up")
        return b.linear(b.mul(gate, up, _scope="mul"), dim, scope="down")


def build_qwen2_5_0_5b(cfg: Dict[str, Any] = QWEN2_5) -> Any:
    """Build the (compact by default) Qwen2.5-0.5B graph."""
    vocab = int(cfg["vocab_size"])
    seq = int(cfg["max_seq_len"])
    dim = int(cfg["hidden"])
    batch = 1
    b = GraphBuilder("qwen2_5_0_5b")
    ids = b.input("input_ids", (batch, seq), dtype="int64")
    attention_mask = b.input("attention_mask", (batch, seq), dtype="float32")

    with b.block("embeddings"):
        tok = b.embedding(ids, b.param((vocab, dim), init="normal", name="tok_emb"),
                          dtype="float32", _scope="token")
        pos = b.param((1, seq, dim), init="truncated_normal", name="pos_emb")
        h = b.add(tok, pos, _scope="add_positions")

    # Additive positional embedding stands in for RoPE; the causal + padding
    # mask is the union of the future mask and the inverted attention mask.
    pad = _padding_indicator(b, attention_mask, "attention_mask")
    causal = _future_mask(b, seq, "causal_mask")
    mask = b.maximum(pad, causal, _scope="attention_mask.merge")

    for i in range(int(cfg["num_layers"])):
        h = b.residual(
            h,
            partial(_qwen_attn_branch, b, cfg=cfg, mask=mask, batch=batch, seq=seq,
                    scope=f"layers.{i}.attn"),
            scope=f"layers.{i}.attn_residual",
        )
        h = b.residual(
            h,
            partial(_qwen_mlp_branch, b, cfg=cfg, scope=f"layers.{i}.mlp"),
            scope=f"layers.{i}.mlp_residual",
        )

    h = b.rmsnorm(h, normalized_shape=(dim,), eps=1e-6, _scope="final_norm")
    logits = b.linear(h, vocab, bias=False, scope="lm_head")
    b.output([logits], names=["logits"])
    return b.build()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def seed_modernbert() -> SeedModel:
    cfg = MODERNBERT
    return SeedModel(
        key="modernbert_base",
        task="Text Classification",
        builder=build_modernbert,
        input_shapes={"input_ids": (1, int(cfg["max_seq_len"])),
                      "attention_mask": (1, int(cfg["max_seq_len"]))},
        input_dtypes={"input_ids": "int64", "attention_mask": "float32"},
        input_ranges={"input_ids": (0, int(cfg["vocab_size"]))},
        param_count=149606402,
        input_size="1x512",
        reference="ModernBERT (Warner et al. 2024)",
        notes=f"compact graph: 1x{cfg['max_seq_len']} tokens, hidden {cfg['hidden']}, "
              f"{cfg['num_layers']} layers; paper: 1x512 tokens, 149,606,402 params",
    )


def seed_qwen2_5_0_5b() -> SeedModel:
    cfg = QWEN2_5
    return SeedModel(
        key="qwen2_5_0_5b",
        task="Text Generation",
        builder=build_qwen2_5_0_5b,
        input_shapes={"input_ids": (1, int(cfg["max_seq_len"])),
                      "attention_mask": (1, int(cfg["max_seq_len"]))},
        input_dtypes={"input_ids": "int64", "attention_mask": "float32"},
        input_ranges={"input_ids": (0, int(cfg["vocab_size"]))},
        param_count=494032768,
        input_size="1x512",
        reference="Qwen2.5 (Qwen Team 2024)",
        notes=f"compact graph: 1x{cfg['max_seq_len']} tokens, hidden {cfg['hidden']}, "
              f"{cfg['num_layers']} layers, GQA {cfg['num_heads']}/{cfg['num_kv_heads']}; "
              f"paper: 1x512 tokens, 494,032,768 params",
    )


MODELS = [
    seed_modernbert(),
    seed_qwen2_5_0_5b(),
]

for _m in MODELS:
    register_seed(_m)
