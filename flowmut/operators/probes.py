"""Synthetic probe TFGs used to deduplicate the operator pool.

Section "Merging, Deduplication, and Validation" defines two operators as
duplicates when they have "the same target, edit, parameters, and applicability
conditions".  Those four properties are exactly the *behaviour* of the operator:
the sites it matches and the graph rewrite it performs there.

This module builds a small, framework-free suite of profiled TFGs that covers the
shapes of data flow the operator pool works on (convolution, normalisation,
activations, pooling, reshaping, attention, arithmetic, embeddings).  Matching
every operator against this suite yields an observable signature -- the set of
sites it matches plus the graph delta it applies -- which is used by
:func:`flowmut.operators.pool.find_duplicate_groups` to detect the CG/TFG
overlap.  Because the probes are pure IR plus fabricated ``F`` annotations, the
deduplication is deterministic and needs no framework adapter.
"""

from __future__ import annotations

from typing import Dict, List

from flowmut.ir.graph import Graph, GraphBuilder
from flowmut.ir.specs import TensorSpec, ValueProfile
from flowmut.tfg.tfg import TFG, build_tfg


def _profile(graph: Graph, scale: float = 1.0, grad: bool = True,
             alias: bool = False) -> Dict[str, List[TensorSpec]]:
    """Fabricate plausible ``F(e)`` annotations for a probe graph."""
    specs: Dict[str, List[TensorSpec]] = {}
    for eid, edge in graph.edges.items():
        node = graph.nodes[edge.producer]
        shape = tuple(node.out_shape) if node.out_shape else tuple(
            edge.spec.shape if edge.spec else (1, 8, 8, 8))
        dtype = edge.spec.dtype if edge.spec else "float32"
        if node.op == "param":
            dtype = "float32"
        base = TensorSpec(shape=shape, dtype=dtype, requires_grad=grad,
                          numel=1)
        profiled = base.with_(
            value=ValueProfile(min=-scale, max=scale, mean=0.0, std=scale,
                               abs_mean=scale, l2=scale,
                               zero_fraction=0.25 if alias else 0.1,
                               magnitude_bin=int(max(-8, min(8, scale)))),
            alias_of=("probe_buffer" if alias and node.op in ("view", "permute",
                                                              "reshape", "clone")
                      else None),
            is_view=node.op in ("view", "permute", "transpose", "reshape",
                                "squeeze", "unsqueeze", "expand", "getitem"),
        )
        # Two profiling inputs so that some edges are polymorphic.
        second = profiled.with_(shape=(max(1, shape[0] + 1),) + shape[1:]) \
            if shape else profiled
        specs[eid] = [profiled, second]
    return specs


def _conv_probe() -> TFG:
    b = GraphBuilder("probe_conv")
    x = b.input("images", (2, 3, 32, 32))
    with b.block("stage0"):
        h = b.conv_block(x, 16, 3, stride=2, scope="conv")
        h = b.conv_block(h, 16, 1, scope="pointwise")
        r = b.add(h, h, _scope="residual")
        h = b.maxpool2d(r, kernel_size=2, stride=2, _scope="pool")
    with b.block("stage1"):
        h = b.conv_block(h, 32, 3, scope="conv", groups=1)
        h = b.global_avgpool(h, keepdim=True, _scope="gap")
        h = b.flatten(h, start_dim=1, _scope="flatten")
    out = b.linear(h, 10, scope="head")
    b.output([out], names=["logits"])
    return build_tfg(b.build(), _profile(b.build()))


def _norm_attn_probe() -> TFG:
    b = GraphBuilder("probe_norm")
    x = b.input("tokens", (2, 16, 32), dtype="float32")
    h = b.layernorm(x, normalized_shape=(32,), _scope="ln")
    h = b.linear(h, 32, scope="qkv")
    q = b.reshape(h, shape=(2, 16, 4, 8), _scope="q")
    k = b.reshape(h, shape=(2, 16, 4, 8), _scope="k")
    v = b.reshape(h, shape=(2, 16, 4, 8), _scope="v")
    a = b.sdpa(q, k, v, _scope="attn")
    a = b.reshape(a, shape=(2, 16, 32), _scope="attn_out")
    a = b.add(a, x, _scope="residual")
    a = b.gelu(a, _scope="mlp_act")
    a = b.linear(a, 32, scope="mlp")
    a = b.dropout(a, p=0.1, _scope="dropout")
    a = b.softmax(a, dim=-1, _scope="softmax")
    b.output([a], names=["probs"])
    return build_tfg(b.build(), _profile(b.build(), scale=0.6))


def _arith_probe() -> TFG:
    b = GraphBuilder("probe_arith")
    x = b.input("x", (2, 8, 16, 16))
    y = b.input("y", (2, 8, 16, 16))
    s = b.add(x, y, _scope="add")
    d = b.mul(s, other=2.0, _scope="scale")
    c = b.concat([d, x], dim=1, _scope="concat")
    p = b.pad(c, pad=(1, 1, 1, 1), _scope="pad")
    u = b.interpolate(p, size=(32, 32), mode="nearest", _scope="upsample")
    m = b.mean(u, dim=(2, 3), keepdim=True, _scope="gap")
    m = b.clamp(m, min=-1.0, max=1.0, _scope="clamp")
    f = b.flatten(m, start_dim=1, _scope="flatten")
    f = b.batchnorm(f, num_features=16, _scope="norm")
    g = b.groupnorm(f, num_groups=1, num_features=16, _scope="gn")
    z = b.cast(x, dtype="float16", _scope="cast")
    zf = b.flatten(z, start_dim=1, _scope="cast_flat")
    out = b.add(g, zf.to("float32") if hasattr(zf, "to") else b.cast(zf, dtype="float32", _scope="back"),
                _scope="out")
    b.output([out], names=["features"])
    return build_tfg(b.build(), _profile(b.build(), scale=1.5, alias=True))


def _alias_probe() -> TFG:
    b = GraphBuilder("probe_alias")
    x = b.input("images", (2, 3, 24, 24))
    h = b.conv_block(x, 8, 3, scope="stem")
    vw = b.reshape(h, shape=(2, 8, 24 * 24), _scope="view")
    cp = b.clone(vw, _scope="clone")
    sm = b.softmax(cp, dim=-1, _scope="softmax")
    r = b.reshape(sm, shape=(2, 8, 24, 24), _scope="restore")
    r2 = b.add(r, h, _scope="residual")
    pooled = b.avgpool2d(r2, kernel_size=2, stride=2, _scope="pool")
    fl = b.flatten(pooled, start_dim=1, _scope="flatten")
    emb_table = b.param((64, 32), init="normal", name="embedding")
    ids = b.input("ids", (2, 12), dtype="int64")
    e = b.embedding(ids, emb_table, _scope="embed")
    e = b.mean(e, dim=1, keepdim=False, _scope="pool_tokens")
    e = b.unsqueeze(e, dim=-1, _scope="unsqueeze")
    e = b.squeeze(e, dim=-1, _scope="squeeze")
    out = b.linear(fl, 32, scope="head")
    b.output([out, e], names=["logits", "tokens"])
    return build_tfg(b.build(), _profile(b.build(), scale=0.8, alias=True))


_PROBE_BUILDERS = (_conv_probe, _norm_attn_probe, _arith_probe, _alias_probe)
_CACHE: List[TFG] = []


def probe_tfgs() -> List[TFG]:
    """The (cached) synthetic probe suite."""
    global _CACHE
    if not _CACHE:
        graphs: List[TFG] = []
        for builder in _PROBE_BUILDERS:
            try:
                graphs.append(builder())
            except Exception:
                continue
        _CACHE = graphs
    return list(_CACHE)
