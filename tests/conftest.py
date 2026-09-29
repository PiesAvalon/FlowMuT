"""Shared pytest fixtures for the FlowMuT test-suite."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowmut.ir.graph import GraphBuilder  # noqa: E402
from flowmut.ir.specs import TensorSpec, ValueProfile  # noqa: E402
from flowmut.tfg import build_tfg  # noqa: E402


def _fabricate_profile(graph, value_scale: float = 1.0):
    specs = {}
    for eid, edge in graph.edges.items():
        node = graph.nodes[edge.producer]
        shape = tuple(edge.spec.shape) if edge.spec else tuple(node.out_shape or (1, 8, 8, 8))
        dtype = edge.spec.dtype if edge.spec else "float32"
        base = TensorSpec(shape=shape, dtype=dtype, requires_grad=True)
        specs[eid] = [
            base.with_(value=ValueProfile(min=-value_scale, max=value_scale,
                                          abs_mean=value_scale, magnitude_bin=0)),
            base.with_(shape=(max(1, shape[0] + 1),) + shape[1:],
                       value=ValueProfile(min=-value_scale, max=value_scale,
                                          abs_mean=value_scale, magnitude_bin=0)),
        ]
    return specs


@pytest.fixture(scope="session")
def tiny_graph():
    b = GraphBuilder("tiny")
    x = b.input("images", (2, 3, 16, 16))
    h = b.conv_block(x, 8, 3, stride=2, scope="stem")
    h = b.relu(h, _scope="act")
    h = b.conv_block(h, 16, 3, scope="body")
    h = b.global_avgpool(h, keepdim=True, _scope="gap")
    h = b.flatten(h, start_dim=1, _scope="flatten")
    logits = b.linear(h, 10, scope="head")
    probs = b.softmax(logits, dim=-1, _scope="softmax")
    b.output([logits, probs], names=["logits", "probs"])
    return b.build()


@pytest.fixture(scope="session")
def tiny_tfg(tiny_graph):
    graph = tiny_graph.copy()
    return build_tfg(graph, _fabricate_profile(graph))


@pytest.fixture(scope="session")
def rich_tfg():
    """A probe-like TFG with alias, view, polymorphic and value-rich edges."""
    from flowmut.operators.probes import probe_tfgs
    tfgs = probe_tfgs()
    assert tfgs
    return tfgs[0]


@pytest.fixture(scope="session")
def torch_adapter():
    from flowmut.adapters.torch_adapter import PyTorchEagerAdapter
    ok, reason = PyTorchEagerAdapter.is_available()
    if not ok:
        pytest.skip(f"PyTorch unavailable: {reason}")
    return PyTorchEagerAdapter(timeout_s=120)
