"""Tests for the IR: graph construction, operator registry and shape inference."""

from __future__ import annotations

import pytest

from flowmut.ir.graph import GraphBuilder
from flowmut.ir.ops import OP_REGISTRY, canonical_op, infer_shape, requirements_for
from flowmut.ir.specs import TensorSpec, broadcast_shape, parse_mode


def test_builder_produces_a_valid_dag(tiny_graph):
    assert tiny_graph.check() == []
    assert tiny_graph.inputs and tiny_graph.outputs
    assert len(tiny_graph.nodes) > 10


def test_shape_propagation_through_chained_layers(tiny_graph):
    order = tiny_graph.topological_order()
    by_scope = {tiny_graph.nodes[n].scope: n for n in order}
    conv = tiny_graph.nodes[by_scope["stem"]]
    assert conv.out_shape == (2, 8, 8, 8)
    gap = tiny_graph.nodes[by_scope["gap"]]
    assert gap.out_shape == (2, 16, 1, 1)
    head = tiny_graph.nodes[by_scope["head"]]
    assert head.out_shape == (2, 10)


def test_topological_order_is_consistent(tiny_graph):
    seen = set()
    for nid in tiny_graph.topological_order():
        for edge in tiny_graph.nodes[nid].inputs:
            assert tiny_graph.edges[edge].producer in seen or tiny_graph.edges[edge].producer == nid
        seen.add(nid)


def test_op_registry_aliases():
    assert canonical_op("bn") == "batchnorm"
    assert canonical_op("cat") == "concat"
    assert canonical_op("conv2d") == "conv2d"
    assert len(OP_REGISTRY) > 80


def test_requirements_are_attached_and_evaluable(tiny_graph):
    for node in tiny_graph.nodes.values():
        assert isinstance(node.requirements, list)
        reqs = requirements_for(node)
        assert len(reqs) == len(node.requirements)
        for req in reqs:
            assert req.kind in ("input", "internal", "output")
            assert isinstance(req.predicate, str)


def test_shape_inference_rules():
    x = TensorSpec(shape=(1, 3, 8, 8))
    w = TensorSpec(shape=(16, 3, 3, 3))
    assert infer_shape("conv2d", [x, w], {"stride": 1, "padding": 1}) == (1, 16, 8, 8)
    assert infer_shape("conv2d", [x, w], {"stride": 2, "padding": 1}) == (1, 16, 4, 4)
    a = TensorSpec(shape=(1, 4, 8))
    b = TensorSpec(shape=(1, 1, 8))
    assert infer_shape("add", [a, b], {}) == (1, 4, 8)
    assert infer_shape("reshape", [a], {"shape": (1, 32)}) == (1, 32)
    assert infer_shape("matmul", [TensorSpec(shape=(2, 4, 8)), TensorSpec(shape=(2, 8, 5))],
                       {}) == (2, 4, 5)


def test_broadcast_shape_helpers():
    assert broadcast_shape((1, 3, 8), (2, 3, 8)) == (2, 3, 8)
    assert broadcast_shape((1, 3, 8), (2, 4, 8)) is None


def test_parse_mode_aliases():
    assert parse_mode("torch-eager") == ("pytorch", "eager")
    assert parse_mode("torch_compiled") == ("pytorch", "compiled")
    assert parse_mode("ms-graph") == ("mindspore", "graph")
    assert parse_mode("mindspore") == ("mindspore", "pynative")
    with pytest.raises(ValueError):
        parse_mode("tensorflow")


def test_copy_is_independent(tiny_graph):
    clone = tiny_graph.copy()
    clone.nodes.pop(next(iter(clone.nodes)))
    assert len(clone.nodes) == len(tiny_graph.nodes) - 1
    assert tiny_graph.check() == []
