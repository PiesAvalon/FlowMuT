"""Tests for the Tensor Flow Graph: annotations, requirements and signatures."""

from __future__ import annotations

import pytest

from flowmut.ir.graph import GraphBuilder
from flowmut.tfg import build_tfg
from flowmut.tfg.predicates import ReqContext, evaluate, predicate_names
from flowmut.tfg.tfg import Violation, aggregate_specs
from flowmut.ir.specs import TensorSpec, ValueProfile


def test_tfg_annotates_every_edge(tiny_tfg):
    summary = tiny_tfg.summary()
    assert summary["profiled_edges"] == len(tiny_tfg.graph.edges)
    assert summary["num_profiles"] == 2
    for edge_id in tiny_tfg.graph.edges:
        assert tiny_tfg.f(edge_id) is not None


def test_aggregate_collapses_polymorphic_shapes(tiny_tfg):
    # The fabricated profiles use batch sizes 2 and 3, so every data edge whose
    # shape starts with the batch dimension is polymorphic.
    data_edges = tiny_tfg.graph.data_edges()
    assert any(tiny_tfg.is_polymorphic(e) for e in data_edges)
    for e in data_edges:
        assert tiny_tfg.f(e) is not None


def test_aggregate_specs_merges_value_profiles():
    a = TensorSpec(shape=(1, 4), value=ValueProfile(min=-1, max=1, abs_mean=0.5,
                                                    magnitude_bin=-1))
    b = TensorSpec(shape=(1, 4), dtype="float16",
                   value=ValueProfile(min=-2, max=2, abs_mean=1.5, magnitude_bin=0))
    merged = aggregate_specs([a, b])
    assert merged.shape == (1, 4)
    # Profiling inputs that disagree on the dtype widen to the dtype that can
    # represent every observed state.
    assert merged.dtype == "float32"
    assert merged.value.min == -2 and merged.value.max == 2


def test_aggregate_specs_promotes_integers_to_float():
    a = TensorSpec(shape=(2, 2), dtype="int64")
    b = TensorSpec(shape=(2, 2), dtype="float32")
    assert aggregate_specs([a, b]).dtype == "float32"
    # Widening, never narrowing, when the profiling inputs disagree.
    assert aggregate_specs([TensorSpec(shape=(2, 2), dtype="int32"), a]).dtype == "int64"
    assert aggregate_specs([TensorSpec(shape=(2, 2), dtype="int8"),
                            TensorSpec(shape=(2, 2), dtype="int16")]).dtype == "int16"


def test_check_node_reports_no_violations_for_a_valid_graph(tiny_tfg):
    for node in tiny_tfg.graph.nodes.values():
        violations = tiny_tfg.check_node(node)
        assert violations == [], (node.op, [str(v) for v in violations])


def test_check_node_detects_a_broken_boundary():
    b = GraphBuilder("broken")
    x = b.input("x", (1, 3, 8, 8))
    b.conv_block(x, 8, 3, scope="c")
    g = b.build()
    tfg = build_tfg(g, {})
    node = g.node_by_scope("c")
    tfg.edge_specs[node.inputs[0]] = [TensorSpec(shape=(1, 3, 8, 8), dtype="int64")]
    violations = tfg.check_node(node)
    assert any("floating point" in v.message for v in violations)


def test_predicate_library_is_populated():
    names = predicate_names()
    for expected in ("dtype_family", "rank_eq", "shape_broadcastable",
                     "matmul_compatible", "channel_groups_match",
                     "reshape_numel_preserved", "attention_dims_match"):
        assert expected in names


def test_unknown_predicate_is_vacuously_true():
    ctx = ReqContext(node=None, attrs={}, specs=[])
    assert evaluate("this_predicate_does_not_exist", ctx) is None


def test_data_flow_signature_includes_tensor_state(tiny_tfg):
    signatures = tiny_tfg.all_signatures()
    assert signatures
    for sig in signatures:
        producer_op, consumer_op, path, tensor_state, position = sig
        assert isinstance(producer_op, str) and isinstance(consumer_op, str)
        assert isinstance(tensor_state, tuple)
        assert tensor_state[0] >= 1          # rank
        assert isinstance(tensor_state[1], str)   # dtype


def test_violation_serialisation():
    v = Violation(kind="input", predicate="rank_eq", node="n1", message="bad", edge="e2")
    payload = v.to_dict()
    assert payload["kind"] == "input" and payload["edge"] == "e2"
    assert "rank_eq" in str(v)


def test_build_tfg_attaches_requirements_to_every_node():
    b = GraphBuilder("reqs")
    x = b.input("x", (1, 3, 8, 8))
    h = b.conv2d(x, b.param((8, 3, 3, 3), name="w"), stride=1, padding=1)
    h = b.softmax(h, dim=1, _scope="sm")
    g = b.build()
    tfg = build_tfg(g, {})
    total = sum(len(n.requirements) for n in g.nodes.values())
    assert total > 0
    conv = g.node_by_scope("") or list(g.nodes.values())[1]
    assert any(r.predicate == "rank_eq" for r in g.nodes[[n.id for n in g.nodes.values()
                                                          if n.op == "conv2d"][0]].requirements)
