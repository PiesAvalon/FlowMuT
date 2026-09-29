"""Tests for the mutation operator pool and the constraint system."""

from __future__ import annotations

import pytest

from flowmut.operators.base import (
    OBJECTS,
    PRIMITIVES,
    Candidate,
    NodeTemplate,
    Pattern,
    Replacement,
    ReplacementError,
    Site,
    apply_replacement,
)
from flowmut.operators.constraints import (
    CONSTRAINT_VARIANTS,
    ConstraintChecker,
)
from flowmut.operators.pool import (
    build_pool,
    generate_candidates,
    pool_composition,
    validate_pool,
)


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def test_pool_is_the_paper_union():
    pool, report = build_pool()
    assert report.cg == 120
    assert report.tfg == 48
    assert len(pool) == 147
    assert len(report.overlaps) == 21
    assert report.cg + report.tfg - len(report.overlaps) == len(pool)
    assert report.tfg_only == 27
    # The 21 operators reached by both derivations carry the combined source.
    assert report.cg_only == 120 - 21
    assert sum(1 for op in pool if op.source == "CG+TFG") == 21
    assert report.method == "behavioural"


def test_declared_overlaps_are_confirmed_behaviourally():
    pool, report = build_pool()
    assert report.table_problems == []
    assert len(report.declared_overlaps) == 21
    assert len(report.confirmed_overlaps) == 21
    assert report.unconfirmed_overlaps == []


def test_pool_composition_matches_the_construction_table():
    from flowmut.operators.pool import cg_operators, tfg_operators
    pool, _ = build_pool()
    composition = pool_composition(pool, cg_operators(), tfg_operators())
    assert composition["paper_total"] == 147
    assert composition["built_total"] == 147
    assert composition["overlap_total"] == 21
    assert composition["tfg_only_total"] == 27
    assert composition["matches_paper"] is True
    for row in composition["rows"]:
        assert row["delta"] == 0, row
        assert row["pool"] == (row["cg_retained"] + row["tfg_derived"]
                               - row["overlap"])


def test_pool_can_be_built_without_deduplication():
    pool, report = build_pool(deduplicate=False, probe_dedup=False)
    assert len(pool) == 168
    assert set(report.overlaps) == set()


def test_pool_operators_are_well_formed():
    pool, _ = build_pool()
    names = [op.name for op in pool]
    assert len(set(names)) == len(names)
    for op in pool:
        assert op.target_object in OBJECTS
        assert op.primitive in PRIMITIVES
        assert op.source in ("CG", "TFG", "CG+TFG")
        assert isinstance(op.pattern, Pattern)
        assert callable(op.build_replacement)
        assert isinstance(op.to_dict(), dict)
        assert op.to_dict()["name"] == op.name


def test_candidate_generation_is_edge_driven(tiny_tfg):
    pool, _ = build_pool()
    candidates = generate_candidates(tiny_tfg, pool)
    assert candidates
    assert all(isinstance(c, Candidate) for c in candidates)
    anchors = {c.site.anchor_edge for c in candidates}
    assert anchors <= set(tiny_tfg.graph.data_edges())


def test_candidate_generation_respects_object_filter(tiny_tfg):
    pool, _ = build_pool()
    candidates = generate_candidates(tiny_tfg, pool, only_objects=["Tensor"])
    assert candidates
    assert {c.operator.target_object for c in candidates} == {"Tensor"}


# ---------------------------------------------------------------------------
# Replacement machinery
# ---------------------------------------------------------------------------

def test_apply_replacement_inserts_on_an_edge(tiny_tfg):
    graph = tiny_tfg.graph
    target = [e for e in graph.data_edges()
              if graph.nodes[graph.edges[e].producer].op == "conv2d"][0]
    site = Site(anchor_edge=target, consumer=graph.consumers_of(target)[0],
                nodes=[graph.consumers_of(target)[0]])
    replacement = Replacement(
        templates=[NodeTemplate(key="c", op="clone", inputs=["{anchor}"])],
        remove=[],
        rewires=[(c, p, "@c") for (c, p) in graph.edges[target].consumers],
    )
    mutated = apply_replacement(graph, site, replacement, "test.clone")
    assert mutated.check() == []
    assert len(mutated.nodes) == len(graph.nodes) + 1
    new_consumer = graph.consumers_of(target)[0]
    assert mutated.edges[mutated.nodes[new_consumer].inputs[0]].producer != \
        graph.edges[target].producer


def test_apply_replacement_deletes_a_node(tiny_tfg):
    graph = tiny_tfg.graph
    relu = [n for n in graph.nodes.values() if n.op == "relu"][0]
    in_edge = relu.inputs[0]
    out_edges = graph.out_edges(relu.id)
    site = Site(anchor_edge=in_edge, consumer=relu.id, nodes=[relu.id])
    replacement = Replacement(remove=[relu.id],
                              outputs={e: in_edge for e in out_edges})
    mutated = apply_replacement(graph, site, replacement, "test.delete")
    assert mutated.check() == []
    assert relu.id not in mutated.nodes


def test_apply_replacement_updates_attrs_in_place(tiny_tfg):
    graph = tiny_tfg.graph
    conv = [n for n in graph.nodes.values() if n.op == "conv2d"][0]
    site = Site(anchor_edge=conv.inputs[0], consumer=conv.id, nodes=[conv.id])
    replacement = Replacement(remove=[], keep=[conv.id],
                              attr_updates={conv.id: {"stride": 2}})
    mutated = apply_replacement(graph, site, replacement, "test.update")
    assert mutated.nodes[conv.id].attrs["stride"] == 2
    assert mutated.check() == []


def test_replacement_refuses_unknown_references(tiny_tfg):
    graph = tiny_tfg.graph
    edge = graph.data_edges()[0]
    site = Site(anchor_edge=edge, consumer=graph.consumers_of(edge)[0],
                nodes=[graph.consumers_of(edge)[0]])
    replacement = Replacement(templates=[NodeTemplate(key="x", op="relu",
                                                      inputs=["@missing"])],
                              remove=[])
    with pytest.raises(ReplacementError):
        apply_replacement(graph, site, replacement, "test.bad")


# ---------------------------------------------------------------------------
# Constraint system
# ---------------------------------------------------------------------------

def test_constraint_variants_are_declared():
    assert set(CONSTRAINT_VARIANTS) == {
        "full", "without_input", "without_internal", "without_output",
        "without_constraints",
    }


def test_full_constraints_reject_fewer_candidates_than_no_constraints(tiny_tfg):
    pool, _ = build_pool()
    candidates = generate_candidates(tiny_tfg, pool)
    full = ConstraintChecker.for_variant("full")
    none = ConstraintChecker.for_variant("without_constraints")
    accepted_full = sum(1 for c in candidates if full.filter(tiny_tfg, c).accepted)
    accepted_none = sum(1 for c in candidates if none.filter(tiny_tfg, c).accepted)
    assert accepted_none > 0
    assert accepted_full <= accepted_none
    assert accepted_full > 0


def test_constraint_categories_are_used(tiny_tfg):
    pool, _ = build_pool()
    checker = ConstraintChecker()
    kinds = set()
    for candidate in generate_candidates(tiny_tfg, pool):
        report = checker.filter(tiny_tfg, candidate)
        for violation in report.violations:
            kinds.add(violation.kind)
    assert kinds <= {"input", "internal", "output", "structural"}


def test_violations_are_reported_with_a_reason(tiny_tfg):
    """A dtype cast feeding a convolution must be rejected as an input violation."""
    from flowmut.operators.base import FunctionalOperator
    graph = tiny_tfg.graph
    conv = [n for n in graph.nodes.values() if n.op == "conv2d"][0]
    edge = conv.inputs[0]
    site = Site(anchor_edge=edge, consumer=conv.id, nodes=[conv.id],
                output_edges=graph.out_edges(conv.id))
    op = FunctionalOperator(
        name="test.cast_input_to_int", target_object="Interface", primitive="Update",
        source="CG", pattern=Pattern(op_in=("conv2d", "linear", "batchnorm")),
        build=lambda tfg, s: Replacement(
            templates=[NodeTemplate(key="cast", op="cast", inputs=["{anchor}"],
                                    attrs={"dtype": "int64"})],
            remove=[],
            rewires=[(c, p, "@cast") for (c, p) in tfg.graph.edges[s.anchor_edge].consumers
                     if not tfg.graph.nodes[c].is_param],
            keep=[s.consumer]),
    )
    report = ConstraintChecker().filter(tiny_tfg, Candidate(site=site, operator=op))
    assert not report.accepted
    # The int64 tensor is produced inside the replacement and consumed outside it,
    # so the violation is an *output* constraint on the convolution's operand.
    assert report.first_violation_kind in ("input", "internal", "output")
    assert any("floating point" in v.message for v in report.violations)


def test_noop_replacements_are_rejected(tiny_tfg):
    from flowmut.operators.base import FunctionalOperator
    graph = tiny_tfg.graph
    relu = [n for n in graph.nodes.values() if n.op == "relu"][0]
    site = Site(anchor_edge=relu.inputs[0], consumer=relu.id, nodes=[relu.id])
    op = FunctionalOperator(
        name="test.noop", target_object="Operator", primitive="Update", source="CG",
        pattern=Pattern(op="relu"),
        build=lambda tfg, s: Replacement(remove=[], keep=[s.consumer]),
    )
    report = ConstraintChecker().filter(tiny_tfg, Candidate(site=site, operator=op))
    assert not report.accepted
    assert "no-op" in report.error


def test_validate_pool_reports_per_operator_results(tiny_tfg):
    pool, _ = build_pool()
    summary = validate_pool(pool, [tiny_tfg], max_sites_per_operator=1)
    assert summary["operators"] == len(pool)
    assert summary["validated"] > 0
    assert "results" in summary
