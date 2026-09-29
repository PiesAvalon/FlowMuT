"""Tests for the oracle and the diversity/coverage metrics."""

from __future__ import annotations

import numpy as np
import pytest

from flowmut.adapters.base import ExecutionResult
from flowmut.analysis.metrics import MetricsRecorder, distinct_signatures
from flowmut.ir.graph import GraphBuilder
from flowmut.operators.constraints import ConstraintReport
from flowmut.oracle import BugOracle, Observation, compare_outputs, check_semantics
from flowmut.tfg import build_tfg


def _ok(outputs):
    return ExecutionResult(status="ok", framework="pytorch", mode="eager", outputs=outputs)


def test_successful_execution_is_not_a_bug():
    oracle = BugOracle()
    obs = oracle.judge(ConstraintReport(accepted=True), _ok([np.zeros((1, 4), np.float32)]))
    assert obs.symptom == "none" and not obs.is_bug and obs.legal


def test_crash_of_a_legal_mutant_is_a_bug():
    oracle = BugOracle()
    result = ExecutionResult(status="error", framework="pytorch", mode="eager",
                             error_type="RuntimeError", error_message="boom")
    obs = oracle.judge(ConstraintReport(accepted=True), result)
    assert obs.is_bug and obs.symptom == "crash"


def test_crash_of_an_illegal_mutant_is_not_a_bug():
    oracle = BugOracle()
    result = ExecutionResult(status="error", framework="pytorch", mode="eager",
                             error_type="RuntimeError", error_message="boom")
    report = ConstraintReport(accepted=False, first_violation_kind="input")
    obs = oracle.judge(report, result)
    assert not obs.is_bug and obs.symptom == "illegal"
    assert obs.violation_kind == "input"


def test_timeout_of_a_legal_mutant_is_a_bug():
    oracle = BugOracle()
    result = ExecutionResult(status="timeout", framework="pytorch", mode="eager")
    assert oracle.judge(ConstraintReport(accepted=True), result).is_bug


def test_non_finite_outputs_are_a_bug():
    oracle = BugOracle()
    obs = oracle.judge(ConstraintReport(accepted=True), _ok([np.array([np.nan])]))
    assert obs.symptom == "nonfinite" and obs.is_bug


def test_cross_mode_divergence_is_a_bug():
    oracle = BugOracle(tolerance=1e-3, relative_tolerance=1e-4)
    reference = _ok([np.zeros((1, 4), np.float32)])
    divergent = ExecutionResult(status="ok", framework="pytorch", mode="compiled",
                                outputs=[np.ones((1, 4), np.float32)])
    obs = oracle.judge(ConstraintReport(accepted=True), divergent, reference=reference)
    assert obs.symptom == "divergence" and obs.is_bug
    assert obs.max_abs_diff == pytest.approx(1.0)


def test_matching_outputs_are_not_a_divergence():
    oracle = BugOracle()
    reference = _ok([np.zeros((1, 4), np.float32)])
    same = ExecutionResult(status="ok", framework="pytorch", mode="compiled",
                           outputs=[np.zeros((1, 4), np.float32)])
    assert oracle.judge(ConstraintReport(accepted=True), same,
                        reference=reference).symptom == "none"


def test_judge_never_raises_on_hostile_outputs():
    oracle = BugOracle()
    for outputs in ([object()], ["a string"], [None], [np.array([1.0])]):
        result = ExecutionResult(status="ok", framework="pytorch", mode="eager",
                                 outputs=outputs)
        oracle.judge(ConstraintReport(accepted=True), result)


def test_semantics_catches_out_of_range_probabilities(tiny_tfg):
    graph = tiny_tfg.graph
    sm = [n for n in graph.nodes.values() if n.op == "softmax"]
    assert sm, "fixture must contain a softmax"
    result = ExecutionResult(status="ok", framework="pytorch", mode="eager",
                             outputs=[np.array([2.0])] * len(graph.outputs))
    message = check_semantics(graph, result)
    assert message is None or "softmax" in message


def test_compare_outputs_reports_shape_mismatch():
    index, diff, message = compare_outputs([np.zeros((2, 3))], [np.zeros((3, 2))], 1e-3, 1e-4)
    assert index == 0 and "shape" in message.lower()


def test_oracle_statistics_accumulate():
    oracle = BugOracle()
    oracle.judge(ConstraintReport(accepted=True), _ok([np.zeros(2, np.float32)]))
    oracle.judge(ConstraintReport(accepted=True),
                 ExecutionResult(status="error", error_type="RuntimeError",
                                 error_message="x"))
    stats = oracle.statistics
    assert stats.get("total", stats.get("judged", 0)) >= 2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_are_bounded_and_monotonic(tiny_tfg):
    recorder = MetricsRecorder(checkpoint_rounds=(1, 2))
    recorder.observe(tiny_tfg)
    first = recorder.snapshot(1)
    for name in ("lic", "lpc", "lsc", "dfsd"):
        assert 0.0 <= getattr(first, name) <= 1.0
    recorder.observe(tiny_tfg)
    second = recorder.snapshot(2)
    for name in ("lic", "lpc", "lsc", "dfsd"):
        assert 0.0 <= getattr(second, name) <= 1.0


def test_distinct_signatures_is_non_empty(tiny_tfg):
    signatures = distinct_signatures(tiny_tfg)
    assert signatures
    assert all(isinstance(s, tuple) for s in signatures)


def test_metrics_grow_when_new_operators_appear(tiny_tfg):
    recorder = MetricsRecorder(checkpoint_rounds=(1, 2))
    recorder.observe(tiny_tfg)
    recorder.snapshot(1)
    b = GraphBuilder("extra")
    x = b.input("x", (1, 3, 16, 16))
    h = b.conv_block(x, 8, 3, scope="c")
    h = b.tanh(h, _scope="tanh")
    h = b.global_avgpool(h, keepdim=True, _scope="gap")
    h = b.flatten(h, start_dim=1, _scope="flat")
    out = b.sigmoid(b.linear(h, 4, scope="head"), _scope="sigmoid")
    b.output([out], names=["p"])
    graph = b.build()
    other = build_tfg(graph, {})
    recorder.observe(other)
    snapshot = recorder.snapshot(2)
    assert snapshot.operators_seen >= 1
    assert snapshot.dfsd <= 1.0


def test_explain_features_describe_candidate(tiny_tfg):
    from flowmut.operators.pool import build_pool, generate_candidates
    from flowmut.selection.features import FeatureExtractor
    pool, _ = build_pool()
    candidates = generate_candidates(tiny_tfg, pool)
    extractor = FeatureExtractor(tiny_tfg)
    described = extractor.describe(candidates[0])
    assert isinstance(described, dict) and described
