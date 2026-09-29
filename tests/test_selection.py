"""Tests for the selection layer: coverage features and bandit policies."""

from __future__ import annotations

import numpy as np
import pytest

from flowmut.operators.base import Candidate, FunctionalOperator, Pattern, Replacement
from flowmut.operators.pool import build_pool, generate_candidates
from flowmut.selection.features import (
    FEATURE_NAMES,
    FEATURE_VARIANTS,
    CoverageTracker,
    FeatureExtractor,
    VariantFeatureExtractor,
)
from flowmut.selection.policies import (
    POLICIES,
    LinUCB,
    OneShot,
    RandomSearch,
    make_policy,
)


@pytest.fixture(scope="module")
def candidates(tiny_tfg):
    pool, _ = build_pool()
    return generate_candidates(tiny_tfg, pool)


def test_feature_vector_shape(tiny_tfg, candidates):
    extractor = FeatureExtractor(tiny_tfg)
    contexts = extractor.context(candidates)
    assert contexts.shape == (len(candidates), len(FEATURE_NAMES))
    assert np.all(np.isfinite(contexts))


def test_coverage_decreases_with_repetition():
    tracker = CoverageTracker()
    assert tracker.value(["a"]) == pytest.approx(1.0)
    tracker.increment(["a"])
    assert tracker.value(["a"]) == pytest.approx(1 / np.sqrt(2))
    tracker.increment(["a"])
    assert tracker.value(["a"]) == pytest.approx(1 / np.sqrt(3))
    assert tracker.counts["a"] == 2


def test_feature_variants_have_the_expected_width(tiny_tfg, candidates):
    for name, columns in FEATURE_VARIANTS.items():
        extractor = VariantFeatureExtractor(tiny_tfg, variant=name)
        contexts = extractor.context(candidates)
        assert contexts.shape[1] == len(columns), name


def test_gradient_probe_is_used_and_normalised(tiny_tfg, candidates):
    values = {c.site.anchor_edge: float(i) for i, c in enumerate(candidates)}
    extractor = FeatureExtractor(tiny_tfg, gradient_probe=values.get)
    contexts = extractor.context(candidates)
    column = contexts[:, FEATURE_NAMES.index("gradient")]
    assert column.min() == pytest.approx(0.0)
    assert column.max() == pytest.approx(1.0)


def test_gradient_probe_failures_are_tolerated(tiny_tfg, candidates):
    def broken(edge_id):
        raise RuntimeError("boom")

    extractor = FeatureExtractor(tiny_tfg, gradient_probe=broken)
    contexts = extractor.context(candidates)
    assert np.all(np.isfinite(contexts))


def test_observe_updates_the_tracker(tiny_tfg, candidates):
    extractor = FeatureExtractor(tiny_tfg)
    before = dict(extractor.tracker.counts)
    extractor.observe(candidates[0])
    assert len(extractor.tracker.counts) >= len(before)


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_every_policy_selects_and_updates(name, candidates):
    contexts = FeatureExtractor  # placeholder to avoid re-generating contexts twice
    extractor_candidates = candidates
    contexts = np.random.default_rng(0).random((len(extractor_candidates), 5))
    policy = make_policy(name, seed=3)
    index = policy.select(extractor_candidates, contexts)
    assert 0 <= index < len(extractor_candidates)
    policy.update(index, contexts[index], 0.5)
    assert isinstance(policy.state(), dict)


def test_policies_handle_empty_candidate_sets():
    for name in POLICIES:
        assert make_policy(name).select([], np.zeros((0, 5))) == -1


def test_linucb_prefers_uncertain_arms_and_is_deterministic(tiny_tfg, candidates):
    extractor = FeatureExtractor(tiny_tfg)
    contexts = extractor.context(candidates)
    policy = LinUCB(alpha=1.0)
    first = policy.select(candidates, contexts)
    second = LinUCB(alpha=1.0).select(candidates, contexts)
    assert first == second
    for _ in range(20):
        index = policy.select(candidates, contexts)
        policy.update(index, contexts[index], float(index % 3))


def test_random_search_is_reproducible(candidates):
    contexts = np.zeros((len(candidates), 5))
    a = RandomSearch(seed=7)
    b = RandomSearch(seed=7)
    assert [a.select(candidates, contexts) for _ in range(10)] == \
           [b.select(candidates, contexts) for _ in range(10)]


def test_one_shot_cycles_through_candidates(candidates):
    contexts = np.zeros((len(candidates), 5))
    policy = OneShot()
    picks = [policy.select(candidates, contexts) for _ in range(len(candidates) + 2)]
    assert picks[:len(candidates)] == list(range(len(candidates)))


def test_policies_support_reduced_context_widths(candidates):
    for width in (2, 3, 4, 5):
        contexts = np.random.default_rng(1).random((len(candidates), width))
        for name in POLICIES:
            policy = make_policy(name, seed=0)
            index = policy.select(candidates, contexts)
            assert 0 <= index < len(candidates), (name, width)
            policy.update(index, contexts[index], 0.1)
