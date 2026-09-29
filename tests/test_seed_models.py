"""Tests for the seed models and the FlowMuT main loop."""

from __future__ import annotations

import numpy as np
import pytest

from flowmut.config import FlowMuTConfig
from flowmut.ir.ops import OP_REGISTRY, canonical_op
from flowmut.loop.engine import FlowMuTEngine
from flowmut.operators.pool import build_pool
from flowmut.seed_models.registry import all_seeds, make_sample, seed_keys

EXPECTED_TASKS = {
    "Image Classification", "Object Detection", "Semantic Segmentation",
    "Text Classification", "Industrial Anomaly Detection", "Keypoint Detection",
    "Scene Text Recognition", "Text Generation", "Image Generation",
    "Image Restoration",
}


# ---------------------------------------------------------------------------
# Seed models
# ---------------------------------------------------------------------------

def test_all_fourteen_seed_models_are_registered():
    import flowmut.seed_models as sm
    assert sm.load_errors() == {}
    assert len(all_seeds()) == 14


def test_seed_models_cover_ten_application_tasks():
    tasks = {s.task for s in all_seeds()}
    assert tasks == EXPECTED_TASKS


@pytest.mark.parametrize("model", all_seeds(), ids=lambda m: m.key)
def test_seed_model_graph_is_valid(model):
    graph = model.fresh_graph()
    assert graph.check() == []
    assert model.input_shapes
    assert len(graph.nodes) >= 20
    assert graph.outputs


@pytest.mark.parametrize("model", all_seeds(), ids=lambda m: m.key)
def test_seed_model_only_uses_registered_ops(model):
    graph = model.fresh_graph()
    unknown = [n.op for n in graph.nodes.values() if canonical_op(n.op) not in OP_REGISTRY]
    assert not unknown


@pytest.mark.parametrize("model", all_seeds(), ids=lambda m: m.key)
def test_seed_model_samples_match_declared_inputs(model):
    sample = make_sample(model)
    assert set(sample) == set(model.input_shapes)
    for name, array in sample.items():
        assert tuple(array.shape) == tuple(model.input_shapes[name])
        assert np.all(np.isfinite(array))


def test_seed_model_metadata_matches_the_benchmark_table():
    expected = {
        "convnext_v2_tiny": "28,635,496",
        "swin_v2_tiny": "28,347,154",
        "efficientnetv2_s": "21,458,488",
        "yolov10_s": "8,128,272",
        "rtdetrv2_s": "20,226,500",
        "segformer_b1": "13,715,798",
        "mask2former_swin_t": "47,441,169",
        "modernbert_base": "149,606,402",
        "efficientad_s": "8,057,856",
        "rtmpose_m": "13,587,611",
        "parseq": "23,832,671",
        "qwen2_5_0_5b": "494,032,768",
        "dit_s_2": "32,963,360",
        "swinir_m_x4": "11,900,199",
    }
    index = {s.key: s for s in all_seeds()}
    assert set(index) == set(expected)
    for key, count in expected.items():
        assert f"{index[key].param_count:,}" == count, key


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_config_normalises_mode_aliases():
    assert FlowMuTConfig(mode="torch-eager").mode == "pytorch:eager"
    assert FlowMuTConfig(mode="ms-graph").mode_name == "graph"


def test_config_reference_mode_for_compiled_and_graph():
    assert FlowMuTConfig(mode="pytorch-compiled").effective_reference_mode == "pytorch:eager"
    assert FlowMuTConfig(mode="ms-graph").effective_reference_mode == "mindspore:pynative"
    assert FlowMuTConfig(mode="pytorch-eager").effective_reference_mode is None


def test_config_ablation_presets():
    assert FlowMuTConfig.ablation("without_constraints").constraint_variant == "without_constraints"
    assert FlowMuTConfig.ablation("coverage_only").feature_variant == "coverage_only"
    assert FlowMuTConfig.ablation("cg_only").include_tfg is False


def test_config_round_trip():
    config = FlowMuTConfig(mode="torch-eager", rounds=7)
    assert FlowMuTConfig.from_json(config.to_json()).rounds == 7


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pool():
    pool, _ = build_pool()
    return pool


def test_engine_setup_profiles_the_seed_model(pool):
    config = FlowMuTConfig(mode="torch-eager", rounds=1, profile_samples=2)
    engine = FlowMuTEngine(config, "efficientnetv2_s", pool=pool)
    engine.setup()
    assert engine.tfg is not None
    assert engine.tfg.edge_specs
    assert engine.samples
    assert engine.policy is not None


def test_engine_runs_a_short_campaign(pool):
    config = FlowMuTConfig(mode="torch-eager", rounds=4, profile_samples=2,
                           max_candidates=80, output_dir="/tmp/flowmut-test")
    engine = FlowMuTEngine(config, "efficientnetv2_s", pool=pool)
    engine.setup()
    result = engine.run()
    assert not result.error, result.error
    assert result.rounds_completed == 4
    assert len(result.iterations) == 4
    assert result.candidate_statistics["total_generated"] > 0
    assert result.trace_memory["distinct_events"] > 0
    assert result.checkpoints
    assert result.final_graph["nodes"] > 0


def test_engine_produces_legal_mutants(pool):
    config = FlowMuTConfig(mode="torch-eager", rounds=5, max_candidates=80,
                           output_dir="/tmp/flowmut-test")
    engine = FlowMuTEngine(config, "efficientnetv2_s", pool=pool)
    engine.setup()
    result = engine.run()
    retained = result.candidate_statistics["total_retained"]
    assert retained > 0
    executed = [i for i in result.iterations if i.status in ("ok", "error")]
    assert executed, "the loop must actually execute mutants"
    assert any(i.seed_updated for i in result.iterations), \
        "at least one accepted mutation should become the next seed"


def test_engine_survives_a_broken_seed_model(pool):
    """A failing campaign must be reported, never raised."""
    config = FlowMuTConfig(mode="torch-eager", rounds=2, max_candidates=20)
    engine = FlowMuTEngine(config, "efficientnetv2_s", pool=pool)
    engine.setup()
    engine.samples = [{"nonexistent_input": np.zeros((1, 3, 8, 8), np.float32)}]
    result = engine.run()
    assert result.rounds_completed >= 0


def test_constraint_ablation_runs(pool):
    config = FlowMuTConfig(mode="torch-eager", rounds=3, max_candidates=60,
                           constraint_variant="without_constraints",
                           output_dir="/tmp/flowmut-test")
    engine = FlowMuTEngine(config, "efficientnetv2_s", pool=pool)
    engine.setup()
    result = engine.run()
    assert not result.error, result.error
    assert result.legality["constraint_variant"] == "without_constraints"


def test_policy_ablation_runs(pool):
    config = FlowMuTConfig(mode="torch-eager", rounds=3, max_candidates=60,
                           policy="random", feature_variant="coverage_only",
                           output_dir="/tmp/flowmut-test")
    engine = FlowMuTEngine(config, "efficientnetv2_s", pool=pool)
    engine.setup()
    result = engine.run()
    assert not result.error, result.error
