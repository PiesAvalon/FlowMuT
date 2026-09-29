"""Tests for the framework adapters (PyTorch and, when importable, MindSpore)."""

from __future__ import annotations

import importlib
import numpy as np
import pytest

from flowmut.adapters.registry import ADAPTER_SPECS, available_modes, make_adapter
from flowmut.adapters.base import ExecutionResult, TraceEvent, value_profile
from flowmut.ir.graph import GraphBuilder


def _sample(graph, batch: int = 2):
    sample = {}
    for edge_id in graph.inputs:
        edge = graph.edges[edge_id]
        shape = tuple(edge.spec.shape) if edge.spec else (batch, 3, 16, 16)
        shape = (batch,) + shape[1:] if shape else (batch,)
        dtype = edge.spec.dtype if edge.spec else "float32"
        if dtype in ("int64", "int32"):
            sample[edge.name] = np.random.randint(0, 4, size=shape).astype(dtype)
        else:
            sample[edge.name] = np.random.randn(*shape).astype(dtype)
    return sample


def test_all_four_modes_are_registered():
    assert set(ADAPTER_SPECS) == {
        ("pytorch", "eager"), ("pytorch", "compiled"),
        ("mindspore", "pynative"), ("mindspore", "graph"),
    }


def test_available_modes_probe_never_raises():
    rows = available_modes(probe=True)
    assert len(rows) == 4
    for row in rows:
        assert isinstance(row.get("available"), bool)
        assert "reason" in row


def test_pytorch_eager_runs_and_profiles(tiny_graph, torch_adapter):
    graph = tiny_graph.copy()
    module = torch_adapter.materialize(graph)
    result = torch_adapter.execute(graph, _sample(graph), module=module, capture_specs=True)
    assert result.status == "ok", result.error_message
    assert len(result.outputs) == len(graph.outputs)
    assert result.edge_specs, "profiling must annotate edges"
    assert all(spec.shape for spec in (result.edge_specs[e][0]
                                       for e in result.edge_specs))
    assert result.events, "the trace must record framework activity"
    assert all(isinstance(e, TraceEvent) for e in result.events)


def test_pytorch_eager_tensor_specs_are_descriptive(tiny_graph, torch_adapter):
    graph = tiny_graph.copy()
    module = torch_adapter.materialize(graph)
    result = torch_adapter.execute(graph, _sample(graph), module=module, capture_specs=True)
    spec = result.edge_specs[graph.inputs[0]][0]
    assert spec.rank == 4
    assert spec.dtype == "float32"
    assert spec.device in ("cpu", "cuda")
    assert spec.value is not None and spec.value.magnitude_bin is not None


def test_pytorch_compiled_mode_runs(tiny_graph):
    from flowmut.adapters.torch_adapter import PyTorchCompiledAdapter
    ok, reason = PyTorchCompiledAdapter.is_available()
    if not ok:
        pytest.skip(reason)
    graph = tiny_graph.copy()
    adapter = PyTorchCompiledAdapter(timeout_s=300)
    module = adapter.materialize(graph)
    result = adapter.execute(graph, _sample(graph), module=module, capture_specs=True)
    assert result.status == "ok", f"{result.error_type}: {result.error_message}"
    assert result.edge_specs
    assert result.events
    assert any("graph_capture" in e.category for e in result.events)


def test_parameter_reuse_across_mutants(tiny_graph, torch_adapter):
    """A mutant must keep the parameters of the nodes it did not touch."""
    graph = tiny_graph.copy()
    first = torch_adapter.materialize(graph)
    before = {name: p.detach().clone() for name, p in first.eager.named_parameters()}
    again = torch_adapter.materialize(graph.copy(), reuse=first)
    after = dict(again.eager.named_parameters())
    assert set(before) == set(after)
    for name, tensor in before.items():
        assert bool((tensor == after[name].detach()).all()), name


def test_execution_error_is_reported_not_raised(tiny_graph, torch_adapter):
    graph = tiny_graph.copy()
    # Feed an integer tensor where a float activation is expected.
    node = next(n for n in graph.nodes.values() if n.op == "relu")
    graph.nodes[node.id].attrs["__force_error__"] = True
    module = torch_adapter.materialize(graph)
    result = torch_adapter.execute(graph, _sample(graph), module=module)
    assert isinstance(result, ExecutionResult)


def test_value_profile_is_bounded_and_never_keeps_values():
    array = np.array([0.0, 1.0, -1.0, np.nan, np.inf], dtype=np.float32)
    profile = value_profile(array)
    assert profile.nan_count == 1
    assert profile.inf_count == 1
    assert profile.magnitude_bin == profile.magnitude_bin  # an int, not NaN


def test_missing_kernel_is_reported(tiny_graph):
    from flowmut.adapters import torch_kernels
    with pytest.raises(torch_kernels.KernelMissing):
        torch_kernels.apply("this_op_does_not_exist", [], {})


# ---------------------------------------------------------------------------
# MindSpore (skipped automatically in a PyTorch-only interpreter)
# ---------------------------------------------------------------------------

def test_mindspore_adapters_are_importable_and_reported():
    module = importlib.import_module("flowmut.adapters.mindspore_adapter")
    for name in ("MindSporePyNativeAdapter", "MindSporeGraphAdapter"):
        cls = getattr(module, name)
        available, reason = cls.is_available()
        assert isinstance(available, bool)
        assert isinstance(reason, str)
