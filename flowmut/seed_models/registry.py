"""Seed model registry.

The benchmark contains 14 seed models spanning ten application tasks.  Every
seed model is defined **once** as a framework-neutral :class:`~flowmut.ir.Graph`
built by a :class:`~flowmut.ir.GraphBuilder`; the framework/mode adapters turn
that graph into a runnable PyTorch or MindSpore program.  This is what lets the
same seed model run in all four evaluated modes without duplicating model code.

Each model module exposes a ``build() -> Graph`` function and registers itself
with :func:`register_seed`; importing :mod:`flowmut.seed_models` loads them all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from flowmut.adapters.base import Sample
from flowmut.ir.graph import Graph, GraphBuilder
from flowmut.ir.specs import TensorSpec, shape_numel


@dataclass
class SeedModel:
    """One benchmark seed model."""

    key: str
    task: str
    builder: Callable[[], Graph]
    #: Input edge name -> shape, in the order the adapter feeds them.
    input_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    input_dtypes: Dict[str, str] = field(default_factory=dict)
    #: Optional ``[low, high)`` sampling range per integer input (e.g. token ids).
    input_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    #: Reported parameter count (from the paper's benchmark table).
    param_count: int = 0
    #: Human-readable input size, e.g. ``"224x224"``.
    input_size: str = ""
    #: Source architecture.
    reference: str = ""
    #: ``True`` when the model needs integer indices rather than float tensors.
    notes: str = ""

    _graph: Optional[Graph] = None

    def graph(self) -> Graph:
        if self._graph is None:
            g = self.builder()
            if not g.name or g.name == "model":
                g.name = self.key
            g.meta.setdefault("seed", self.key)
            g.meta.setdefault("task", self.task)
            self._graph = g
        return self._graph

    def fresh_graph(self) -> Graph:
        g = self.builder()
        if not g.name or g.name == "model":
            g.name = self.key
        g.meta.setdefault("seed", self.key)
        g.meta.setdefault("task", self.task)
        g.meta["seed_model"] = self.key
        return g

    def to_dict(self) -> Dict[str, Any]:
        g = self.graph()
        return {
            "key": self.key,
            "task": self.task,
            "input_shapes": {k: list(v) for k, v in self.input_shapes.items()},
            "input_dtypes": dict(self.input_dtypes),
            "param_count": self.param_count,
            "input_size": self.input_size,
            "reference": self.reference,
            "graph": g.summary(),
        }


_REGISTRY: Dict[str, SeedModel] = {}
_ORDER: List[str] = []


def register_seed(seed: SeedModel) -> SeedModel:
    if seed.key in _REGISTRY:
        raise KeyError(f"duplicate seed model key {seed.key!r}")
    _REGISTRY[seed.key] = seed
    _ORDER.append(seed.key)
    return seed


def seed(key: str) -> SeedModel:
    if key not in _REGISTRY:
        raise KeyError(f"unknown seed model {key!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def all_seeds() -> List[SeedModel]:
    return [_REGISTRY[k] for k in _ORDER]


def seed_keys() -> List[str]:
    return list(_ORDER)


def seeds_by_task() -> Dict[str, List[SeedModel]]:
    out: Dict[str, List[SeedModel]] = {}
    for s in all_seeds():
        out.setdefault(s.task, []).append(s)
    return out


def build_graph(key: str) -> Graph:
    return seed(key).fresh_graph()


# ---------------------------------------------------------------------------
# Profiling inputs X_p
# ---------------------------------------------------------------------------

def make_sample(seed_model: SeedModel, batch_size: Optional[int] = None,
                rng: Optional[np.random.Generator] = None,
                scale: float = 1.0) -> Sample:
    """One deterministic profiling input for ``seed_model``."""
    rng = rng or np.random.default_rng(seed_model.param_count or 0)
    sample: Sample = {}
    for name, shape in seed_model.input_shapes.items():
        shape = tuple(shape)
        if batch_size is not None and shape:
            shape = (batch_size,) + shape[1:]
        dtype = seed_model.input_dtypes.get(name, "float32")
        if dtype in ("int64", "int32"):
            lo, hi = seed_model.input_ranges.get(name, (0, max(2, int(shape[-1]) if shape else 8)))
            arr = rng.integers(int(lo), max(int(lo) + 1, int(hi)), size=shape, dtype=np.int64)
            arr = arr.astype(dtype)
        elif dtype == "bool":
            arr = rng.integers(0, 2, size=shape).astype(bool)
        else:
            arr = (rng.standard_normal(shape) * scale).astype(dtype)
        sample[name] = arr
    return sample


def default_samples(seed_model: SeedModel, count: int = 2,
                    batch_sizes: Optional[Sequence[int]] = None) -> List[Sample]:
    """``X_p``: the profiling inputs used to annotate ``F(e)``.

    Different batch sizes are used on purpose so that edges whose shape depends
    on the batch dimension become polymorphic in the TFG.
    """
    sizes = list(batch_sizes) if batch_sizes else [1, 2][:max(1, count)]
    while len(sizes) < count:
        sizes.append(sizes[-1] + 1)
    rng = np.random.default_rng(abs(hash(seed_model.key)) % (2 ** 31))
    return [make_sample(seed_model, batch_size=sizes[i], rng=rng,
                        scale=1.0 if i % 2 == 0 else 0.5)
            for i in range(count)]


def input_specs(seed_model: SeedModel) -> List[TensorSpec]:
    """Declared specs of the seed model's graph inputs."""
    specs: List[TensorSpec] = []
    for name, shape in seed_model.input_shapes.items():
        specs.append(TensorSpec(shape=tuple(shape),
                                dtype=seed_model.input_dtypes.get(name, "float32"),
                                numel=shape_numel(shape)))
    return specs
