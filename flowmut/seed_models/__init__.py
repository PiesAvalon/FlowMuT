"""The 14 benchmark seed models.

Each model module defines its architecture once, framework-neutrally, and
registers a :class:`~flowmut.seed_models.registry.SeedModel`.
"""

from flowmut.seed_models.registry import (  # noqa: F401
    SeedModel,
    all_seeds,
    build_graph,
    default_samples,
    make_sample,
    register_seed,
    seed,
    seed_keys,
    seeds_by_task,
)

# Importing the model modules populates the registry.  Failures are collected
# rather than raised so that a partial installation still yields a usable zoo.
_LOAD_ERRORS: dict = {}

for _mod in (
    "vision_classification",
    "vision_detection",
    "vision_segmentation",
    "text_models",
    "vision_misc",
    "generative",
):
    try:
        __import__(f"flowmut.seed_models.{_mod}")
    except Exception as exc:  # pragma: no cover - reported through load_errors()
        _LOAD_ERRORS[_mod] = f"{type(exc).__name__}: {exc}"


def load_errors() -> dict:
    return dict(_LOAD_ERRORS)


__all__ = [
    "SeedModel", "register_seed", "seed", "all_seeds", "seed_keys",
    "seeds_by_task", "build_graph", "default_samples", "make_sample",
    "load_errors",
]
