"""Candidate features and bandit policies for FlowMuT selection.

Public API
----------
``FEATURE_NAMES`` / ``FEATURE_VARIANTS``
    The five context features and the RQ4 ablation configurations.
``CoverageTracker`` / ``FeatureExtractor`` / ``VariantFeatureExtractor``
    Coverage bookkeeping and candidate-to-context vector extraction.
``SelectionPolicy`` and the concrete policies
    ``LinUCB`` (the FlowMuT policy), ``UCB1``, ``RandomSearch``,
    ``FixedPriority``, ``OneShot``, ``CoverageGreedy``, ``ImpactGreedy``.
``POLICIES`` / ``make_policy``
    Registry and factory used by the main loop and experiment configs.
"""

from flowmut.selection.features import (  # noqa: F401
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURE_VARIANTS,
    CoverageTracker,
    FeatureExtractor,
    VariantFeatureExtractor,
)
from flowmut.selection.policies import (  # noqa: F401
    POLICIES,
    CoverageGreedy,
    FixedPriority,
    ImpactGreedy,
    LinUCB,
    OneShot,
    RandomSearch,
    SelectionPolicy,
    UCB1,
    make_policy,
)

__all__ = [
    # features
    "FEATURE_NAMES",
    "FEATURE_INDEX",
    "FEATURE_VARIANTS",
    "CoverageTracker",
    "FeatureExtractor",
    "VariantFeatureExtractor",
    # policies
    "SelectionPolicy",
    "LinUCB",
    "UCB1",
    "RandomSearch",
    "FixedPriority",
    "OneShot",
    "CoverageGreedy",
    "ImpactGreedy",
    "POLICIES",
    "make_policy",
]
