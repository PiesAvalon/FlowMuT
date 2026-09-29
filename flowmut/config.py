"""FlowMuT configuration.

One dataclass describes a whole campaign: which framework/mode, which seed
models, which operator-pool subset, which constraint and selection variants, and
the iteration budget.  The ablations of Section "Research Questions" are
expressed as first-class options here rather than as separate code paths, so the
complete FlowMuT and its variants share one implementation.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

from flowmut.ir.specs import MODES, parse_mode
from flowmut.operators.constraints import CONSTRAINT_VARIANTS

DEFAULT_CHECKPOINTS: Tuple[int, ...] = (50, 100, 150, 200, 250)


@dataclass
class FlowMuTConfig:
    """A single FlowMuT campaign configuration."""

    # -- target ------------------------------------------------------------
    mode: str = "torch-eager"
    device: str = "cpu"

    # -- budget ------------------------------------------------------------
    rounds: int = 250
    time_budget_s: Optional[float] = None
    seed: int = 20240929

    # -- seed models -------------------------------------------------------
    seed_models: List[str] = field(default_factory=list)      # empty == all
    profile_samples: int = 2
    sample_batch_sizes: List[int] = field(default_factory=lambda: [1, 2])

    # -- operator pool -----------------------------------------------------
    include_cg: bool = True
    include_tfg: bool = True
    deduplicate: bool = True
    only_objects: List[str] = field(default_factory=list)
    only_operators: List[str] = field(default_factory=list)
    max_candidates: Optional[int] = None

    # -- constraints (RQ3) -------------------------------------------------
    constraint_variant: str = "full"

    # -- selection (RQ4) ---------------------------------------------------
    policy: str = "linucb"
    policy_alpha: float = 1.0
    feature_variant: str = "full"
    gradient_probe: bool = True

    # -- oracle ------------------------------------------------------------
    divergence_check: bool = True
    reference_mode: str = ""          # empty == the framework's eager/pynative mode
    tolerance: float = 1e-3
    relative_tolerance: float = 1e-4
    reproduce_attempts: int = 1
    timeout_s: Optional[float] = 180.0
    max_consecutive_failures: int = 15

    # -- reporting ---------------------------------------------------------
    checkpoint_rounds: List[int] = field(default_factory=lambda: list(DEFAULT_CHECKPOINTS))
    output_dir: str = "runs"
    run_name: str = ""
    verbose: int = 1
    save_mutants: bool = False
    save_final_graph: bool = True

    # -- derived -----------------------------------------------------------
    def __post_init__(self) -> None:
        self.framework, self.mode_name = parse_mode(self.mode)
        self.mode = f"{self.framework}:{self.mode_name}"
        if self.constraint_variant not in CONSTRAINT_VARIANTS:
            raise ValueError(f"unknown constraint_variant {self.constraint_variant!r}")
        if not self.profile_samples:
            self.profile_samples = 1
        if not self.run_name:
            self.run_name = f"{self.framework}-{self.mode_name}"

    # -- helpers -----------------------------------------------------------
    @property
    def effective_reference_mode(self) -> Optional[str]:
        """The semantically equivalent mode results are compared against."""
        if not self.divergence_check:
            return None
        if self.reference_mode:
            fw, name = parse_mode(self.reference_mode)
            return f"{fw}:{name}"
        if self.framework == "pytorch" and self.mode_name == "compiled":
            return "pytorch:eager"
        if self.framework == "mindspore" and self.mode_name == "graph":
            return "mindspore:pynative"
        return None

    def checkpoint_set(self) -> List[int]:
        return sorted({int(c) for c in self.checkpoint_rounds if int(c) <= self.rounds})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowMuTConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> "FlowMuTConfig":
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        lines = [
            f"FlowMuT config: {self.framework}:{self.mode_name} on {self.device}",
            f"  rounds={self.rounds} seed={self.seed} profile_samples={self.profile_samples}",
            f"  pool: cg={self.include_cg} tfg={self.include_tfg} dedup={self.deduplicate}"
            + (f" objects={self.only_objects}" if self.only_objects else "")
            + (f" operators={len(self.only_operators)}" if self.only_operators else ""),
            f"  constraints={self.constraint_variant} policy={self.policy}"
            f"(alpha={self.policy_alpha}) features={self.feature_variant}",
            f"  oracle: divergence={self.effective_reference_mode} "
            f"tol={self.tolerance} reproduce={self.reproduce_attempts}",
            f"  seed models: {self.seed_models or 'ALL'}",
        ]
        return "\n".join(lines)

    # -- preset ablation variants -----------------------------------------
    @classmethod
    def ablation(cls, name: str, **overrides: Any) -> "FlowMuTConfig":
        """Named configurations used by the RQ3/RQ4 analyses."""
        presets: Dict[str, Dict[str, Any]] = {
            "full": {},
            "without_constraints": {"constraint_variant": "without_constraints"},
            "without_input": {"constraint_variant": "without_input"},
            "without_internal": {"constraint_variant": "without_internal"},
            "without_output": {"constraint_variant": "without_output"},
            "without_linucb": {"policy": "random"},
            "random_search": {"policy": "random"},
            "fixed_priority": {"policy": "fixed"},
            "one_shot": {"policy": "one_shot"},
            "ucb1": {"policy": "ucb1"},
            "without_site": {"feature_variant": "without_site"},
            "without_tensor": {"feature_variant": "without_tensor"},
            "without_operator": {"feature_variant": "without_operator"},
            "without_downstream": {"feature_variant": "without_downstream"},
            "without_gradient": {"feature_variant": "without_gradient"},
            "coverage_only": {"feature_variant": "coverage_only"},
            "impact_only": {"feature_variant": "impact_only"},
            "cg_only": {"include_tfg": False},
            "tfg_only": {"include_cg": False},
        }
        if name not in presets:
            raise KeyError(f"unknown ablation {name!r}; expected {sorted(presets)}")
        cfg = dict(presets[name])
        cfg.update(overrides)
        return cls(**cfg)


def default_configs() -> List[FlowMuTConfig]:
    """One configuration per evaluated framework/mode."""
    return [FlowMuTConfig(mode=f"{fw}-{mode}") for fw, mode in MODES]


def load_config(path: str) -> FlowMuTConfig:
    with open(path, "r", encoding="utf-8") as handle:
        return FlowMuTConfig.from_json(handle.read())
