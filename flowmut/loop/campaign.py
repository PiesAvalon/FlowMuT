"""Campaign drivers.

A *campaign* is one ``(seed model, framework, mode)`` run of the FlowMuT loop.
The experiment design of the paper runs a 12-hour bug-discovery campaign for each
of the 14 seed models on each framework, and a comparative evaluation of 10
independent runs x 250 mutation rounds for every model/framework pair.  Both are
produced here from the same :class:`FlowMuTEngine` by varying only the
configuration.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from flowmut.adapters.registry import available_modes
from flowmut.config import FlowMuTConfig
from flowmut.loop.engine import CampaignResult, FlowMuTEngine
from flowmut.operators.base import OperatorPool
from flowmut.operators.pool import build_pool
from flowmut.seed_models.registry import SeedModel, all_seeds, seed_keys


def select_seed_models(keys: Optional[Sequence[str]] = None) -> List[SeedModel]:
    """Resolve a seed-model key list (empty/``None`` selects the whole zoo)."""
    if not keys:
        return all_seeds()
    wanted = list(keys)
    known = set(seed_keys())
    unknown = [k for k in wanted if k not in known]
    if unknown:
        raise KeyError(f"unknown seed models {unknown}; available: {sorted(known)}")
    index = {s.key: s for s in all_seeds()}
    return [index[k] for k in wanted]


@dataclass
class CampaignPlan:
    """A set of campaigns to execute."""

    configs: List[FlowMuTConfig] = field(default_factory=list)
    seed_models: List[str] = field(default_factory=list)
    out_dir: str = "runs"
    #: Skip a configuration entirely when its adapter is unavailable.
    skip_unavailable: bool = True

    def cells(self) -> List[Any]:
        models = select_seed_models(self.seed_models)
        return [(cfg, m) for cfg in self.configs for m in models]


class CampaignRunner:
    """Runs campaigns sequentially, sharing the operator pool across them."""

    def __init__(self, plan: CampaignPlan, verbose: int = 1):
        self.plan = plan
        self.verbose = verbose
        self._pools: Dict[Any, OperatorPool] = {}
        self.results: List[CampaignResult] = []

    def pool_for(self, config: FlowMuTConfig) -> OperatorPool:
        key = (config.include_cg, config.include_tfg, config.deduplicate)
        if key not in self._pools:
            pool, _report = build_pool(include_cg=config.include_cg,
                                       include_tfg=config.include_tfg,
                                       deduplicate=config.deduplicate)
            self._pools[key] = pool
        return self._pools[key]

    def availability(self) -> List[Dict[str, Any]]:
        return available_modes(probe=True)

    def run(self) -> List[CampaignResult]:
        available = {f"{m['framework']}:{m['mode']}": m.get("available", False)
                     for m in self.availability()}
        for config, model in self.plan.cells():
            if config.mode not in available and self.plan.skip_unavailable:
                if self.verbose:
                    print(f"[skip] {config.mode} unavailable in this interpreter")
                continue
            if self.verbose:
                print(f"[run ] {model.key} on {config.mode} "
                      f"({config.rounds} rounds, policy={config.policy}, "
                      f"constraints={config.constraint_variant})")
            engine = FlowMuTEngine(config, model.key, pool=self.pool_for(config))
            started = time.time()
            try:
                result = engine.run()
            except Exception as exc:  # a campaign must never abort the driver
                result = CampaignResult(seed_model=model.key, framework=config.framework,
                                        mode=config.mode_name,
                                        rounds_requested=config.rounds,
                                        error=f"{type(exc).__name__}: {exc}",
                                        config=config.to_dict())
                result.duration_s = time.time() - started
            self.results.append(result)
            if self.verbose:
                print("      " + result.summary().replace("\n", "\n      "))
        return self.results


def run_campaigns(configs: Sequence[FlowMuTConfig],
                  seed_models: Sequence[str] = (),
                  out_dir: str = "runs",
                  verbose: int = 1,
                  write: bool = True) -> List[CampaignResult]:
    """Convenience entry point used by the CLI and by the smoke tests."""
    from flowmut.loop.report import write_campaign, write_summary
    plan = CampaignPlan(configs=list(configs), seed_models=list(seed_models),
                        out_dir=out_dir)
    runner = CampaignRunner(plan, verbose=verbose)
    results = runner.run()
    if write and out_dir:
        for result in results:
            write_campaign(result, out_dir)
        write_summary(results, out_dir)
    return results
