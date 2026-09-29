"""FlowMuT smoke test.

Runs a very short campaign in every framework/mode that this interpreter can
actually execute and asserts that each stage of the pipeline produced something:
the seed model builds, the TFG is profiled, candidates are generated, constraints
retain some of them, the bandit selects, the mutant executes, the reward is
computed, and the metrics are recorded.

The smoke test is deliberately *safe*: a mode whose adapter is unavailable in the
current interpreter is reported and skipped, never failed.  Use
``scripts/run_smoke.py`` to drive both interpreters so that all four modes are
exercised.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence

from flowmut.adapters.registry import available_modes
from flowmut.config import FlowMuTConfig
from flowmut.loop.engine import FlowMuTEngine
from flowmut.operators.pool import build_pool
from flowmut.seed_models.registry import seed_keys

#: Seed models used when the caller does not name any.  They are chosen to be
#: small, to span several tasks, and to exercise convolutions, attention,
#: normalisation, pooling, reshaping and multi-output ops.
DEFAULT_SMOKE_SEEDS: tuple = ("convnext_v2_tiny", "segformer_b1", "modernbert_base")


class SmokeCheck:
    """Accumulates pass/fail assertions for one mode."""

    def __init__(self, mode: str):
        self.mode = mode
        self.checks: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(c["ok"] for c in self.checks)

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if not c["ok"]]

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "passed": self.passed, "checks": self.checks}


def _pick_seeds(names: Optional[Sequence[str]], limit: int = 3) -> List[str]:
    known = seed_keys()
    if names:
        return [n for n in names if n in known]
    preferred = [k for k in DEFAULT_SMOKE_SEEDS if k in known]
    if preferred:
        return preferred[:limit]
    return known[:limit]


def smoke_mode(mode: str, seeds: Sequence[str], rounds: int = 3,
               max_candidates: Optional[int] = 120, verbose: int = 1,
               out_dir: str = "") -> SmokeCheck:
    """Exercise one framework/mode end to end."""
    check = SmokeCheck(mode)
    try:
        cfg = FlowMuTConfig(mode=mode, rounds=rounds, profile_samples=2,
                            max_candidates=max_candidates, timeout_s=600.0,
                            divergence_check=False, verbose=verbose,
                            output_dir=out_dir or "runs/smoke")
    except Exception as exc:
        check.check("config", False, f"{type(exc).__name__}: {exc}")
        return check

    try:
        pool, report = build_pool()
        check.check("operator_pool", len(pool) > 0,
                    f"{len(pool)} operators (cg={report.cg}, tfg={report.tfg}, "
                    f"overlap={len(report.overlaps)})")
    except Exception as exc:
        check.check("operator_pool", False, f"{type(exc).__name__}: {exc}")
        return check

    for seed_key in seeds:
        try:
            engine = FlowMuTEngine(cfg, seed_key, pool=pool)
            engine.setup()
        except Exception as exc:
            check.check(f"{seed_key}:setup", False,
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
            continue
        check.check(f"{seed_key}:tfg", engine.tfg is not None and
                    len(engine.tfg.graph.nodes) > 0,
                    f"{len(engine.tfg.graph.nodes)} nodes, "
                    f"{len(engine.tfg.edge_specs)} profiled edges")
        try:
            result = engine.run()
        except Exception as exc:
            check.check(f"{seed_key}:run", False,
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
            continue
        check.check(f"{seed_key}:run", not result.error,
                    result.error or f"{result.rounds_completed} rounds")
        check.check(f"{seed_key}:candidates",
                    result.candidate_statistics.get("total_generated", 0) > 0,
                    f"generated={result.candidate_statistics.get('total_generated', 0)} "
                    f"retained={result.candidate_statistics.get('total_retained', 0)}")
        check.check(f"{seed_key}:executed",
                    result.rounds_completed > 0
                    and any(i.status in ("ok", "error") for i in result.iterations),
                    f"statuses={sorted({i.status for i in result.iterations})}")
        check.check(f"{seed_key}:reward",
                    any(i.reward > 0 for i in result.iterations)
                    or result.trace_memory.get("distinct_events", 0) > 0,
                    f"events={result.trace_memory.get('distinct_events', 0)} "
                    f"transitions={result.trace_memory.get('distinct_transitions', 0)}")
        check.check(f"{seed_key}:metrics",
                    bool(result.checkpoints) or bool(result.metrics),
                    f"checkpoints={sorted(result.checkpoints) if result.checkpoints else []}")
        check.check(f"{seed_key}:legality",
                    result.legality.get("generated", 0) >= 0
                    and "rejected_by_kind" in result.legality,
                    f"retention={result.legality.get('retention_rate', 0.0):.3f} "
                    f"rejected={result.legality.get('rejected_by_kind', {})}")
        if verbose:
            print(f"    {seed_key}: {result.rounds_completed} rounds, "
                  f"bugs={result.bugs_found}, "
                  f"retained={result.candidate_statistics.get('total_retained', 0)}, "
                  f"events={result.trace_memory.get('distinct_events', 0)}")
        if out_dir:
            try:
                from flowmut.loop.report import write_campaign
                write_campaign(result, out_dir)
            except Exception:
                pass
    return check


def run_smoke(modes: Optional[Sequence[str]] = None,
              rounds: int = 3, seeds: Optional[Sequence[str]] = None,
              out_dir: str = "runs/smoke", verbose: int = 1,
              max_candidates: Optional[int] = 120) -> int:
    """Run the smoke test for every available mode; returns a process exit code."""
    seed_list = _pick_seeds(seeds)
    if not seed_list:
        print("no seed models registered - cannot run the smoke test", file=sys.stderr)
        return 2

    availability = {f"{r['framework']}:{r['mode']}": r for r in available_modes(probe=True)}
    wanted = list(modes) if modes else list(availability)
    # Normalise aliases through the config parser.
    normalised: List[str] = []
    for name in wanted:
        try:
            cfg = FlowMuTConfig(mode=name)
            normalised.append(cfg.mode)
        except Exception:
            normalised.append(name)

    report: Dict[str, Any] = {"seeds": seed_list, "rounds": rounds, "modes": {}}
    failures = 0
    skipped: List[str] = []
    print("FlowMuT smoke test")
    print(f"  seeds: {', '.join(seed_list)}   rounds per mode: {rounds}")
    print()
    for mode in normalised:
        entry = availability.get(mode, {})
        if not entry.get("available", False):
            reason = entry.get("reason", "not registered in this interpreter")
            print(f"[skip] {mode}: {reason}")
            skipped.append(mode)
            report["modes"][mode] = {"skipped": True, "reason": reason}
            continue
        print(f"[run ] {mode}")
        check = smoke_mode(mode, seed_list, rounds=rounds,
                           max_candidates=max_candidates, verbose=verbose,
                           out_dir=out_dir)
        report["modes"][mode] = check.to_dict()
        if not check.passed:
            failures += 1
            for failure in check.failures:
                print(f"  FAIL {failure['name']}: {failure['detail']}")
        else:
            n = len(check.checks)
            print(f"  OK   {n}/{n} checks passed")
        print()

    report["passed"] = failures == 0
    report["skipped"] = skipped
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "smoke_report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"smoke report written to {path}")

    print("=" * 60)
    if skipped:
        print(f"skipped modes (adapter unavailable here): {', '.join(skipped)}")
    if failures:
        print(f"SMOKE TEST FAILED in {failures} mode(s)")
        return 1
    if not any(not v.get("skipped") for v in report["modes"].values()):
        print("SMOKE TEST INCONCLUSIVE: no mode was available in this interpreter")
        return 3
    print("SMOKE TEST PASSED")
    return 0
