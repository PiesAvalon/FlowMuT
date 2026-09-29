"""FlowMuT command line interface.

Examples
--------
    python -m flowmut.cli modes
    python -m flowmut.cli seeds
    python -m flowmut.cli pool --composition
    python -m flowmut.cli tfg --seed convnext_v2_tiny --mode torch-eager
    python -m flowmut.cli run --seed convnext_v2_tiny --mode torch-eager --rounds 25
    python -m flowmut.cli smoke --rounds 4 --modes torch-eager mindspore-pynative
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Sequence

from flowmut.adapters.registry import available_modes, make_adapter_for
from flowmut.config import FlowMuTConfig
from flowmut.operators.pool import build_pool, pool_composition


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_modes(args: argparse.Namespace) -> int:
    rows = available_modes(probe=True)
    width = max(len(f"{r['framework']}:{r['mode']}") for r in rows)
    for row in rows:
        key = f"{row['framework']}:{row['mode']}"
        status = "available" if row.get("available") else "unavailable"
        print(f"{key:<{width}}  {status:<12} {row.get('reason', '')}")
    return 0


def cmd_seeds(args: argparse.Namespace) -> int:
    import flowmut.seed_models as sm
    errors = sm.load_errors()
    for key, message in errors.items():
        print(f"[warn] failed to load seed module {key}: {message}", file=sys.stderr)
    seeds = sm.all_seeds()
    if not seeds:
        print("no seed models registered", file=sys.stderr)
        return 1
    print(f"{'key':<24}{'task':<30}{'input':<16}{'nodes':>6}{'params':>8}{'edges':>7}")
    for model in seeds:
        graph = model.graph()
        actual_params = sum(int(e.spec.numel) for e in graph.edges.values()
                            if graph.nodes[e.producer].op == "param"
                            and e.spec is not None)
        print(f"{model.key:<24}{model.task:<30}{model.input_size:<16}"
              f"{len(graph.nodes):>6}{actual_params:>8}{len(graph.edges):>7}")
    if args.json:
        print(json.dumps([m.to_dict() for m in seeds], indent=2, default=str))
    return 0


def cmd_pool(args: argparse.Namespace) -> int:
    from flowmut.operators.pool import cg_operators, tfg_operators
    pool, report = build_pool(include_cg=not args.no_cg, include_tfg=not args.no_tfg,
                              deduplicate=not args.no_dedup)
    print(json.dumps(report.to_dict(), indent=2))
    print()
    print(json.dumps(pool.summary(), indent=2))
    if args.composition:
        print()
        comp = pool_composition(pool, cg_operators() if not args.no_cg else [],
                                tfg_operators() if not args.no_tfg else [])
        header = (f"{'object':<10}{'primitive':<11}{'CG':>4}{'TFG':>5}{'ovl':>5}"
                  f"{'only':>5}{'pool':>5}{'paper':>6}{'delta':>6}")
        print(header)
        print("-" * len(header))
        for row in comp["rows"]:
            print(f"{row['object']:<10}{row['primitive']:<11}"
                  f"{row['cg_retained']:>4}{row['tfg_derived']:>5}{row['overlap']:>5}"
                  f"{row['tfg_only']:>5}{row['pool']:>5}{row['paper_total']:>6}"
                  f"{row['delta']:>6}")
        print("-" * len(header))
        print(f"{'TOTAL':<21}{comp['cg_total']:>4}{comp['tfg_total']:>5}"
              f"{comp['overlap_total']:>5}{comp['tfg_only_total']:>5}"
              f"{comp['built_total']:>5}{comp['paper_total']:>6}"
              f"{comp['built_total'] - comp['paper_total']:>6}")
        print(f"matches the paper's construction table: {comp['matches_paper']}")
    if args.list:
        for op in pool:
            print(f"{op.target_object:<9} {op.primitive:<10} {op.source:<7} {op.name}")
    return 0


def cmd_tfg(args: argparse.Namespace) -> int:
    from flowmut.seed_models.registry import default_samples, seed as get_seed
    from flowmut.tfg import build_tfg
    model = get_seed(args.seed)
    graph = model.fresh_graph()
    adapter = make_adapter_for(args.mode, device=args.device, timeout_s=args.timeout)
    samples = default_samples(model, count=max(1, args.samples))
    bundle = adapter.materialize(graph)
    edge_specs: Dict[str, list] = {}
    for sample in samples:
        result = adapter.execute(graph, sample, module=bundle, capture_specs=True)
        if result.failed:
            print(f"profiling failed: {result.error_type}: {result.error_message}",
                  file=sys.stderr)
            print(result.error_traceback[-2000:], file=sys.stderr)
            return 2
        for eid, specs in result.edge_specs.items():
            edge_specs.setdefault(eid, []).extend(specs)
    tfg = build_tfg(graph, edge_specs)
    print(tfg.describe(max_nodes=args.max_nodes))
    print()
    print(json.dumps(tfg.summary(), indent=2, default=str))
    if args.json:
        payload = {
            "graph": {
                "nodes": [n.to_dict() for n in graph.nodes.values()],
                "edges": [e.to_dict() for e in graph.edges.values()],
                "inputs": list(graph.inputs),
                "outputs": list(graph.outputs),
            },
            "profiles": len(samples),
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        print(f"wrote {args.json}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from flowmut.operators.pool import validate_pool
    from flowmut.seed_models.registry import default_samples, seed as get_seed
    from flowmut.tfg import build_tfg
    pool, report = build_pool()
    print(f"pool size: {len(pool)} (cg={report.cg} tfg={report.tfg} overlap={len(report.overlaps)})")
    tfgs = []
    for key in args.seeds:
        model = get_seed(key)
        graph = model.fresh_graph()
        adapter = make_adapter_for(args.mode, device=args.device, timeout_s=args.timeout)
        bundle = adapter.materialize(graph)
        edge_specs: Dict[str, list] = {}
        for sample in default_samples(model, count=1):
            result = adapter.execute(graph, sample, module=bundle, capture_specs=True)
            if result.failed:
                continue
            for eid, specs in result.edge_specs.items():
                edge_specs.setdefault(eid, []).extend(specs)
        tfgs.append(build_tfg(graph, edge_specs))
    summary = validate_pool(pool, tfgs)
    summary.pop("results", None)
    print(json.dumps(summary, indent=2))
    return 0


def _config_from_args(args: argparse.Namespace) -> FlowMuTConfig:
    kwargs: Dict[str, Any] = {
        "mode": args.mode,
        "device": args.device,
        "rounds": args.rounds,
        "seed": args.seed_rng,
        "seed_models": list(args.seed or []),
        "profile_samples": args.profile_samples,
        "constraint_variant": args.constraints,
        "policy": args.policy,
        "policy_alpha": args.alpha,
        "feature_variant": args.features,
        "include_cg": not args.no_cg,
        "include_tfg": not args.no_tfg,
        "deduplicate": not args.no_dedup,
        "max_candidates": args.max_candidates,
        "timeout_s": args.timeout,
        "divergence_check": not args.no_divergence,
        "output_dir": args.out,
        "verbose": args.verbose,
        "reproduce_attempts": args.reproduce_attempts,
        "only_objects": args.objects or [],
        "only_operators": args.operators or [],
        "gradient_probe": not args.no_gradient,
    }
    if args.ablation:
        base = FlowMuTConfig.ablation(args.ablation)
        merged = base.to_dict()
        merged.update({k: v for k, v in kwargs.items()})
        return FlowMuTConfig.from_dict(merged)
    return FlowMuTConfig(**kwargs)


def cmd_run(args: argparse.Namespace) -> int:
    from flowmut.loop.campaign import run_campaigns
    config = _config_from_args(args)
    if args.dry_run:
        print(config.describe())
        return 0
    print(config.describe())
    if args.isolate:
        return _run_isolated(args, config)
    results = run_campaigns([config], seed_models=args.seed or [],
                            out_dir=args.out, verbose=args.verbose)
    for result in results:
        print()
        print(result.summary())
    print(f"\nartifacts written to {os.path.abspath(args.out)}")
    return 0 if all(not r.error for r in results) else 1


def _run_isolated(args: argparse.Namespace, config: FlowMuTConfig) -> int:
    """Run every campaign in its own interpreter.

    A mutation can crash a compiler backend hard (Inductor has been observed to
    abort the process on an out-of-bounds generated kernel).  Isolating each
    campaign keeps the remaining campaigns alive and turns the crash into a
    recorded result rather than a lost run.
    """
    import subprocess
    from flowmut.loop.campaign import select_seed_models
    os.makedirs(args.out, exist_ok=True)
    config_path = os.path.join(args.out, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(config.to_json())
    failures = 0
    for model in select_seed_models(args.seed):
        result_path = os.path.join(args.out, f"result__{model.key}.json")
        cmd = [sys.executable, "-m", "flowmut.cli", "_campaign",
               "--config", config_path, "--seed", model.key,
               "--result", result_path]
        print(f"[run ] {model.key} on {config.mode} (isolated)")
        proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        if proc.returncode != 0:
            failures += 1
            print(f"[fail] {model.key}: interpreter exited with {proc.returncode} "
                  f"-- a hard crash inside the framework is a FlowMuT finding")
        summary = _read_result_summary(result_path)
        if summary:
            print("      " + summary.replace("\n", "\n      "))
    print(f"\nartifacts written to {os.path.abspath(args.out)}")
    return 1 if failures else 0


def _read_result_summary(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return ""
    return (f"{payload.get('seed_model', '?')} [{payload.get('framework')}:"
            f"{payload.get('mode')}] rounds={payload.get('rounds_completed')} "
            f"bugs={payload.get('bugs_found', 0)}"
            + (f"  ERROR: {payload['error'][:200]}" if payload.get("error") else ""))


def cmd_campaign(args: argparse.Namespace) -> int:
    """Internal: run exactly one campaign (used by ``run --isolate``)."""
    from flowmut.loop.campaign import run_campaigns
    config = FlowMuTConfig.from_json(open(args.config, "r", encoding="utf-8").read())
    results = run_campaigns([config], seed_models=[args.seed], out_dir=args.out,
                            verbose=0)
    if not results:
        return 1
    result = results[0]
    if args.result:
        with open(args.result, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, default=str)
    return 0 if not result.error else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    from flowmut.smoke import run_smoke
    rc = run_smoke(modes=args.modes, rounds=args.rounds, seeds=args.seed,
                   out_dir=args.out, verbose=args.verbose,
                   max_candidates=args.max_candidates)
    return rc


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("flowmut",
                                     description="FlowMuT: data-flow-guided mutation testing for DL frameworks")
    parser.add_argument("--version", action="version", version="flowmut 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("modes", help="list framework/mode adapters and their availability")
    p.set_defaults(func=cmd_modes)

    p = sub.add_parser("seeds", help="list the 14 benchmark seed models")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_seeds)

    p = sub.add_parser("pool", help="build and report the mutation operator pool")
    p.add_argument("--composition", action="store_true", help="compare with the paper's table")
    p.add_argument("--list", action="store_true", help="list every operator")
    p.add_argument("--no-cg", action="store_true")
    p.add_argument("--no-tfg", action="store_true")
    p.add_argument("--no-dedup", action="store_true")
    p.set_defaults(func=cmd_pool)

    p = sub.add_parser("tfg", help="profile a seed model and print its TFG")
    p.add_argument("--seed", required=True)
    p.add_argument("--mode", default="torch-eager")
    p.add_argument("--device", default="cpu")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-nodes", type=int, default=60)
    p.add_argument("--json", default="")
    p.set_defaults(func=cmd_tfg)

    p = sub.add_parser("validate", help="validate the operator pool against profiled TFGs")
    p.add_argument("--seeds", nargs="*", default=["convnext_v2_tiny"])
    p.add_argument("--mode", default="torch-eager")
    p.add_argument("--device", default="cpu")
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=cmd_validate)

    for name, help_text in (("run", "run a FlowMuT campaign"),):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--seed", nargs="*", default=[], help="seed model keys (default: all)")
        p.add_argument("--mode", default="torch-eager")
        p.add_argument("--device", default="cpu")
        p.add_argument("--rounds", type=int, default=250)
        p.add_argument("--profile-samples", type=int, default=2)
        p.add_argument("--constraints", default="full")
        p.add_argument("--policy", default="linucb")
        p.add_argument("--alpha", type=float, default=1.0)
        p.add_argument("--features", default="full")
        p.add_argument("--ablation", default="")
        p.add_argument("--objects", nargs="*", default=[])
        p.add_argument("--operators", nargs="*", default=[])
        p.add_argument("--max-candidates", type=int, default=400,
                       help="cap the candidates handed to the constraint "
                            "filter each round; 0 filters every candidate")
        p.add_argument("--timeout", type=float, default=180.0)
        p.add_argument("--reproduce-attempts", type=int, default=1)
        p.add_argument("--seed-rng", type=int, default=20240929)
        p.add_argument("--no-cg", action="store_true")
        p.add_argument("--no-tfg", action="store_true")
        p.add_argument("--no-dedup", action="store_true")
        p.add_argument("--no-divergence", action="store_true")
        p.add_argument("--no-gradient", action="store_true")
        p.add_argument("--isolate", action="store_true",
                       help="run each campaign in its own interpreter so a hard "
                            "framework crash cannot abort the others")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--out", default="runs")
        p.add_argument("--verbose", type=int, default=1)
        p.set_defaults(func=cmd_run)
    # make --max-candidates 0 mean "no cap"
    p.set_defaults(max_candidates=None)

    p = sub.add_parser("_campaign", help=argparse.SUPPRESS)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--result", default="")
    p.add_argument("--out", default="runs")
    p.set_defaults(func=cmd_campaign)

    p = sub.add_parser("smoke", help="run the FlowMuT smoke test")
    p.add_argument("--modes", nargs="*", default=None)
    p.add_argument("--seed", nargs="*", default=None)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-candidates", type=int, default=120)
    p.add_argument("--out", default="runs/smoke")
    p.add_argument("--verbose", type=int, default=1)
    p.set_defaults(func=cmd_smoke)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "max_candidates", None) == 0:
        args.max_candidates = None
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
