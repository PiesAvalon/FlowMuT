#!/usr/bin/env python3
"""Launch full FlowMuT campaigns across selected models and framework modes.

Each (model, mode) campaign runs in its own process.  Results are stored in a
fresh directory so a later run cannot silently overwrite earlier artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import flowmut.seed_models as seed_models  # noqa: E402
from scripts.run_smoke import find_ms_python  # noqa: E402


MODES = {
    "pytorch": ("eager", "compiled"),
    "mindspore": ("pynative", "graph"),
}


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run full FlowMuT campaigns (one process per model and mode).")
    p.add_argument("--framework", choices=("pytorch", "mindspore", "all"),
                   default="all", help="framework to test (default: all)")
    p.add_argument("--mode", choices=("all", "eager", "compiled", "pynative", "graph"),
                   default="all", help="execution mode within the selected framework")
    p.add_argument("--seeds", nargs="+", default=["all"],
                   help="seed model keys, or 'all' (default: all)")
    p.add_argument("--list-seeds", action="store_true", help="list seed model keys and exit")
    p.add_argument("--rounds", type=positive_int, default=250,
                   help="mutation rounds per model and mode (default: 250)")
    p.add_argument("--device", default="cpu", help="adapter device (default: cpu)")
    p.add_argument("--max-candidates", type=nonnegative_int, default=400,
                   help="candidates checked per round; 0 means no cap")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="framework execution timeout in seconds")
    p.add_argument("--seed-rng", type=int, default=20240929,
                   help="random seed for each campaign")
    p.add_argument("--reproduce-attempts", type=positive_int, default=1)
    p.add_argument("--torch-python", default=sys.executable,
                   help="Python interpreter with PyTorch installed")
    p.add_argument("--ms-python", default="",
                   help="Python interpreter with MindSpore installed; otherwise auto-discover")
    p.add_argument("--out", type=Path, default=None,
                   help="new output directory (default: runs/experiments/<timestamp>)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan without checking environments or running campaigns")
    return p


def selected_modes(framework: str, mode: str) -> list[tuple[str, str]]:
    frameworks = tuple(MODES) if framework == "all" else (framework,)
    selected = [(fw, name) for fw in frameworks for name in MODES[fw]
                if mode == "all" or mode == name]
    if not selected:
        raise ValueError(f"mode {mode!r} is not part of framework {framework!r}")
    return selected


def selected_seeds(names: list[str]) -> list[str]:
    known = seed_models.seed_keys()
    if names == ["all"]:
        return known
    if "all" in names:
        raise ValueError("'all' cannot be combined with individual seed models")
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"unknown seed model(s): {', '.join(unknown)}; use --list-seeds")
    if len(names) != len(set(names)):
        raise ValueError("duplicate seed model in --seeds")
    return names


def python_path(value: str) -> str:
    expanded = os.path.expanduser(value)
    resolved = shutil.which(expanded) if os.path.sep not in expanded else expanded
    if not resolved or not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
        raise ValueError(f"Python interpreter not found or not executable: {value!r}")
    return str(Path(resolved).resolve())


def check_mode(python: str, framework: str, mode: str, env: dict[str, str]) -> None:
    code = (
        "import sys; from flowmut.adapters.registry import adapter_class; "
        f"cls = adapter_class({framework!r}, {mode!r}); "
        "ok, reason = cls.is_available(); print(reason); sys.exit(0 if ok else 1)"
    )
    probe = subprocess.run([python, "-c", code], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    if probe.returncode != 0:
        detail = (probe.stdout + probe.stderr).strip()
        raise RuntimeError(f"{framework}:{mode} unavailable in {python}: {detail}")
    print(f"[ready] {framework}:{mode} in {python}: {probe.stdout.strip()}")


def write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def run_job(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return proc.wait()
        except KeyboardInterrupt:
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
            raise


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if seed_models.load_errors():
        p.error(f"seed model modules failed to load: {seed_models.load_errors()}")
    if args.list_seeds:
        for key in seed_models.seed_keys():
            print(key)
        return 0
    if args.timeout <= 0:
        p.error("--timeout must be positive")
    try:
        modes = selected_modes(args.framework, args.mode)
        seeds = selected_seeds(args.seeds)
    except ValueError as exc:
        p.error(str(exc))

    out = args.out or (ROOT / "runs" / "experiments" /
                       datetime.now().strftime("%Y%m%d-%H%M%S"))
    out = out.resolve()
    print(f"Models: {', '.join(seeds)}")
    print(f"Modes: {', '.join(f'{fw}:{mode}' for fw, mode in modes)}")
    print(f"Rounds: {args.rounds} per campaign ({len(seeds) * len(modes)} campaigns)")
    print(f"Output: {out}")
    if args.dry_run:
        return 0
    if out.exists():
        p.error(f"output path already exists: {out}; choose a fresh --out")

    try:
        interpreters = {
            "pytorch": python_path(args.torch_python) if any(fw == "pytorch" for fw, _ in modes) else "",
            "mindspore": python_path(args.ms_python or find_ms_python())
            if any(fw == "mindspore" for fw, _ in modes) else "",
        }
    except ValueError as exc:
        p.error(str(exc))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MS_DEV_DISABLE_PREBUILD", "1")
    for fw, mode in modes:
        try:
            check_mode(interpreters[fw], fw, mode, env)
        except RuntimeError as exc:
            p.error(str(exc))

    out.mkdir(parents=True)
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "framework": args.framework,
        "modes": [f"{fw}:{mode}" for fw, mode in modes],
        "seeds": seeds,
        "rounds": args.rounds,
        "device": args.device,
        "max_candidates": args.max_candidates,
        "timeout": args.timeout,
        "seed_rng": args.seed_rng,
        "reproduce_attempts": args.reproduce_attempts,
        "interpreters": {fw: interpreters[fw] for fw, _ in modes},
        "jobs": [],
    }
    manifest_path = out / "manifest.json"
    write_manifest(manifest_path, manifest)
    interrupted = False
    for fw, mode in modes:
        for seed in seeds:
            job_out = out / fw / mode / seed
            job_out.mkdir(parents=True)
            command = [interpreters[fw], "-u", "-m", "flowmut.cli", "run",
                       "--seed", seed, "--mode", f"{fw}:{mode}",
                       "--rounds", str(args.rounds), "--device", args.device,
                       "--max-candidates", str(args.max_candidates),
                       "--timeout", str(args.timeout),
                       "--seed-rng", str(args.seed_rng),
                       "--reproduce-attempts", str(args.reproduce_attempts),
                       "--out", str(job_out)]
            log_path = job_out / "run.log"
            report_path = job_out / f"campaign__{seed}__{fw}-{mode}.json"
            print(f"\n[run] {fw}:{mode} / {seed}", flush=True)
            job = {"framework": fw, "mode": mode, "seed": seed,
                   "command": command, "output": str(job_out.relative_to(out)),
                   "status": "running"}
            manifest["jobs"].append(job)
            write_manifest(manifest_path, manifest)
            try:
                returncode = run_job(command, log_path, env)
            except KeyboardInterrupt:
                job["status"] = "interrupted"
                interrupted = True
                write_manifest(manifest_path, manifest)
                break
            job["returncode"] = returncode
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    job["rounds_completed"] = report.get("rounds_completed", 0)
                    job["bugs_found"] = report.get("bugs_found", 0)
                    if report.get("error"):
                        job["error"] = report["error"]
                    elif job["rounds_completed"] != args.rounds:
                        job["error"] = (f"completed {job['rounds_completed']} of "
                                        f"{args.rounds} requested rounds")
                except (OSError, ValueError) as exc:
                    job["error"] = f"cannot read campaign report: {exc}"
            else:
                job["error"] = "campaign report missing"
            job["status"] = "passed" if returncode == 0 and not job.get("error") else "failed"
            print(f"[{job['status']}] {fw}:{mode} / {seed}", flush=True)
            write_manifest(manifest_path, manifest)
        if interrupted:
            break
    manifest["status"] = ("interrupted" if interrupted else
                          "failed" if any(j["status"] == "failed" for j in manifest["jobs"])
                          else "passed")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    by_mode = {}
    for job in manifest["jobs"]:
        key = f"{job['framework']}:{job['mode']}"
        entry = by_mode.setdefault(key, {"campaigns": 0, "passed": 0,
                                         "failed": 0, "rounds_completed": 0,
                                         "bug_reports": 0})
        entry["campaigns"] += 1
        entry["passed"] += job["status"] == "passed"
        entry["failed"] += job["status"] == "failed"
        entry["rounds_completed"] += job.get("rounds_completed", 0)
        entry["bug_reports"] += job.get("bugs_found", 0)
    (out / "summary.json").write_text(
        json.dumps({"status": manifest["status"], "by_mode": by_mode}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest: {manifest_path}")
    print(f"Summary: {out / 'summary.json'}")
    print(f"Campaigns: {sum(j['status'] == 'passed' for j in manifest['jobs'])} passed, "
          f"{sum(j['status'] == 'failed' for j in manifest['jobs'])} failed")
    return 130 if interrupted else 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
