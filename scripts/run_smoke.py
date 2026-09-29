#!/usr/bin/env python3
"""Run the FlowMuT smoke test across every framework/mode adapters.

PyTorch and MindSpore live in *different* Python interpreters (PyTorch 2.12.1 on
Python 3.13 and MindSpore 2.7.1 on Python 3.9), which is exactly how the paper's
implementation is deployed.  This driver therefore runs the smoke test once per
interpreter and merges the reports:

    python scripts/run_smoke.py                 # both frameworks, all four modes
    python scripts/run_smoke.py --torch-only
    python scripts/run_smoke.py --rounds 5 --seeds convnext_v2_tiny

The MindSpore interpreter is discovered from ``--ms-python``, the
``FLOWMUT_MS_PYTHON`` environment variable, or a list of well-known conda
environments.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MS_ENV_CANDIDATES = (
    "mindspore-gpu-py39",
    "lumosmut-ms-py312",
    "rondo-ms29-cpu-py312",
    "mindspore-gpu-legacy",
)


def find_ms_python(explicit: str = "") -> str:
    if explicit:
        return explicit
    env = os.environ.get("FLOWMUT_MS_PYTHON", "")
    if env:
        return env
    for env_name in MS_ENV_CANDIDATES:
        candidate = os.path.expanduser(f"~/miniconda3/envs/{env_name}/bin/python")
        if os.path.exists(candidate):
            probe = subprocess.run(
                [candidate, "-c", "import mindspore; print(mindspore.__version__)"],
                capture_output=True, text=True)
            if probe.returncode == 0:
                return candidate
    return ""


def run_one(python: str, modes, seeds, rounds: int, max_candidates: int,
            out_dir: str, env_extra=None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MS_DEV_DISABLE_PREBUILD", "1")
    if env_extra:
        env.update(env_extra)
    payload = {
        "modes": list(modes), "seeds": list(seeds), "rounds": rounds,
        "max_candidates": max_candidates, "out_dir": out_dir,
    }
    code = (
        "import json, sys;"
        "sys.path.insert(0, %r);"
        "from flowmut.smoke import run_smoke;"
        "cfg = json.loads(sys.argv[1]);"
        "rc = run_smoke(modes=cfg['modes'] or None, seeds=cfg['seeds'] or None,"
        "               rounds=cfg['rounds'], max_candidates=cfg['max_candidates'],"
        "               out_dir=cfg['out_dir']);"
        "sys.exit(rc)"
    ) % ROOT
    print(f"\n### {python}\n", flush=True)
    proc = subprocess.run([python, "-c", code, json.dumps(payload)], env=env,
                          cwd=ROOT)
    report_path = os.path.join(out_dir, "smoke_report.json")
    report = {}
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception:
            report = {}
    return {"python": python, "returncode": proc.returncode, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description="FlowMuT smoke test driver")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--seeds", nargs="*", default=[])
    parser.add_argument("--out", default=os.path.join(ROOT, "runs", "smoke"))
    parser.add_argument("--ms-python", default="")
    parser.add_argument("--torch-python", default=sys.executable)
    parser.add_argument("--torch-only", action="store_true")
    parser.add_argument("--mindspore-only", action="store_true")
    args = parser.parse_args()

    torch_modes = ["torch-eager", "torch-compiled"]
    ms_modes = ["ms-pynative", "ms-graph"]

    runs = []
    if not args.mindspore_only:
        runs.append((args.torch_python, torch_modes))
    if not args.torch_only:
        ms_python = find_ms_python(args.ms_python)
        if ms_python:
            runs.append((ms_python, ms_modes))
        else:
            print("[warn] no MindSpore interpreter found; skipping MindSpore modes.",
                  file=sys.stderr)
            print("       pass --ms-python /path/to/python or set FLOWMUT_MS_PYTHON.",
                  file=sys.stderr)

    # One subprocess per mode: a mutation that crashes a compiler backend must
    # not take the rest of the smoke test with it (and the crash is a finding).
    def normalise(name: str) -> str:
        try:
            from flowmut.config import FlowMuTConfig
            return FlowMuTConfig(mode=name).mode
        except Exception:
            return name

    merged_modes: dict = {}
    results = []
    for python, modes in runs:
        for alias in modes:
            mode = normalise(alias)
            family = "pytorch" if mode.startswith("pytorch") else "mindspore"
            per_mode_out = os.path.join(args.out, family, mode.replace(":", "-"))
            run = run_one(python, [mode], args.seeds, args.rounds,
                          args.max_candidates, per_mode_out)
            results.append(run)
            merged_modes.update(run["report"].get("modes", {}))
            if run["returncode"] not in (0, 3) and mode not in merged_modes:
                merged_modes[mode] = {
                    "passed": False,
                    "checks": [{"name": "process", "ok": False,
                                "detail": f"interpreter exited with {run['returncode']} "
                                          f"(a hard crash inside the framework is reported "
                                          f"as a FlowMuT finding)"}],
                }

    merged = {
        "runs": [{"python": r["python"], "returncode": r["returncode"]}
                 for r in results],
        "modes": merged_modes,
        "seeds": args.seeds,
        "rounds": args.rounds,
    }
    os.makedirs(args.out, exist_ok=True)
    merged_path = os.path.join(args.out, "smoke_all.json")
    with open(merged_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, default=str)

    print("\n" + "=" * 66)
    print("FlowMuT smoke summary")
    print("=" * 66)
    failed = 0
    ran = 0
    for mode, entry in sorted(merged["modes"].items()):
        if entry.get("skipped"):
            print(f"  SKIP  {mode:<20} {entry.get('reason', '')}")
            continue
        ran += 1
        status = "OK  " if entry.get("passed") else "FAIL"
        if not entry.get("passed"):
            failed += 1
        print(f"  {status}  {mode:<20} "
              f"{sum(1 for c in entry.get('checks', []) if c['ok'])}"
              f"/{len(entry.get('checks', []))} checks")
    print(f"\nmerged report: {merged_path}")
    if failed or any(r["returncode"] not in (0, 3) for r in results):
        print("SMOKE TEST FAILED")
        return 1
    if not ran:
        print("SMOKE TEST INCONCLUSIVE: no mode ran")
        return 3
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
