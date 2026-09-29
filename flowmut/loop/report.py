"""Campaign reporting: JSON artifacts and a Markdown summary."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from flowmut.loop.engine import CampaignResult


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def write_campaign(result: CampaignResult, out_dir: str,
                   include_graph: bool = False) -> str:
    """Persist one campaign as JSON and return the path."""
    os.makedirs(out_dir, exist_ok=True)
    name = _safe(f"{result.seed_model}__{result.framework}-{result.mode}")
    path = os.path.join(out_dir, f"campaign__{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(with_graph=include_graph), handle,
                  indent=2, sort_keys=True, default=str)
    return path


def write_summary(results: Sequence[CampaignResult], out_dir: str) -> str:
    """Write an aggregate JSON summary plus a Markdown report."""
    os.makedirs(out_dir, exist_ok=True)
    payload = aggregate(results)
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(results, payload))
    return json_path


def aggregate(results: Sequence[CampaignResult]) -> Dict[str, Any]:
    by_mode: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"campaigns": 0, "rounds": 0, "bugs": 0, "legal_mutants": 0,
                 "generated": 0, "retained": 0, "duration_s": 0.0,
                 "checkpoints": {}})
    bugs: List[Dict[str, Any]] = []
    for result in results:
        key = f"{result.framework}:{result.mode}"
        entry = by_mode[key]
        entry["campaigns"] += 1
        entry["rounds"] += result.rounds_completed
        entry["bugs"] += result.bugs_found
        entry["legal_mutants"] += sum(1 for i in result.iterations
                                      if i.legal and i.status == "ok")
        entry["generated"] += result.candidate_statistics.get("total_generated", 0)
        entry["retained"] += result.candidate_statistics.get("total_retained", 0)
        entry["duration_s"] += result.duration_s
        for checkpoint, values in (result.checkpoints or {}).items():
            entry["checkpoints"].setdefault(checkpoint, []).append(values)
        for bug in result.bugs:
            record = bug.to_dict() if hasattr(bug, "to_dict") else dict(bug)
            record.setdefault("seed_model", result.seed_model)
            record.setdefault("mode", f"{result.framework}:{result.mode}")
            bugs.append(record)
    for key, entry in by_mode.items():
        for checkpoint, rows in entry["checkpoints"].items():
            keys = {k for row in rows for k in row}
            entry["checkpoints"][checkpoint] = {
                k: sum(row.get(k, 0.0) for row in rows) / max(1, len(rows))
                for k in sorted(keys) if isinstance(rows[0].get(k), (int, float))
            }
    return {
        "campaigns": len(results),
        "by_mode": dict(by_mode),
        "bugs": bugs,
        "distinct_bugs": len({(b.get("symptom"), b.get("seed_model"), b.get("operator"))
                              for b in bugs}),
        "symptoms": dict(Counter(b.get("symptom", "?") for b in bugs)),
        "errors": [{"seed_model": r.seed_model, "mode": f"{r.framework}:{r.mode}",
                    "error": r.error} for r in results if r.error],
    }


def render_markdown(results: Sequence[CampaignResult],
                    payload: Optional[Dict[str, Any]] = None) -> str:
    payload = payload or aggregate(results)
    lines: List[str] = ["# FlowMuT campaign report", ""]
    lines.append(f"- campaigns: **{payload['campaigns']}**")
    lines.append(f"- distinct bugs: **{payload['distinct_bugs']}**")
    if payload["symptoms"]:
        lines.append("- symptoms: " + ", ".join(f"{k}={v}"
                                                for k, v in sorted(payload["symptoms"].items())))
    lines.append("")

    lines.append("## Per mode")
    lines.append("")
    lines.append("| mode | campaigns | rounds | legal mutants | generated | retained | retention | bugs | time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, entry in sorted(payload["by_mode"].items()):
        generated = entry["generated"] or 0
        retained = entry["retained"] or 0
        retention = (retained / generated) if generated else 0.0
        lines.append(f"| {key} | {entry['campaigns']} | {entry['rounds']} | "
                     f"{entry['legal_mutants']} | {generated} | {retained} | "
                     f"{retention:.3f} | {entry['bugs']} | {entry['duration_s']:.1f} |")
    lines.append("")

    checkpoints = sorted({int(c) for r in results for c in (r.checkpoints or {})})
    if checkpoints:
        lines.append("## Diversity and coverage at checkpoints")
        lines.append("")
        header = "| mode | round | LIC | LPC | LSC | DFSD | distinct signatures |"
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|")
        for key, entry in sorted(payload["by_mode"].items()):
            for checkpoint in checkpoints:
                row = entry["checkpoints"].get(str(checkpoint)) or \
                    entry["checkpoints"].get(checkpoint)
                if not row:
                    continue
                lines.append(f"| {key} | {checkpoint} | {row.get('lic', 0):.4f} | "
                             f"{row.get('lpc', 0):.4f} | {row.get('lsc', 0):.4f} | "
                             f"{row.get('dfsd', 0):.4f} | "
                             f"{row.get('distinct_signatures', 0)} |")
        lines.append("")

    lines.append("## Legality (RQ3)")
    lines.append("")
    lines.append("| seed model | mode | variant | generated | retained | retention | rejected (input/internal/output/structural) |")
    lines.append("|---|---|---|---|---|---|---|")
    for result in results:
        legality = result.legality or {}
        rejected = legality.get("rejected_by_kind", {})
        lines.append(
            f"| {result.seed_model} | {result.framework}:{result.mode} | "
            f"{legality.get('constraint_variant', 'full')} | "
            f"{legality.get('generated', 0)} | {legality.get('retained', 0)} | "
            f"{legality.get('retention_rate', 0.0):.3f} | "
            f"{rejected.get('input', 0)}/{rejected.get('internal', 0)}/"
            f"{rejected.get('output', 0)}/{rejected.get('structural', 0)} |")
    lines.append("")

    lines.append("## Campaigns")
    lines.append("")
    for result in results:
        lines.append("```")
        lines.append(result.summary())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
