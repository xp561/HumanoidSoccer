"""Summarize per-motion R1 Isaac Lab tracking evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "E_gmpjpe_mm",
    "E_mpjpe_mm",
    "E_vel_mm_per_frame",
    "E_dof_mpjpe_mm",
    "AUJ_proxy_peak_link_jerk_deviation",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if len(payload["metrics"]) != 1:
            raise RuntimeError(f"Expected one environment in {path}")
        rows.append({"motion": payload["motion_files"][0], **payload["metrics"][0]})
    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 Standard motions, found {len(rows)}")
    with (args.output_dir / "per_motion.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "std": float(np.std([row[metric] for row in rows], ddof=1)),
            "min": float(np.min([row[metric] for row in rows])),
            "max": float(np.max([row[metric] for row in rows])),
        }
        for metric in METRICS
    }
    report = {
        "motions": len(rows),
        "aggregate": aggregate,
        "definition": {
            "E_gmpjpe_mm": "Existing repository global root-origin-aligned body-position error.",
            "E_mpjpe_mm": "Existing repository root-relative body-position error.",
            "E_vel_mm_per_frame": "Existing repository root finite-difference velocity error, in mm/frame.",
            "AUJ_proxy_peak_link_jerk_deviation": "Mean absolute difference of per-link peak third position differences; explicitly not asserted as the paper's AUJ.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# R1 Standard tracking evaluation", "", "| Metric | Mean | SD | Min | Max |", "|---|---:|---:|---:|---:|"]
    for metric in METRICS:
        item = aggregate[metric]
        lines.append(f"| {metric} | {item['mean']:.4f} | {item['std']:.4f} | {item['min']:.4f} | {item['max']:.4f} |")
    lines.extend(["", "The first three metrics use the original repository formulas. `AUJ_proxy_peak_link_jerk_deviation` remains separately named because the repository lacks the paper's reference-dataset AUJ implementation.", ""])
    (args.output_dir / "table_iii_tracking.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
