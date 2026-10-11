"""Build paper-aligned soccer evaluation tables from vectorized Isaac Lab batches."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NAME = re.compile(r"f_(-?\d+\.\d+)_l_(-?\d+\.\d+)\.json$")


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return center - half_width, center + half_width


def read_directory(directory: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("f_*_l_*.json")):
        match = NAME.match(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text())
        trials = int(payload["trials"])
        successes = round(float(payload["success_goal_rate"]) * trials)
        rows.append({
            "condition": label,
            "forward_m": float(match.group(1)),
            "lateral_m": float(match.group(2)),
            "trials": trials,
            "successes": successes,
            "success_goal_rate": successes / trials,
            "contact_rate": float(payload["first_contact_rate"]),
            "accuracy_cosine": payload["kick_accuracy_cosine"],
            "accuracy_valid_trials": int(payload["kick_accuracy_valid_trials"]),
            "max_ball_speed": float(payload["max_ball_speed"]),
        })
    if len(rows) != 143:
        raise RuntimeError(f"Expected 143 grid positions in {directory}; found {len(rows)}")
    return rows


def aggregate(rows: list[dict], name: str) -> dict:
    trials = sum(row["trials"] for row in rows)
    successes = sum(row["successes"] for row in rows)
    valid = sum(row["accuracy_valid_trials"] for row in rows if row["accuracy_cosine"] is not None)
    accuracy = sum(
        row["accuracy_cosine"] * row["accuracy_valid_trials"]
        for row in rows if row["accuracy_cosine"] is not None
    ) / valid
    lower, upper = wilson(successes, trials)
    return {
        "scope": name,
        "positions": len(rows),
        "trials": trials,
        "successes": successes,
        "success_goal_rate": successes / trials,
        "success_goal_rate_wilson_95": [lower, upper],
        "first_contact_rate": sum(row["contact_rate"] * row["trials"] for row in rows) / trials,
        "kick_accuracy_cosine": accuracy,
        "kick_accuracy_valid_trials": valid,
    }


def heatmap(rows: list[dict], metric: str, title: str, output: Path) -> None:
    forwards = sorted({row["forward_m"] for row in rows})
    laterals = sorted({row["lateral_m"] for row in rows})
    values = np.full((len(laterals), len(forwards)), np.nan)
    lookup = {(row["forward_m"], row["lateral_m"]): row[metric] for row in rows}
    for iy, lateral in enumerate(laterals):
        for ix, forward in enumerate(forwards):
            values[iy, ix] = lookup[(forward, lateral)]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    image = ax.imshow(values, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Ball forward position (m)")
    ax.set_ylabel("Ball lateral position (m)")
    ax.set_xticks(range(len(forwards)), [f"{value:.2f}" for value in forwards], rotation=45)
    ax.set_yticks(range(len(laterals)), [f"{value:.2f}" for value in laterals])
    fig.colorbar(image, ax=ax, label=metric.replace("_", " "))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--rolling-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    static_rows = read_directory(args.static_dir, "static")
    rolling_rows = read_directory(args.rolling_dir, "rolling")
    for rows, label in ((static_rows, "static"), (rolling_rows, "rolling")):
        for row in rows:
            row["in_paper_table_iv_workspace"] = 0.5 <= row["forward_m"] <= 2.0 and abs(row["lateral_m"]) <= 1.0
        heatmap(rows, "success_goal_rate", f"R1 {label}: goal success rate", args.output_dir / f"{label}_success_heatmap.png")
        heatmap(rows, "accuracy_cosine", f"R1 {label}: kick direction cosine", args.output_dir / f"{label}_accuracy_heatmap.png")

    all_rows = static_rows + rolling_rows
    with (args.output_dir / "per_position.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    scopes = {
        "full_11x13_grid": lambda rows: rows,
        "paper_table_iv_workspace": lambda rows: [row for row in rows if row["in_paper_table_iv_workspace"]],
    }
    summary = {scope: {label: aggregate(select(rows), scope) for label, rows in (("static", static_rows), ("rolling", rolling_rows))} for scope, select in scopes.items()}
    summary["protocol"] = {
        "grid": "forward 0.50..3.00 m and lateral -1.50..1.50 m, 0.25 m increments",
        "trials_per_position": 50,
        "robot_initial_position": "uniform disk with 0.5 m radius",
        "rolling_ball_speed": "independently uniform [0.1, 0.3] m/s per environment",
        "rolling_ball_direction": "independently uniform [0, 2pi) per environment",
        "success_predicate": "reused exp.mujoco_soccer.metrics.goal_crossed",
        "model": "model_103998.pt",
        "motions": "motions/r1-converted (10 Standard motion files selected by the environment)",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    paper = {"static": (0.913, 0.9689), "rolling": (0.719, 0.8892)}
    comparison = ["# R1 soccer evaluation: Table IV-aligned comparison", "", "The primary comparison uses the paper Table IV workspace: forward 0.5–2.0 m, lateral -1.0–1.0 m.", "", "| Condition | R1 success rate (95% Wilson CI) | R1 direction cosine | Paper PAiD success rate | Paper PAiD direction cosine |", "|---|---:|---:|---:|---:|"]
    for label in ("static", "rolling"):
        values = summary["paper_table_iv_workspace"][label]
        low, high = values["success_goal_rate_wilson_95"]
        comparison.append(f"| {label} | {values['success_goal_rate']:.4f} ({low:.4f}–{high:.4f}) | {values['kick_accuracy_cosine']:.4f} | {paper[label][0]:.4f} | {paper[label][1]:.4f} |")
    comparison.extend(["", "## Protocol and comparability", "", "- Success uses the repository's original `goal_crossed` predicate; accuracy is the cosine of outgoing ball direction and ball-to-goal direction at contact.", "- The R1 evaluation uses Isaac Lab and R1-converted Standard motions, whereas the paper results use G1/MuJoCo. Values are therefore a protocol-aligned baseline comparison, not a claim of identical simulator or robot conditions.", "- The rolling-speed interval, grid, trial count, and initial-position radius follow the paper. Rolling direction is sampled uniformly as an explicit extension because the paper text specifies speed but not a direction distribution in its Table IV description.", "- R1 has no converted Stylized motion set, so this report does not claim a Table III Stylized comparison.", ""])
    (args.output_dir / "table_iv_comparison.md").write_text("\n".join(comparison), encoding="utf-8")


if __name__ == "__main__":
    main()
