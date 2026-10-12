"""Create a single, protocol-aligned G1/R1 soccer comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rate(item: dict) -> str:
    low, high = item["success_goal_rate_wilson_95"]
    return f"{item['success_goal_rate']:.4f} ({low:.4f}-{high:.4f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    g1, r1 = (json.loads(path.read_text()) for path in (args.g1, args.r1))
    scope = "paper_table_iv_workspace"
    lines = [
        "# G1 and R1 soccer evaluation comparison",
        "",
        "All values use the same 11 x 13 protocol, 50 trials per ball position, and the Table IV workspace filter (forward 0.5-2.0 m, lateral -1.0-1.0 m).",
        "",
        "| Condition | G1 success rate (95% Wilson CI) | R1 success rate (95% Wilson CI) | G1 direction cosine | R1 direction cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in ("static", "rolling"):
        left, right = g1[scope][condition], r1[scope][condition]
        lines.append(f"| {condition} | {rate(left)} | {rate(right)} | {left['kick_accuracy_cosine']:.4f} | {right['kick_accuracy_cosine']:.4f} |")
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Both checkpoints were evaluated in the same Isaac Lab soccer evaluator, with the same goal-crossing predicate and rolling-ball sampling implementation.",
        "- The models use their respective robot bodies and Standard motion sets; this measures task outcomes under a common protocol, not morphology-normalized capability.",
        "- Confidence intervals quantify trial sampling uncertainty only; they do not capture simulator-model uncertainty.",
        "",
    ])
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
