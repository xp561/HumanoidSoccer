"""Serial, auditable batch for ROS/MuJoCo integration diagnostics."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "logs/r1_logic_fix/r1_ros_mujoco_20261023"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-existing", action="store_true", help="Resume saved cases only when source and assets are unchanged"
    )
    args = parser.parse_args()
    rows = []
    tasks = [(m, rolling, None) for m in range(10) for rolling in (False, True)]
    tasks += [(0, rolling, fault) for rolling in (False, True) for fault in ("cancel", "ball_dropout", "owner_loss")]
    for motion, rolling, fault in tasks:
        name = f"motion_{motion:02d}_{'rolling' if rolling else 'static'}"
        if fault:
            name += "_" + fault
        folder = OUT / name
        if (
            args.reuse_existing
            and motion == 0
            and not rolling
            and fault is None
            and (OUT / "smoke/result.json").exists()
        ):
            folder = OUT / "smoke"
        result_path = folder / "result.json"
        if not args.reuse_existing or not result_path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-u",
                "-m",
                "exp.r1_bitbots_adapter.mujoco_bridge",
                "--motion",
                str(motion),
                "--output",
                str(folder),
            ]
            if rolling:
                command.append("--rolling")
            if fault:
                command += ["--fault", fault]
            with (folder / "driver.log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError("Integration case failed; inspect " + str(folder / "driver.log"))
        result = json.loads(result_path.read_text())
        if not result["passed"]:
            raise RuntimeError("Failed comparison: " + str(result_path))
        ros = result["ros"]
        row = dict(
            name=name,
            result=str(result_path),
            passed=True,
            policy_commands=ros["policy_commands"],
            fault=fault,
            condition="rolling" if rolling else "static",
            post_terminal_commands=ros["post_terminal_commands"],
            kicked_goal=ros["final_controlled"]["metrics"]["kicked_goal"],
            numerical_failure=ros["final_controlled"]["metrics"]["numerical_failure"],
        )
        if fault is None:
            row["maximum_trajectory_error"] = max(result["max_abs_trajectory_error"].values())
            row["score_agreement"] = result["score_agreement"]
        rows.append(row)
        (OUT / "progress.json").write_text(json.dumps(dict(completed=len(rows), total=len(tasks), rows=rows), indent=2))
        print(json.dumps(row), flush=True)
    pairs = [x for x in rows if x["fault"] is None]
    faults = [x for x in rows if x["fault"] is not None]
    summary = dict(
        passed=True,
        paired_cases=len(pairs),
        paired_physical_runs=2 * len(pairs),
        fault_physical_runs=len(faults),
        physical_runs=2 * len(pairs) + len(faults),
        maximum_trajectory_error=max(x["maximum_trajectory_error"] for x in pairs),
        all_scores_agree=all(x["score_agreement"] for x in pairs),
        post_terminal_commands=sum(x["post_terminal_commands"] for x in rows),
        numerical_failures=sum(x["numerical_failure"] for x in rows),
        diagnostic_goals={
            c: dict(
                goals=sum(x["kicked_goal"] for x in pairs if x["condition"] == c),
                cases=sum(x["condition"] == c for x in pairs),
            )
            for c in ("static", "rolling")
        },
        scope="Nominal clean-observation interface diagnostics; not randomized formal shooting evaluation.",
        rows=rows,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
