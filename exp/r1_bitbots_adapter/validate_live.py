"""Offline physical acceptance for the live preparation/recovery controller."""

import argparse
import json
import threading
from dataclasses import replace
from pathlib import Path

import rclpy

from .live_session import Session
from .readiness import ReadinessConfig


def run_case(name, output):
    config = ReadinessConfig()
    if name == "recovery_timeout":
        config = replace(config, recovery_timeout=config.dwell / 5)
    elif name == "preparation_timeout":
        config = replace(config, preparation_timeout=config.dwell / 5)
    rclpy.init()
    session = Session(
        output / name, realtime=False, render=False, config=config, joint_offset=0.04 if name == "perturbed" else 0.0
    )
    finished = threading.Event()
    watcher = None
    try:
        session.setup(rolling=name == "rolling")
        if name in ("cancel_preparation", "cancel_kick"):
            desired = "preparing" if name == "cancel_preparation" else "kicking"

            def cancel_at_phase():
                entered = None
                while not finished.wait(0.002):
                    with session.lock:
                        status = session.status.copy()
                    if status.get("stage") == desired:
                        if entered is None:
                            entered = status["simulation_seconds"]
                        if status["simulation_seconds"] - entered >= 0.08:
                            session.cancel.set()
                            return

            watcher = threading.Thread(target=cancel_at_phase, daemon=True)
            watcher.start()
        if name == "settled":
            for _ in range(150):
                session.controlled_step("hold", session.posture)
        session.shot()
        report = json.loads(Path(session.last_report).read_text())
        if name in ("nominal", "perturbed", "settled"):
            passed = report["recovered"] and report["outcome"] == "policy_completed"
        elif name == "rolling":
            passed = (
                report["recovered"]
                and report["policy_commands"] == 0
                and report["outcome"] == "Arbitrary world yaw is not yet validated"
            )
        elif name.startswith("cancel"):
            stopped = report["outcome"] == "cancelled"
            safe_return = report["recovered"] and report["final_owner"] == "hold"
            failed_closed = (
                not report["recovered"]
                and report["final_stage"] == "fault"
                and report["final_owner"] == "hcm"
                and not any(event["stage"] == "ready" for event in report["events"])
            )
            passed = stopped and (safe_return or failed_closed)
        else:
            expected = "stopping_timeout" if name == "preparation_timeout" else "recovery_timeout"
            passed = (
                not report["recovered"]
                and report["final_stage"] == "fault"
                and report["events"][-1]["reason"] == expected
            )
        passed = passed and report["policy_commands_after_terminal"] == 0
        return dict(
            case=name,
            passed=passed,
            recovered=report["recovered"],
            outcome=report["outcome"],
            final_stage=report["final_stage"],
            report=session.last_report,
            shot_metrics=report["shot_metrics"],
            final_diagnostics=report["final_diagnostics"],
        )
    finally:
        finished.set()
        if watcher is not None:
            watcher.join(timeout=1)
        if session.plant is not None:
            session.plant.close()
        if hasattr(session, "executor"):
            session.executor.shutdown(timeout_sec=3)
            for node in (session.bridge, session.router, session.adapter):
                node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("logs/r1_logic_fix/r1_live_acceptance"))
    parser.add_argument(
        "--case",
        choices=(
            "nominal",
            "perturbed",
            "settled",
            "rolling",
            "cancel_preparation",
            "cancel_kick",
            "preparation_timeout",
            "recovery_timeout",
        ),
    )
    args = parser.parse_args()
    cases = (
        [args.case]
        if args.case
        else [
            "nominal",
            "perturbed",
            "settled",
            "rolling",
            "cancel_preparation",
            "cancel_kick",
            "preparation_timeout",
            "recovery_timeout",
        ]
    )
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for name in cases:
        result = run_case(name, args.output)
        rows.append(result)
        print(json.dumps(result), flush=True)
        (args.output / "summary.json").write_text(
            json.dumps(dict(passed=all(r["passed"] for r in rows), cases=rows), indent=2)
        )
    if not all(r["passed"] for r in rows):
        raise SystemExit("Live physical acceptance has failing cases")


if __name__ == "__main__":
    main()
