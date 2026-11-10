"""Physical fail-closed and cancellation checks for the opt-in walking session."""

import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from .mujoco_bridge import spin_until
from .walking_session import WalkingSession
from .readiness import ReadinessConfig


def main():
    root = Path("logs/r1_logic_fix/r1_walk_handoff_20261027/failures")
    results = []
    for case in ("cancel_preparation", "recovery_timeout"):
        rclpy.init()
        config = ReadinessConfig()
        if case == "recovery_timeout":
            config = replace(config, recovery_timeout=config.dwell / 5)
        session = WalkingSession(
            root / case, realtime=False, render=False, config=config
        )
        try:
            session.setup()
            pub = session.bridge.create_publisher(Twist, "cmd_vel", 10)

            def yaw():
                q = session.state["root_quat"]
                return float(
                    np.arctan2(
                        2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
                    )
                )

            target_yaw = yaw()

            def step(vx):
                session.sync(owner=False)
                error = np.arctan2(
                    np.sin(target_yaw - yaw()), np.cos(target_yaw - yaw())
                )
                msg = Twist()
                msg.linear.x = vx
                msg.angular.z = float(np.clip(3 * error, -0.3, 0.3))
                session.walker.command_ns = None
                pub.publish(msg)
                spin_until(
                    session.executor, lambda: session.walker.command_ns == session.ns
                )
                session.controlled_step("walk", session.posture)

            for _ in range(150):
                step(0.0)
            for _ in range(30):
                step(0.2)
            if case == "cancel_preparation":
                original = session.controlled_step

                def controlled(source, target):
                    result = original(source, target)
                    if session.stage == "preparing":
                        session.cancel.set()
                    return result

                session.controlled_step = controlled
            session.shot()
            report = json.loads(Path(session.last_report).read_text())
            if case == "cancel_preparation":
                passed = (
                    report["outcome"] == "cancelled" and report["policy_commands"] == 0
                )
                passed = passed and (
                    report["recovered"]
                    and report["final_owner"] == "walk"
                    or not report["recovered"]
                    and report["final_owner"] == "hcm"
                )
            else:
                passed = (
                    report["outcome"] == "policy_completed"
                    and report["policy_commands"] == 400
                    and not report["recovered"]
                    and report["final_stage"] == "fault"
                    and report["final_owner"] == "hcm"
                    and report["events"][-1]["reason"] == "recovery_timeout"
                    and not any(e["stage"] == "ready" for e in report["events"])
                )
            passed = passed and report["policy_commands_after_terminal"] == 0
            row = dict(
                case=case,
                passed=passed,
                outcome=report["outcome"],
                recovered=report["recovered"],
                owner=report["final_owner"],
                policy_commands=report["policy_commands"],
            )
            results.append(row)
            print(json.dumps(row), flush=True)
        finally:
            if session.plant is not None:
                session.plant.close()
            if hasattr(session, "executor"):
                session.executor.shutdown(timeout_sec=3)
                for name in ("walker", "bridge", "router", "adapter"):
                    if hasattr(session, name):
                        getattr(session, name).destroy_node()
            rclpy.shutdown()
    (root / "summary.json").write_text(
        json.dumps(
            dict(passed=all(r["passed"] for r in results), cases=results), indent=2
        )
    )
    if not all(r["passed"] for r in results):
        raise SystemExit("Walking failure acceptance failed")


if __name__ == "__main__":
    main()
