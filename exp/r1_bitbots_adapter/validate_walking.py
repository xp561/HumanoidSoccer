"""Continuous single-world ROS walk/stop/kick/recover/walk acceptance."""

import argparse
import base64
import subprocess
import json
from pathlib import Path
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from .walking_session import WalkingSession
from .mujoco_bridge import spin_until


def run(output, walk_seconds, record=False):
    session = WalkingSession(output, realtime=False, render=False)
    trace = []
    video = None
    if record:
        video = subprocess.Popen(
            [
                "/usr/bin/ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-framerate",
                "10",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output / "continuous.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
        original_consume = session.consume
        frame_index = 0

        def consume(command):
            nonlocal frame_index
            result = original_consume(command)
            if frame_index % 5 == 0:
                video.stdin.write(
                    base64.b64decode(session.plant.rpc(dict(op="render"))["jpeg"])
                )
            frame_index += 1
            return result

        session.consume = consume
    try:
        session.setup()
        pub = session.bridge.create_publisher(Twist, "cmd_vel", 10)
        session.sync(owner=False)

        def step(command, label):
            session.sync(owner=False)
            if command is not None:
                msg = Twist()
                msg.linear.x = float(command[0])
                msg.linear.y = float(command[1])
                msg.angular.z = float(command[2])
                session.walker.command_ns = None
                pub.publish(msg)
                spin_until(
                    session.executor, lambda: session.walker.command_ns == session.ns
                )
            session.controlled_step(session.router.gate.owner, session.posture)
            trace.append(
                dict(
                    label=label,
                    seconds=session.seconds,
                    root_pos=session.state["root_pos"].tolist(),
                    diagnostics=session.measured,
                    owner=session.router.gate.owner,
                )
            )
            if session.unsafe():
                raise RuntimeError("Unsafe posture during " + label)

        def yaw():
            q = session.state["root_quat"]
            return float(
                np.arctan2(
                    2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
                )
            )

        target_yaw = yaw()

        def heading_command(vx):
            error = np.arctan2(np.sin(target_yaw - yaw()), np.cos(target_yaw - yaw()))
            return [vx, 0, float(np.clip(3 * error, -0.3, 0.3))]

        for _ in range(150):
            step(heading_command(0), "initial_stand")
        start = session.state["root_pos"].copy()
        for _ in range(round(walk_seconds / 0.02)):
            step(heading_command(0.2), "walk_before")
        before = session.state["root_pos"].copy()
        session.shot()
        report = json.loads(Path(session.last_report).read_text())
        if report["recovered"]:
            after = session.state["root_pos"].copy()
            for _ in range(150):
                step(heading_command(0.2), "walk_after")
            displacement = float(
                np.linalg.norm(session.state["root_pos"][:2] - after[:2])
            )
            for _ in range(200):
                step(None, "command_dropout_stop")
        else:
            displacement = None
        final = session.measured
        stopped = (
            final["linear_speed"] < 0.06
            and final["angular_speed"] < 0.12
            and final["tilt"] < 0.2
            and final["both_feet_loaded"]
            and not final["body_ground_contact"]
        )
        passed = (
            report["recovered"]
            and report["outcome"] == "policy_completed"
            and report["policy_commands"] == 400
            and report["policy_commands_after_terminal"] == 0
            and displacement is not None
            and displacement > 0.3
            and stopped
            and session.router.gate.owner == "walk"
        )
        result = dict(
            passed=passed,
            recovered=report["recovered"],
            outcome=report["outcome"],
            prewalk_displacement=float(np.linalg.norm(before[:2] - start[:2])),
            postwalk_displacement=displacement,
            final_diagnostics=session.measured,
            final_owner=session.router.gate.owner,
            shot_report=str(session.last_report),
            no_reset_after_initialization=True,
            trace=trace,
        )
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
        if not passed:
            raise RuntimeError("Continuous handoff acceptance failed; see result.json")
    finally:
        if video is not None:
            video.stdin.close()
            if video.wait(timeout=30) != 0:
                raise RuntimeError("Video encoder failed")
        if session.plant is not None:
            session.plant.close()
        if hasattr(session, "executor"):
            session.executor.shutdown(timeout_sec=3)
            for name in ("walker", "bridge", "router", "adapter"):
                if hasattr(session, name):
                    getattr(session, name).destroy_node()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--walk-seconds", type=float, default=1.0)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    try:
        run(args.output, args.walk_seconds, args.record)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
