"""Prepared-state timing replay with the same commanded walking after recovery."""

import argparse
import json
import os
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist

from .mujoco_bridge import spin_until
from .replay_handoff_timing import TimedSession


class ForwardSession(TimedSession):
    def setup(self):
        super().setup()
        self.resume_command_pub = self.bridge.create_publisher(Twist, "cmd_vel", 10)

    def controlled_step(self, source, target):
        if source == "walk":
            self.sync(owner=False)
            msg = Twist()
            msg.linear.x = 0.15
            self.resume_command_pub.publish(msg)
            spin_until(self.executor, lambda: self.walker.command_ns == self.ns)
        return super().controlled_step(source, target)

    def observe_and_resume(self):
        result = super().observe_and_resume()
        result["commanded_resume_window_s"] = result.pop("resumed_zero_walker_seconds")
        result["resume_body_velocity_mps"] = [0.15, 0.0, 0.0]
        result["healthy_full_resume"] = bool(
            result["recovered"]
            and not result["unsafe_after_ready"]
            and result["final_owner"] == "walk"
            and not self.unsafe()
            and result["commanded_resume_window_s"] >= 4.0 - 1e-9
        )
        result["scope"] = (
            "paired prepared-state simulation; same body-forward 0.15 m/s cmd_vel for 4 s after recovery; raw Action and frozen walker; no native navigation during replay"
        )
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arena-manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--duration", type=float, choices=[5.2, 8.0], required=True)
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "185" or os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp":
        raise RuntimeError("Requires explicit isolated Fast DDS domain 185")
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    session = ForwardSession(args.output, args.arena_manifest, args.snapshot, args.duration)
    try:
        session.setup()
        session.run_prepared()
        result = session.observe_and_resume()
        result.update(duration_s=args.duration, source_snapshot=str(args.snapshot.resolve()))
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        session.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
