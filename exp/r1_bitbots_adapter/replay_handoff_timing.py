"""Opt-in raw-Action timing diagnostic with a common physical observation window."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rclpy

from .handoff import Phase
from .replay_prepared_kick import PreparedSession


class TimedSession(PreparedSession):
    def __init__(self, output, manifest, snapshot, duration):
        self.duration = duration
        self.fixed_goal = np.asarray(json.loads(snapshot.with_suffix(".json").read_text())["goal"])
        super().__init__(output, manifest, snapshot)

    def setup(self):
        super().setup()
        self.adapter.cfg["execution_seconds"] = self.duration
        original = self.adapter._request

        def fixed_goal_request(request):
            ref, _, timeout = original(request)
            return ref, self.fixed_goal.copy(), timeout

        self.adapter._request = fixed_goal_request

    def observe_and_resume(self):
        # run_prepared writes the measured recovery result before this extension.
        trial = json.loads(Path(self.last_report).read_text())
        kick_zero = next(e["seconds"] for e in self.events if e["stage"] == "kicking")
        recovered = bool(trial["recovered"])
        # All cases retain 8 s of physical ball observation from the policy start.
        # Successful cases also exercise 4 s of real zero-command walking.
        start = self.seconds
        target = max(kick_zero + 8.0, start + (4.0 if recovered else 0.0))
        origin = self.state["root_pos"].copy()
        unsafe_after_ready = False
        while self.seconds < target - 1e-8:
            if self.router.gate.phase == Phase.READY and self.unsafe():
                unsafe_after_ready = True
                self.fault("unsafe_after_ready_during_walker_test")
            self.controlled_step(self.router.gate.owner, self.posture.copy())
            if len(self.kick_commands) != self.terminal_kick_count:
                raise RuntimeError("Late policy command during observation extension")
        return dict(
            recovered=recovered,
            policy_commands=trial["policy_commands"],
            late_commands=len(self.kick_commands) - self.terminal_kick_count,
            outcome=trial["outcome"],
            recovery_end_seconds_from_kick=start - kick_zero,
            observation_seconds_from_kick=self.seconds - kick_zero,
            recovery_failure=None if recovered else trial["events"][-1]["reason"],
            final_owner=self.router.gate.owner,
            resumed_zero_walker_seconds=self.seconds - start if recovered else 0.0,
            unsafe_after_ready=unsafe_after_ready,
            walker_displacement_m=float(np.linalg.norm(self.state["root_pos"][:2] - origin[:2])) if recovered else None,
            scope="prepared-state simulation only; native raw Action with diagnostic duration; original recovery; common >=8 s true-physics scoring; zero-command walker after successful recovery; HCM hold if faulted",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arena-manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--duration", type=float, choices=[5.2, 6.0, 8.0], required=True)
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "185":
        raise RuntimeError("Requires isolated simulation domain 185")
    args.output.mkdir(parents=True, exist_ok=True)
    if os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp":
        raise RuntimeError("Diagnostic replay requires explicit Fast DDS inside Pixi")
    rclpy.init()
    session = TimedSession(args.output, args.arena_manifest, args.snapshot, args.duration)
    try:
        session.setup()
        session.run_prepared()
        summary = session.observe_and_resume()
        summary.update(
            duration_s=args.duration,
            rmw_implementation=os.environ["RMW_IMPLEMENTATION"],
            source_snapshot=str(args.snapshot.resolve()),
            arena_manifest=str(args.arena_manifest.resolve()),
            fixed_goal=session.fixed_goal.tolist(),
        )
        (args.output / "result.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        session.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
