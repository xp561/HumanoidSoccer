"""Diagnostic-only ball-height observation intervention over the native raw Action."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rclpy

from .replay_prepared_kick import PreparedSession
from .runtime import inverse_rotate


class ObservedHeightSession(PreparedSession):
    def __init__(self, output, manifest, snapshot, observed_radius):
        physical = json.loads(manifest.read_text())["ball"]["radius"]
        self.height_offset = float(observed_radius - physical)
        self.observed_radius = observed_radius
        self.observation_rows = []
        super().__init__(output, manifest, snapshot)

    def setup(self):
        super().setup()
        original = self.adapter.runtime.observation

        def diagnostic_observation(**state):
            truth = np.array(state["ball_pos"], dtype=float, copy=True)
            shifted = truth.copy()
            shifted[2] += self.height_offset
            baseline = original(**state)
            modified = original(**{**state, "ball_pos": shifted})
            # Only the three ball coordinates change. All goal/reference and
            # proprioceptive terms retain exactly the same state in this call.
            np.testing.assert_array_equal(modified[:-6], baseline[:-6])
            np.testing.assert_array_equal(modified[-3:], baseline[-3:])
            expected = inverse_rotate(state["root_quat"], [0, 0, self.height_offset])
            np.testing.assert_allclose(modified[-6:-3] - baseline[-6:-3], expected, atol=2e-6, rtol=0)
            self.observation_rows.append((truth, shifted, baseline.copy(), modified.copy()))
            return modified

        self.adapter.runtime.observation = diagnostic_observation

    def close(self):
        try:
            super().close()
        finally:
            if self.observation_rows:
                np.savez_compressed(
                    self.output / "policy_inputs.npz",
                    **{
                        name: np.asarray([row[i] for row in self.observation_rows])
                        for i, name in enumerate(
                            ["true_ball_world", "input_ball_world", "true_observation", "input_observation"]
                        )
                    },
                )
            (self.output / "intervention.json").write_text(
                json.dumps(
                    dict(
                        observed_resting_radius_m=self.observed_radius,
                        world_z_offset_m=self.height_offset,
                        observed_policy_frames=len(self.observation_rows),
                        scope="policy input only; dynamic world-z offset; all physics, sensors, goal and recovery retain truth",
                    ),
                    indent=2,
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arena-manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--observed-radius", type=float, required=True, choices=[0.07, 0.11])
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "185":
        raise RuntimeError("Requires isolated simulation domain 185")
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    session = ObservedHeightSession(args.output, args.arena_manifest, args.snapshot, args.observed_radius)
    try:
        session.setup()
        session.run_prepared()
        report = json.loads(Path(session.last_report).read_text())
        result = {k: report[k] for k in ["recovered", "outcome", "final_stage", "final_owner", "policy_commands"]}
        result.update(
            late_commands=report["policy_commands_after_terminal"],
            source_snapshot=str(args.snapshot.resolve()),
            arena_manifest=str(args.arena_manifest.resolve()),
            observed_radius_m=args.observed_radius,
            observation_z_offset_m=session.height_offset,
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        session.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
