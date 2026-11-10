"""Opt-in simulation telemetry and immutable prepared-state snapshots."""

import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from .match_worker import ArenaPlant
from .match_arena import sha256, world_to_field
from .physics_worker import serial


class DiagnosticPlant(ArenaPlant):
    def __init__(self, request):
        super().__init__(request)
        self.output = Path(request["diagnostic_output"]).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.sample_rows = []
        self.control_rows = []
        self.command_rows = []
        self.goal_count = 0
        self.policy_epoch = None
        self.snapshot_source = request.get("snapshot_path")
        self.initial_robot_difference = None
        if self.snapshot_source:
            self.restore_snapshot(self.snapshot_source)
        self.physics_observer = self.observe_diagnostic

    def restore_snapshot(self, path):
        path = Path(path)
        receipt = json.loads(path.with_suffix(".json").read_text())
        if sha256(path) != receipt["snapshot_sha256"]:
            raise ValueError("Prepared-state snapshot hash mismatch")
        r = self.robot
        with np.load(path) as saved:
            spec = int(saved["spec"])
            if len(saved["integration_state"]) != mujoco.mj_stateSize(r.model, spec):
                raise ValueError("Prepared-state model dimensions differ")
            mujoco.mj_setState(r.model, r.data, saved["integration_state"], spec)
            for name in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
                getattr(r.model, name)[:] = saved[name]
            # Only initialization changes ball resting height as radius changes.
            # Robot coordinates and velocities are identical in all factorial cells.
            r.data.qpos[r.ball_q + 2] += self.radius - float(saved["radius"])
            robot_q = np.r_[np.arange(r.root_q, r.root_q + 7), r.qpos_ids]
            self.initial_robot_difference = float(
                np.max(np.abs(r.data.qpos[robot_q] - saved["qpos"][robot_q]))
            )
            np.testing.assert_array_equal(r.data.qvel, saved["qvel"])
            assert self.initial_robot_difference == 0
        # Each fresh replay has its own elapsed clock; no state writes after start.
        r.data.time = 0
        mujoco.mj_forward(r.model, r.data)
        from .match_geometry import ArenaTracker

        self.tracker = ArenaTracker(
            self.geometry,
            self.radius,
            world_to_field(self.state()["ball_pos"], self.heading),
        )
        self.physics_samples_observed = 0

    def capture_snapshot(self, goal):
        r = self.robot
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        values = np.empty(mujoco.mj_stateSize(r.model, spec))
        mujoco.mj_getState(r.model, r.data, values, spec)
        path = self.output / "prepared_state.npz"
        np.savez(
            path,
            integration_state=values,
            spec=spec,
            qpos=r.data.qpos.copy(),
            qvel=r.data.qvel.copy(),
            ctrl=r.data.ctrl.copy(),
            radius=self.radius,
            actuator_gainprm=r.model.actuator_gainprm.copy(),
            actuator_biasprm=r.model.actuator_biasprm.copy(),
            actuator_forcerange=r.model.actuator_forcerange.copy(),
        )
        receipt = dict(
            snapshot_sha256=sha256(path),
            mujoco=mujoco.__version__,
            arena_manifest=str(self.manifest_path),
            arena_model_sha256=self.manifest["generated_model_sha256"],
            motion=self.name,
            goal=list(goal),
            physics_seconds=float(r.data.time),
            radius=self.radius,
            mass=self.manifest["ball"]["mass"],
            qpos=r.data.qpos.tolist(),
            qvel=r.data.qvel.tolist(),
            diagnostics=self.diagnostics(),
            source_snapshot=self.snapshot_source,
            robot_initialization_max_difference=self.initial_robot_difference,
        )
        path.with_suffix(".json").write_text(json.dumps(receipt, indent=2))

    def goal(self, goal):
        self.goal_count += 1
        if self.goal_count != 1:
            raise RuntimeError("Diagnostic trial requires exactly one policy start")
        self.capture_snapshot(goal)
        self.policy_epoch = float(self.robot.data.time)
        super().goal(goal)

    def observe_diagnostic(self, before, after, seconds):
        super().observe_ball(before, after, seconds)
        if self.policy_epoch is None:
            return
        r = self.robot
        ball_id = r.model.geom("soccer_ball_geom").id
        forces = {bid: 0.0 for bid in sorted(r.foot_bodies)}
        for i, contact in enumerate(r.data.contact):
            other = (
                contact.geom2
                if contact.geom1 == ball_id
                else contact.geom1
                if contact.geom2 == ball_id
                else -1
            )
            if other < 0:
                continue
            body = int(r.model.geom_bodyid[other])
            if body in forces:
                force = np.empty(6)
                mujoco.mj_contactForce(r.model, r.data, i, force)
                forces[body] += max(float(force[0]), 0)
        velocity = r.data.qvel[r.ball_v : r.ball_v + 3]
        point = world_to_field(after, self.heading)
        vel = world_to_field(velocity, self.heading)
        feet = [
            r.model.body(n).id
            for n in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        self.sample_rows.append(
            [
                seconds - self.policy_epoch,
                int(self.policy_active),
                *point,
                *vel,
                *[forces[bid] for bid in feet],
            ]
        )

    def step(self, request):
        result = super().step(request)
        if self.policy_epoch is not None:
            r = self.robot
            feet = [
                r.model.body(n).id
                for n in ("left_ankle_roll_link", "right_ankle_roll_link")
            ]
            positions = np.concatenate(
                [world_to_field(r.data.xpos[i], self.heading) for i in feet]
            )
            self.command_rows.append(np.asarray(request["positions"]).copy())
            self.control_rows.append(
                [
                    float(r.data.time) - self.policy_epoch,
                    int(self.policy_active),
                    *positions,
                    self.diagnostics()["tilt"],
                    self.diagnostics()["height"],
                ]
            )
        return result

    def dump(self):
        np.savez_compressed(
            self.output / "contact_trace.npz",
            samples=np.asarray(self.sample_rows),
            controls=np.asarray(self.control_rows),
            commands=np.asarray(self.command_rows),
        )
        (self.output / "telemetry.json").write_text(
            json.dumps(
                dict(
                    policy_starts=self.goal_count,
                    physics_samples=len(self.sample_rows),
                    contact_columns=[
                        "elapsed",
                        "policy_active",
                        "ball_x",
                        "ball_y",
                        "ball_z",
                        "ball_vx",
                        "ball_vy",
                        "ball_vz",
                        "left_normal_force",
                        "right_normal_force",
                    ],
                    control_columns=[
                        "elapsed",
                        "policy_active",
                        "left_x",
                        "left_y",
                        "left_z",
                        "right_x",
                        "right_y",
                        "right_z",
                        "tilt",
                        "root_height",
                    ],
                    scope="one policy start; force samples at physics rate; touch episodes are not counts of intentional kicks",
                ),
                indent=2,
            )
        )


def main():
    plant = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "init":
                    if plant is not None:
                        raise ValueError("Reset requires a new worker")
                    plant = DiagnosticPlant(request)
                    result = dict(
                        state=plant.state(),
                        diagnostics=plant.diagnostics(),
                        mujoco=mujoco.__version__,
                        physics_dt=plant.robot.model.opt.timestep,
                    )
                elif op == "arena_status":
                    result = plant.arena_status()
                elif op == "plan":
                    result = dict(positions=plant.plan(request))
                elif op == "end_policy":
                    plant.policy_active = False
                    result = dict(ok=True)
                elif op == "render":
                    result = dict(jpeg=plant.render())
                elif op == "goal":
                    plant.goal(request["goal"])
                    plant.steps = 0
                    plant.policy_active = True
                    result = dict(ok=True)
                elif op in ("step", "hold"):
                    result = plant.step(request)
                    result["diagnostics"] = plant.diagnostics()
                elif op == "close":
                    plant.dump()
                    print(json.dumps(dict(ok=True)), flush=True)
                    return
                else:
                    raise ValueError("Unknown diagnostic worker operation")
            except Exception:
                result = dict(error=traceback.format_exc())
            print(json.dumps(result, default=serial, allow_nan=False), flush=True)
    finally:
        if plant is not None:
            plant.dump()
            if plant.renderer is not None:
                plant.renderer.close()


if __name__ == "__main__":
    main()
