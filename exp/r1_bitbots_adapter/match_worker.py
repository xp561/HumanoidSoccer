"""Opt-in independent arena worker; ordinary live worker stays on frozen model."""

import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from .live_worker import LivePlant
from .physics_worker import serial
from .match_arena import sha256, world_to_field
from .match_geometry import ArenaTracker, FieldGeometry


class ArenaPlant(LivePlant):
    def __init__(self, request):
        self.manifest_path = Path(request["arena_manifest"]).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        path = self.manifest_path.parent / "model.xml"
        if sha256(path) != self.manifest["generated_model_sha256"]:
            raise ValueError("Arena model hash mismatch")
        if (
            sha256(self.manifest["source_model"])
            != self.manifest["source_model_sha256"]
        ):
            raise ValueError("Frozen source model hash mismatch")
        super().__init__(request, model_path=path)
        self.radius = self.manifest["ball"]["radius"]
        # Initialization only: original Plant uses its frozen profile radius.
        self.robot.data.qpos[self.robot.ball_q + 2] = self.radius
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.geometry = FieldGeometry(self.manifest["config"])
        self.heading = self.manifest["field_heading_in_odom_rad"]
        initial = world_to_field(self.state()["ball_pos"], self.heading)
        self.tracker = ArenaTracker(self.geometry, self.radius, initial)
        self.physics_samples_observed = 0
        self.physics_observer = self.observe_ball

    def observe_ball(self, before, after, seconds):
        self.physics_samples_observed += 1
        self.tracker.observe(world_to_field(after, self.heading), seconds)

    def arena_status(self):
        expected = round(self.robot.data.time / self.robot.model.opt.timestep)
        if self.physics_samples_observed != expected:
            raise RuntimeError("Arena observer missed physics samples")
        return dict(
            events=self.tracker.events,
            physics_samples_observed=self.physics_samples_observed,
            ball_field=world_to_field(self.state()["ball_pos"], self.heading),
            root_field=world_to_field(self.state()["root_pos"], self.heading),
            ball_profile=self.manifest["ball_profile"],
            physical_resets_after_initialization=0,
            scope="first-event detector only; restarts and GameController not yet connected",
        )

    def render(self):
        # Native renderer's close view is useful for controller debugging.
        return super().render()


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
                    plant = ArenaPlant(request)
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
                    print(json.dumps(dict(ok=True)), flush=True)
                    return
                else:
                    raise ValueError("Unknown arena worker operation")
            except Exception:
                result = dict(error=traceback.format_exc())
            print(json.dumps(result, default=serial, allow_nan=False), flush=True)
    finally:
        if plant is not None and plant.renderer is not None:
            plant.renderer.close()


if __name__ == "__main__":
    main()
