"""Physical worker with opt-in rendering and measured support diagnostics."""

import base64
import io
import json
import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from .physics_worker import Plant, serial


class LivePlant(Plant):
    def __init__(self, request):
        super().__init__(request)
        self.renderer = None
        self.policy_active = False
        self.robot.model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
        self.robot.model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        # Reset-only perturbation exercises preparation without changing the model.
        offset = float(request.get("initial_joint_offset", 0.0))
        if not np.isfinite(offset) or abs(offset) > 0.1:
            raise ValueError("Initial simulation perturbation exceeds the test envelope")
        ids = [self.meta["joint_names"].index(n) for n in self.meta["action_joint_names"] if "shoulder_pitch" in n]
        self.robot.data.qpos[self.robot.qpos_ids[ids]] += offset
        mujoco.mj_forward(self.robot.model, self.robot.data)

    def step(self, request):
        metric = self.metric
        if not self.policy_active:
            self.metric = None
        try:
            result = super().step(request)
        finally:
            if not self.policy_active:
                self.metric = metric
        if self.metric is not None:
            result["metrics"] = self.metric.result()
        return result

    def plan(self, request):
        from .balance import BalancePlanner

        if request.get("capture") or not hasattr(self, "balance"):
            self.balance = BalancePlanner(self)
        return self.balance.target(request["posture"])

    def diagnostics(self):
        r = self.robot
        ground = r.model.geom("ground").id
        supported = set()
        forces = {bid: 0.0 for bid in r.foot_bodies}
        body_ground = False
        for index, contact in enumerate(r.data.contact):
            other = contact.geom2 if contact.geom1 == ground else contact.geom1 if contact.geom2 == ground else -1
            if other < 0:
                continue
            force = np.empty(6)
            mujoco.mj_contactForce(r.model, r.data, index, force)
            if force[0] <= 0:
                continue
            body = int(r.model.geom_bodyid[other])
            if body in r.foot_bodies:
                forces[body] += float(force[0])
            elif body in r.body_ids:
                body_ground = True
        robot_weight = float(r.model.body_subtreemass[r.root_id] * np.linalg.norm(r.model.opt.gravity))
        supported = {bid for bid, force in forces.items() if force >= 0.02 * robot_weight}
        q = r.data.qpos[r.root_q + 3 : r.root_q + 7]
        up = np.empty(3)
        mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), q)
        return dict(
            height=float(r.data.qpos[r.root_q + 2]),
            tilt=float(np.arccos(np.clip(up[2], -1.0, 1.0))),
            linear_speed=float(np.linalg.norm(r.data.qvel[r.root_v : r.root_v + 3])),
            angular_speed=float(np.linalg.norm(r.data.qvel[r.root_v + 3 : r.root_v + 6])),
            joint_speed=float(np.max(np.abs(r.data.qvel[r.dof_ids][self.meta["action_joint_ids"]]))),
            both_feet_loaded=supported == r.foot_bodies,
            foot_normal_forces=[forces[b] for b in sorted(r.foot_bodies)],
            body_ground_contact=body_ground,
        )

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.robot.model, height=480, width=640)
            self.camera = mujoco.MjvCamera()
            self.camera.lookat[:] = [0.0, -0.5, 0.4]
            self.camera.distance = 3.2
            self.camera.azimuth = 225.0
            self.camera.elevation = -25.0
        self.renderer.update_scene(self.robot.data, camera=self.camera)
        scene = self.renderer.scene
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.ones(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array([1.0, 0.7, 0.1, 1.0]),
        )
        start = self.robot.data.qpos[self.robot.root_q : self.robot.root_q + 3].copy()
        start[2] = 0.08
        forward = self.robot.data.xmat[self.robot.root_id].reshape(3, 3)[:, 0].copy()
        forward[2] = 0.0
        forward /= np.linalg.norm(forward)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.015, start, start + 0.5 * forward)
        scene.ngeom += 1
        output = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(output, format="JPEG", quality=80)
        return base64.b64encode(output.getvalue()).decode("ascii")


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
                    plant = LivePlant(request)
                    result = dict(
                        state=plant.state(),
                        diagnostics=plant.diagnostics(),
                        mujoco=mujoco.__version__,
                        physics_dt=plant.robot.model.opt.timestep,
                    )
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
                    raise ValueError("Unknown live worker operation")
            except Exception:
                result = dict(error=traceback.format_exc())
            print(json.dumps(result, default=serial, allow_nan=False), flush=True)
    finally:
        if plant is not None and plant.renderer is not None:
            plant.renderer.close()


if __name__ == "__main__":
    main()
