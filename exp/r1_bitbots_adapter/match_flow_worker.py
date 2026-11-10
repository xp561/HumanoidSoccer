"""Opt-in arena referee: explicit ball placement, never a robot reset."""

import json
import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from .match_worker import ArenaPlant
from .match_arena import field_to_world, world_to_field
from .match_geometry import ArenaTracker
from .physics_worker import serial


class MatchFlowPlant(ArenaPlant):
    def __init__(self, request):
        super().__init__(request)
        self.ball_placements = []
        self.event_archive = []
        self.epoch = 0
        self.initial_referee_launch = request.get("initial_referee_ball_velocity_field")
        if self.initial_referee_launch is not None:
            velocity = np.asarray(self.initial_referee_launch, float)
            if (
                velocity.shape != (3,)
                or not np.isfinite(velocity).all()
                or velocity[2] != 0
                or np.linalg.norm(velocity) > 3
            ):
                raise ValueError("Invalid initial referee launch probe")
            self.robot.data.qvel[self.robot.ball_v : self.robot.ball_v + 3] = (
                field_to_world(velocity, self.heading)
            )
            mujoco.mj_forward(self.robot.model, self.robot.data)

    def feed(self):
        return dict(epoch=self.epoch, events=self.tracker.events)

    def step(self, request):
        result = super().step(request)
        result["match_feed"] = self.feed()
        return result

    def restart_ball(self, request):
        r = self.robot
        d = self.diagnostics()
        if (
            self.policy_active
            or not request.get("stopped_ack")
            or not self.tracker.events
        ):
            raise ValueError(
                "Ball placement requires a stopped terminal epoch and ended policy"
            )
        if not (
            d["both_feet_loaded"]
            and not d["body_ground_contact"]
            and d["tilt"] <= 0.2
            and d["linear_speed"] <= 0.15
            and d["angular_speed"] <= 0.5
            and d["joint_speed"] <= 0.8
        ):
            raise ValueError("Robot must be measured stable before ball placement")
        point = np.asarray(request["ball_field"], float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Invalid ball placement")
        c = self.manifest["config"]
        if (
            abs(point[0]) + self.radius >= c["field_length"] / 2
            or abs(point[1]) + self.radius >= c["field_width"] / 2
            or abs(point[2] - self.radius) > 1e-9
        ):
            raise ValueError(
                "Ball placement must be wholly inside the field at resting height"
            )
        robot = world_to_field(self.state()["root_pos"], self.heading)
        if np.linalg.norm(point[:2] - np.asarray(robot[:2])) < 0.4:
            raise ValueError("Referee placement is too close to robot")
        robot_q = r.data.qpos.copy()
        robot_v = r.data.qvel.copy()
        before = world_to_field(self.state()["ball_pos"], self.heading)
        self.event_archive.extend(
            dict(epoch=self.epoch, **e) for e in self.tracker.events
        )
        # Sole post-initialization state write: explicitly requested referee ball reset.
        r.data.qpos[r.ball_q : r.ball_q + 3] = field_to_world(point, self.heading)
        r.data.qpos[r.ball_q + 3 : r.ball_q + 7] = [1, 0, 0, 0]
        r.data.qvel[r.ball_v : r.ball_v + 6] = 0
        r.data.qacc_warmstart[r.ball_v : r.ball_v + 6] = 0
        robot_q[r.ball_q : r.ball_q + 7] = r.data.qpos[r.ball_q : r.ball_q + 7]
        robot_v[r.ball_v : r.ball_v + 6] = r.data.qvel[r.ball_v : r.ball_v + 6]
        np.testing.assert_array_equal(r.data.qpos, robot_q)
        np.testing.assert_array_equal(r.data.qvel, robot_v)
        mujoco.mj_forward(r.model, r.data)
        self.epoch += 1
        self.tracker = ArenaTracker(self.geometry, self.radius, point)
        self.ball_placements.append(
            dict(
                epoch=self.epoch,
                seconds=float(r.data.time),
                before_field=before,
                after_field=point.tolist(),
                robot_qpos_change=0,
                robot_qvel_change=0,
                reason="explicit simulation referee restart service",
            )
        )
        return dict(
            state=self.state(),
            diagnostics=self.diagnostics(),
            match_feed=self.feed(),
            placement=self.ball_placements[-1],
        )

    def arena_status(self):
        result = super().arena_status()
        result.update(
            events=self.event_archive
            + [dict(epoch=self.epoch, **e) for e in self.tracker.events],
            ball_placements=self.ball_placements,
            epoch=self.epoch,
            physical_resets_after_initialization=len(self.ball_placements),
            robot_state_resets_after_initialization=0,
            initial_referee_launch_field=self.initial_referee_launch,
            scope="explicit simulation referee ball placement; robot state continuous; per-epoch terminal detector",
        )
        return result


def main():
    plant = None
    try:
        for line in sys.stdin:
            try:
                req = json.loads(line)
                op = req["op"]
                if op == "init":
                    if plant is not None:
                        raise ValueError("Reset requires a new worker")
                    plant = MatchFlowPlant(req)
                    result = dict(
                        state=plant.state(),
                        diagnostics=plant.diagnostics(),
                        mujoco=mujoco.__version__,
                        physics_dt=plant.robot.model.opt.timestep,
                    )
                elif op == "arena_status":
                    result = plant.arena_status()
                elif op == "plan":
                    result = dict(positions=plant.plan(req))
                elif op == "end_policy":
                    plant.policy_active = False
                    result = dict(ok=True)
                elif op == "render":
                    result = dict(jpeg=plant.render())
                elif op == "restart_ball":
                    result = plant.restart_ball(req)
                elif op == "goal":
                    plant.goal(req["goal"])
                    plant.steps = 0
                    plant.policy_active = True
                    result = dict(ok=True)
                elif op in ("step", "hold"):
                    result = plant.step(req)
                    result["diagnostics"] = plant.diagnostics()
                elif op == "close":
                    print(json.dumps(dict(ok=True)), flush=True)
                    return
                else:
                    raise ValueError("Unknown match-flow operation")
            except Exception:
                result = dict(error=traceback.format_exc())
            print(json.dumps(result, default=serial, allow_nan=False), flush=True)
    finally:
        if plant is not None and plant.renderer is not None:
            plant.renderer.close()


if __name__ == "__main__":
    main()
