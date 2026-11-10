"""MuJoCo worker using the original inference environment and physics assets."""

import json
import sys
import traceback
from pathlib import Path

import mujoco
import numpy as np

from exp.r1_sim2sim.benchmark import foot_contact
from exp.r1_sim2sim.common_metrics import ShotMetrics
from exp.r1_sim2sim.model import Robot
from exp.r1_sim2sim.protocol import termination_values


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Plant:
    def __init__(self, request, *, model_path=None):
        root = Path.cwd()
        formal = root / "logs/r1_logic_fix/r1_mujoco_formal_20261017"
        self.meta = json.loads((formal / "artifacts/metadata.json").read_text())
        self.protocol = json.loads((formal / "runtime_protocol.json").read_text())
        self.profile = json.loads((formal / "profile.json").read_text())
        self.robot = Robot(formal / "model.xml" if model_path is None else model_path, self.meta)
        self.walk_contract = None
        if request.get("walking", False):
            from .walking_runtime import WalkingRuntime
            self.walk_contract = WalkingRuntime().contract
        index = request["motion"]
        self.name = self.meta["motion_names"][index]
        with np.load(root / "motions/r1-corrected" / (self.name + ".npz")) as archive:
            self.motion = {name: archive[name] for name in archive.files}
        m = self.motion
        self.root_id = list(m["body_names"]).index("pelvis_link")
        self.reference_ids = [list(m["body_names"]).index(n) for n in self.protocol["reference_body_names"]]
        self.initial_root = m["body_pos_w"][0, self.root_id].copy()
        self.initial_root[:2] = 0
        original_root = m["body_pos_w"][0, self.root_id]
        ball = np.array([*self.meta["motion_nominal_ball_xy"][index], self.profile["ball_radius"]])
        ball[:2] -= original_root[:2]
        ball[:2] += np.asarray(request.get("ball_offset", [0, 0]))
        limits = np.asarray(self.protocol["soft_joint_pos_limits"])
        self.robot.set_state(
            root_pos=self.initial_root,
            root_quat=m["body_quat_w"][0, self.root_id],
            joint_pos=np.clip(m["joint_pos"][0], limits[:, 0], limits[:, 1]),
            joint_vel=m["joint_vel"][0],
            root_lin_vel=m["body_lin_vel_w"][0, self.root_id],
            root_ang_vel=m["body_ang_vel_w"][0, self.root_id],
            ball_pos=ball,
            ball_vel=request.get("ball_velocity", [0, 0, 0]),
        )
        self.steps = 0
        self.metric = None
        self.substeps = round(self.meta["control_dt"] / self.robot.model.opt.timestep)
        if abs(self.substeps * self.robot.model.opt.timestep - self.meta["control_dt"]) > 1e-12:
            raise ValueError("Nonintegral control decimation")
        if abs(self.robot.model.opt.timestep - self.profile["physics_dt"]) > 1e-12:
            raise ValueError("Physics timestep differs from frozen profile")
        self.min_root_height = float(self.initial_root[2])

    def state(self):
        return {
            **self.robot.state(),
            "ball_velocity": self.robot.data.qvel[self.robot.ball_v : self.robot.ball_v + 3].copy(),
        }

    def goal(self, goal):
        goal = np.asarray(goal, float)
        self.metric = ShotMetrics(
            goal, goal[:2] - self.initial_root[:2], self.profile["goal_width"], self.profile["outgoing_speed_threshold"]
        )

    def step(self, request):
        robot = self.robot
        if request["op"] == "hold":
            robot.data.ctrl[robot.action_ids] = robot.data.qpos[robot.qpos_ids][self.meta["action_joint_ids"]]
        else:
            names = request["joint_names"]
            if len(names) != len(set(names)) or set(names) != set(self.meta["action_joint_names"]):
                raise ValueError("Command does not cover action joints exactly")
            order = [names.index(name) for name in self.meta["action_joint_names"]]
            profiles = [dict(kp=np.asarray(self.meta["stiffness"])[self.meta["action_joint_ids"]],
                             kd=np.asarray(self.meta["damping"])[self.meta["action_joint_ids"]],
                             max_torques=np.asarray(self.meta["effort_limits"])[self.meta["action_joint_ids"]])]
            if self.walk_contract is not None:
                c = self.walk_contract
                walk_order = [c["joint_names"].index(n) for n in self.meta["action_joint_names"]]
                profiles.append({key:np.asarray(c[field])[walk_order] for key,field in
                                 (("kp","kp"),("kd","kd"),("max_torques","effort_limit"))})
            actual = {key:np.asarray(request[key], float)[order] for key in profiles[0]}
            if not any(all(np.array_equal(actual[key], profile[key]) for key in profile) for profile in profiles):
                raise ValueError("Plant/controller gain or effort contract mismatch")
            ids = robot.action_ids
            robot.model.actuator_gainprm[ids, 0] = actual["kp"]
            robot.model.actuator_biasprm[ids, 1] = -actual["kp"]
            robot.model.actuator_biasprm[ids, 2] = -actual["kd"]
            robot.model.actuator_forcerange[ids, 0] = -actual["max_torques"]
            robot.model.actuator_forcerange[ids, 1] = actual["max_torques"]
            target = np.asarray(request["positions"], float)
            if target.shape != (len(order),) or not np.isfinite(target).all():
                raise ValueError("Invalid targets")
            robot.data.ctrl[robot.action_ids] = target[order]
        buffer = []
        bad_warnings = [
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
        ]
        observer = getattr(self, "physics_observer", None)
        for _ in range(self.substeps):
            before = robot.data.qpos[robot.ball_q : robot.ball_q + 3].copy()
            counts = [robot.data.warning[w].number for w in bad_warnings]
            mujoco.mj_step(robot.model, robot.data)
            if (
                not np.isfinite(robot.data.qpos).all()
                or not np.isfinite(robot.data.qvel).all()
                or any(robot.data.warning[w].number > count for w, count in zip(bad_warnings, counts, strict=True))
            ):
                raise RuntimeError("Numerically invalid physical step")
            after = robot.data.qpos[robot.ball_q : robot.ball_q + 3].copy()
            if observer is not None:
                observer(before, after, float(robot.data.time))
            velocity = robot.data.qvel[robot.ball_v : robot.ball_v + 3].copy()
            buffer.append((before, after, velocity, foot_contact(robot), float(robot.data.time)))
        mujoco.mj_forward(robot.model, robot.data)
        state = self.state()
        frame = min(self.steps, len(self.motion["joint_pos"]) - 1)
        flags, _ = termination_values(
            self.protocol,
            state["root_pos"],
            state["root_quat"],
            robot.data.xpos[robot.body_ids],
            self.motion["body_pos_w"][frame, self.reference_ids],
            self.motion["body_quat_w"][frame, self.reference_ids],
            self.steps + 1,
        )
        if self.metric is not None:
            if any(flags.values()) and self.metric.tracking_failure_time is None:
                self.metric.mark_tracking_failure(float(robot.data.time))
            for sample in buffer:
                self.metric.observe(*sample)
        self.steps += 1
        self.min_root_height = min(self.min_root_height, float(state["root_pos"][2]))
        return dict(
            state=state,
            seconds=float(robot.data.time),
            control_steps=self.steps,
            min_root_height=self.min_root_height,
            metrics=None if self.metric is None else self.metric.result(),
        )


def main():
    plant = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "init":
                plant = Plant(request)
                result = dict(
                    state=plant.state(),
                    motion=plant.name,
                    mujoco=mujoco.__version__,
                    physics_dt=float(plant.robot.model.opt.timestep),
                )
            elif op == "goal":
                plant.goal(request["goal"])
                result = dict(ok=True)
            elif op in ("step", "hold"):
                result = plant.step(request)
            elif op == "close":
                print(json.dumps(dict(ok=True)), flush=True)
                return
            else:
                raise ValueError("Unknown operation")
        except Exception:
            result = dict(error=traceback.format_exc())
        print(json.dumps(result, default=serial, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
