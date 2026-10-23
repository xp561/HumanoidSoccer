"""Fixed-horizon R1 benchmark with explicit geometry and simultaneous scores."""
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import mujoco
from .common_metrics import ShotMetrics
from .interface import RecurrentPolicy, observation
from .model import Robot
from .protocol import noisy_observation, sample_initial, termination_values


def foot_contact(robot):
    ball = robot.model.geom("soccer_ball_geom").id
    for index, contact in enumerate(robot.data.contact):
        other = contact.geom2 if contact.geom1 == ball else contact.geom1 if contact.geom2 == ball else -1
        if other >= 0 and robot.model.geom_bodyid[other] in robot.foot_bodies:
            force = np.empty(6)
            mujoco.mj_contactForce(robot.model, robot.data, index, force)
            if np.linalg.norm(force[:3]) > 0:
                return True
    return False


def initial_overlap(robot):
    ball = robot.model.geom("soccer_ball_geom").id
    bodies = set(robot.body_ids.tolist())
    for contact in robot.data.contact:
        other = contact.geom2 if contact.geom1 == ball else contact.geom1 if contact.geom2 == ball else -1
        if other >= 0 and robot.model.geom_bodyid[other] in bodies:
            return True
    return False


class Benchmark:
    def __init__(self, repo, artifacts, runtime_path, model_path, profile):
        self.repo = Path(repo)
        self.artifacts = Path(artifacts)
        self.runtime_path = Path(runtime_path)
        self.model_path = Path(model_path)
        self.profile = dict(profile)
        self.metadata = json.loads((self.artifacts / "metadata.json").read_text())
        self.runtime = json.loads(self.runtime_path.read_text())
        self.runtime["enable_corruption"] = profile["observation_noise"]
        if not json.loads((self.artifacts / "validation.json").read_text())["passed"]:
            raise RuntimeError("Interface validation must pass")
        self.actor = RecurrentPolicy(self.artifacts / "policy.onnx", self.metadata)
        self.robot = Robot(self.model_path, self.metadata)
        self.motions = []
        self.reference_ids = []
        self.root_ids = []
        for name in self.metadata["motion_names"]:
            with np.load(self.repo / "motions/r1-corrected" / (name + ".npz")) as archive:
                motion = {key: archive[key] for key in archive.files}
            if list(motion["joint_names"]) != self.metadata["joint_names"]:
                raise ValueError("Motion joint order does not match runtime metadata")
            if abs(float(motion["fps"].reshape(-1)[0]) * self.metadata["control_dt"] - 1) > 1e-6:
                raise ValueError("Reference and policy frequencies differ")
            self.motions.append(motion)
            self.root_ids.append(list(motion["body_names"]).index("pelvis_link"))
            self.reference_ids.append([list(motion["body_names"]).index(n) for n in self.runtime["reference_body_names"]])
        self.nominal = np.asarray(self.metadata["motion_nominal_ball_xy"]).copy()
        if profile["initial_center"] == "world_origin":
            for index, motion in enumerate(self.motions):
                self.nominal[index] -= motion["body_pos_w"][0, self.root_ids[index], :2]
        elif profile["initial_center"] != "reference_root":
            raise ValueError("Unknown initial disk center")
        model = self.robot.model
        ball = model.geom("soccer_ball_geom").id
        self.physics_dt = float(model.opt.timestep)
        if abs(self.physics_dt - profile["physics_dt"]) > 1e-12:
            raise ValueError("Physics timestep does not match profile")
        self.control_dt = self.metadata["control_dt"]
        self.substeps = round(self.control_dt / self.physics_dt)
        if abs(self.substeps * self.physics_dt - self.control_dt) > 1e-12:
            raise ValueError("Control decimation is not integral")
        self.steps = round(profile["horizon_seconds"] / self.control_dt)
        if abs(self.steps * self.control_dt - profile["horizon_seconds"]) > 1e-12:
            raise ValueError("Horizon is not an integral control step count")
        body = model.geom_bodyid[ball]
        if abs(float(model.body_mass[body]) - profile["ball_mass"]) > 1e-8:
            raise ValueError("Ball mass does not match profile")
        if abs(float(model.geom_size[ball, 0]) - profile["ball_radius"]) > 1e-8:
            raise ValueError("Ball radius does not match profile")

    def run(self, case, output, *, save_trace=False):
        meta, runtime, robot, profile = self.metadata, self.runtime, self.robot, self.profile
        selected = int(np.argmin(np.linalg.norm(self.nominal - [case["lateral"], -case["forward"]], axis=1)))
        motion = self.motions[selected]
        root_index = self.root_ids[selected]
        self.actor.reset()
        initial_rng = np.random.default_rng(case["seed"])
        noise_rng = np.random.default_rng(np.random.SeedSequence([case["seed"], 1]))
        for attempt in range(profile["spawn_max_attempts"]):
            offset, ball_velocity = sample_initial(initial_rng, profile["robot_initial_radius"], case["speed_range"])
            root = motion["body_pos_w"][0, root_index].copy()
            if profile["initial_center"] == "world_origin":
                root[:2] = offset
            else:
                root[:2] += offset
            limits = np.asarray(runtime["soft_joint_pos_limits"])
            robot.set_state(root_pos=root, root_quat=motion["body_quat_w"][0, root_index],
                            joint_pos=np.clip(motion["joint_pos"][0], limits[:, 0], limits[:, 1]),
                            joint_vel=motion["joint_vel"][0],
                            root_lin_vel=motion["body_lin_vel_w"][0, root_index],
                            root_ang_vel=motion["body_ang_vel_w"][0, root_index],
                            ball_pos=[case["lateral"], -case["forward"], profile["ball_radius"]],
                            ball_vel=ball_velocity)
            if not profile["reject_initial_overlap"] or not initial_overlap(robot):
                break
        else:
            raise RuntimeError("No non-overlapping initialization within declared attempt budget")
        initial_root = robot.state()["root_pos"].copy()
        initial_ball = robot.state()["ball_pos"].copy()
        goal = np.asarray(profile["goal_world"], dtype=float)
        normal = goal[:2] / np.linalg.norm(goal[:2])
        metric = ShotMetrics(goal, normal, profile["goal_width"], profile["outgoing_speed_threshold"])
        sys.path.insert(0, str(self.repo / "scripts/rsl_rl"))
        from r1_eval_metrics import KickTrial
        legacy = KickTrial()
        history = round(runtime["contact_history_length"] * runtime["contact_update_dt"] / self.physics_dt)
        contacts = deque([False] * history, maxlen=history)
        previous_control = initial_ball.copy()
        last = np.zeros(len(meta["action_joint_names"]), dtype=np.float32)
        traces = {}
        first_tracking_flags = None
        max_joint_speed = np.zeros(len(meta["joint_names"]))
        bad_warnings = [mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC]
        for step in range(self.steps):
            frame = min(step, len(motion["joint_pos"]) - 1)
            state = robot.state()
            clean = observation(meta, **state, ref_joint_pos=motion["joint_pos"][frame],
                                ref_joint_vel=motion["joint_vel"][frame],
                                ref_ang_vel=motion["body_ang_vel_w"][frame, root_index],
                                last_action=last, goal_pos=goal)
            obs = noisy_observation(clean, meta, runtime, noise_rng)
            action = self.actor.step(obs)
            robot.action(action)
            last = action
            buffer = []
            for _ in range(self.substeps):
                previous = robot.data.qpos[robot.ball_q:robot.ball_q+3].copy()
                warning_counts = [robot.data.warning[w].number for w in bad_warnings]
                mujoco.mj_step(robot.model, robot.data)
                bad = (not np.isfinite(robot.data.qpos).all() or not np.isfinite(robot.data.qvel).all()
                       or any(robot.data.warning[w].number > number for w, number in zip(bad_warnings, warning_counts)))
                if bad:
                    metric.numerical_failure = True
                    break
                current = robot.data.qpos[robot.ball_q:robot.ball_q+3].copy()
                velocity = robot.data.qvel[robot.ball_v:robot.ball_v+3].copy()
                contact = foot_contact(robot)
                contacts.append(contact)
                max_joint_speed = np.maximum(max_joint_speed, np.abs(robot.data.qvel[robot.dof_ids]))
                buffer.append((previous, current, velocity, contact, float(robot.data.time)))
            if metric.numerical_failure:
                break
            mujoco.mj_forward(robot.model, robot.data)
            state = robot.state()
            flags, errors = termination_values(runtime, state["root_pos"], state["root_quat"],
                robot.data.xpos[robot.body_ids], motion["body_pos_w"][frame, self.reference_ids[selected]],
                motion["body_quat_w"][frame, self.reference_ids[selected]], step+1)
            if any(flags.values()) and metric.tracking_failure_time is None:
                metric.mark_tracking_failure(float(robot.data.time))
                first_tracking_flags = {name: flag for name, flag in flags.items() if flag}
            for sample in buffer:
                metric.observe(*sample)
            current = state["ball_pos"]
            velocity = robot.data.qvel[robot.ball_v:robot.ball_v+3].copy()
            if any(flags.values()):
                legacy.finish("episode_end")
            else:
                legacy.observe(previous_control, current, velocity, goal, normal,
                               profile["goal_width"], profile["outgoing_speed_threshold"], any(contacts))
            previous_control = current.copy()
            if save_trace:
                for key, value in dict(root_pos=state["root_pos"], root_quat=state["root_quat"],
                                       joint_pos=state["joint_pos"], joint_vel=state["joint_vel"],
                                       ball_pos=current, ball_vel=velocity, action=action, obs=obs).items():
                    traces.setdefault(key, []).append(np.array(value, copy=True))
        result = {**case, **metric.result(), "legacy_strict_goal": bool(legacy.success),
                  "legacy_strict_contact": bool(legacy.contacted),
                  "legacy_strict_accuracy_cosine": float(legacy.kick_accuracy_cosine) if np.isfinite(legacy.kick_accuracy_cosine) else None,
                  "motion": meta["motion_names"][selected], "initial_root": initial_root.tolist(),
                  "initial_ball": initial_ball.tolist(), "initial_offset": offset.tolist(),
                  "initial_ball_velocity": ball_velocity.tolist(), "spawn_attempts": attempt+1,
                  "control_steps": step+1, "seconds": float(robot.data.time),
                  "first_tracking_flags": first_tracking_flags, "max_joint_speed": max_joint_speed.tolist(),
                  "model": str(self.model_path), "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()}
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        if save_trace:
            np.savez(output.with_suffix(".npz"), **{key: np.stack(values) for key, values in traces.items()})
        return result
