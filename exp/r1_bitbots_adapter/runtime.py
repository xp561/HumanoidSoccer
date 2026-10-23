"""Named-joint adapter for the validated recurrent R1 export."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from exp.r1_sim2sim.interface import inverse_rotate, vector

TERMS = [
    "command", "projected_gravity", "motion_ref_ang_vel", "base_ang_vel",
    "joint_pos", "joint_vel", "actions", "target_point_pos",
    "target_destination_pos_local",
]


@dataclass(frozen=True)
class JointTargets:
    joint_names: tuple
    positions: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limits: np.ndarray


class Runtime:
    def __init__(self, bundle):
        import onnxruntime as ort
        bundle = Path(bundle)
        self.meta = json.loads((bundle / "metadata.json").read_text())
        m = self.meta
        self.names = tuple(m["joint_names"])
        self.action_names = tuple(m["action_joint_names"])
        n, a = len(self.names), len(self.action_names)
        if len(set(self.names)) != n or len(set(self.action_names)) != a:
            raise ValueError("Duplicate joint names")
        raw_ids = np.asarray(m["action_joint_ids"])
        if raw_ids.shape != (a,) or not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("Invalid action joint indices")
        self.ids = raw_ids.astype(int)
        if np.any(self.ids < 0) or np.any(self.ids >= n):
            raise ValueError("Action joint index out of bounds")
        if tuple(self.names[i] for i in self.ids) != self.action_names:
            raise ValueError("Action names and indices disagree")
        if m["observation_terms"] != TERMS:
            raise ValueError("Unsupported observation ordering")
        self.default = vector(m["default_joint_pos"], n, "defaults")
        self.offset = vector(m["action_offset"], a, "offset")
        self.scale = vector(m["action_scale"], a, "scale")
        if np.any(self.scale == 0):
            raise ValueError("Zero action scale")
        self.kp = vector(m["stiffness"], n, "kp")[self.ids]
        self.kd = vector(m["damping"], n, "kd")[self.ids]
        self.effort = vector(m["effort_limits"], n, "effort")[self.ids]
        if np.any(self.kp < 0) or np.any(self.kd < 0) or np.any(self.effort <= 0):
            raise ValueError("Invalid joint gains or effort limits")
        self.dt = float(m["control_dt"])
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Invalid policy period")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(bundle / "policy.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        inputs = {x.name: x for x in self.session.get_inputs()}
        outputs = {x.name: x for x in self.session.get_outputs()}
        self.obs_dim = 4*n + a + 15
        if set(inputs) != {"obs", "h_in", "c_in"}:
            raise ValueError("Expected explicit recurrent inputs")
        if set(outputs) != {"actions", "h_out", "c_out"}:
            raise ValueError("Expected explicit recurrent outputs")
        self.state_shape = tuple(inputs["h_in"].shape)
        expected_shapes = {
            "obs": (1, self.obs_dim), "h_in": self.state_shape,
            "c_in": self.state_shape, "actions": (1, a),
            "h_out": self.state_shape, "c_out": self.state_shape,
        }
        if len(self.state_shape) != 3 or self.state_shape[1] != 1:
            raise ValueError("Unsupported recurrent state layout")
        for name, value in {**inputs, **outputs}.items():
            if tuple(value.shape) != expected_shapes[name] or value.type != "tensor(float)":
                raise ValueError("Incompatible ONNX tensor: " + name)
        self.reset()

    def reset(self):
        self.h = np.zeros(self.state_shape, np.float32)
        self.c = np.zeros(self.state_shape, np.float32)
        self.last_action = np.zeros(len(self.action_names), np.float32)

    def named_joints(self, names, positions, velocities):
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError("Duplicate incoming joint names")
        q = vector(positions, len(names), "incoming positions")
        dq = vector(velocities, len(names), "incoming velocities")
        lookup = {name: i for i, name in enumerate(names)}
        missing = set(self.names) - lookup.keys()
        if missing:
            raise ValueError("Missing observed joints: " + ", ".join(sorted(missing)))
        indices = [lookup[name] for name in self.names]
        return q[indices], dq[indices]

    def observation(self, *, ref_joint_pos, ref_joint_vel, ref_ang_vel,
                    root_pos, root_quat, root_ang_vel, joint_pos, joint_vel,
                    ball_pos, goal_pos, last_action=None):
        n = len(self.names)
        root = vector(root_pos, 3, "root position")
        quat = vector(root_quat, 4, "wxyz root quaternion")
        previous = self.last_action if last_action is None else last_action
        parts = (
            vector(ref_joint_pos, n, "reference position"),
            vector(ref_joint_vel, n, "reference velocity"),
            inverse_rotate(quat, [0, 0, -1]),
            vector(ref_ang_vel, 3, "reference world angular velocity"),
            inverse_rotate(quat, vector(root_ang_vel, 3, "world angular velocity")),
            vector(joint_pos, n, "joint position") - self.default,
            vector(joint_vel, n, "joint velocity"),
            vector(previous, len(self.action_names), "previous action"),
            inverse_rotate(quat, vector(ball_pos, 3, "world ball center") - root),
            inverse_rotate(quat, vector(goal_pos, 3, "world goal") - root),
        )
        result = np.concatenate(parts).astype(np.float32)
        return vector(result, self.obs_dim, "observation").astype(np.float32)

    def step(self, obs):
        x = vector(obs, self.obs_dim, "observation").astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Observation overflow")
        action, h, c = self.session.run(
            ["actions", "h_out", "c_out"],
            {"obs": x[None], "h_in": self.h, "c_in": self.c})
        if action.shape != (1, len(self.action_names)):
            raise ValueError("Invalid action shape")
        for state in (h, c):
            if state.shape != self.state_shape or not np.isfinite(state).all():
                raise ValueError("Invalid recurrent output")
        action = vector(action[0], len(self.action_names), "action").astype(np.float32)
        self.targets(action)
        self.h, self.c = h, c
        self.last_action = action.copy()
        return action

    def targets(self, action):
        a = vector(action, len(self.action_names), "action")
        positions = self.offset + self.scale*a
        vector(positions, len(self.action_names), "target positions")
        return JointTargets(self.action_names, positions, self.kp.copy(),
                            self.kd.copy(), self.effort.copy())


class Reference:
    """Reference progression uses elapsed simulation time and never wraps."""
    def __init__(self, path, runtime):
        with np.load(path, allow_pickle=False) as archive:
            self.data = {key: archive[key] for key in archive.files}
        d = self.data
        if tuple(d["joint_names"]) != runtime.names:
            raise ValueError("Reference joint order mismatch")
        self.root = list(d["body_names"]).index("pelvis_link")
        fps = vector(np.asarray(d["fps"]).reshape(-1), 1, "fps")[0]
        if abs(fps*runtime.dt - 1) > 1e-6:
            raise ValueError("Reference frequency mismatch")
        self.dt = runtime.dt
        self.frames = len(d["joint_pos"])
        if self.frames == 0:
            raise ValueError("Empty reference")
        for key in ("joint_pos", "joint_vel"):
            if d[key].shape != (self.frames, len(runtime.names)) or not np.isfinite(d[key]).all():
                raise ValueError("Invalid reference " + key)
        for key in ("body_pos_w", "body_ang_vel_w"):
            if d[key].shape != (self.frames, len(d["body_names"]), 3) or not np.isfinite(d[key]).all():
                raise ValueError("Invalid reference " + key)

    def frame(self, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid elapsed simulation time")
        return min(int(np.floor(elapsed/self.dt + 1e-9)), self.frames-1)

    def sample(self, elapsed):
        frame = self.frame(elapsed)
        return dict(ref_joint_pos=self.data["joint_pos"][frame],
                    ref_joint_vel=self.data["joint_vel"][frame],
                    ref_ang_vel=self.data["body_ang_vel_w"][frame, self.root])

    def ended(self, elapsed):
        self.frame(elapsed)
        return elapsed >= self.frames*self.dt


def anchored_goal(direction_xy, root_quat_wxyz, root_world, distance, height):
    """Convert a body heading command into a fixed world goal point.

    Distance and height are explicit task configuration, not network strength.
    """
    direction = vector(direction_xy, 2, "kick heading")
    if np.linalg.norm(direction) < 1e-12:
        raise ValueError("Zero kick heading")
    q = vector(root_quat_wxyz, 4, "root orientation")
    q = q / np.linalg.norm(q) if np.linalg.norm(q) > 1e-12 else q
    inverse_rotate(q, [0, 0, 1])
    yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
    heading = yaw + np.arctan2(direction[1], direction[0])
    root = vector(root_world, 3, "root position")
    if not np.isfinite(distance) or distance <= 0 or not np.isfinite(height):
        raise ValueError("Invalid goal geometry")
    return np.array([root[0]+distance*np.cos(heading),
                     root[1]+distance*np.sin(heading), height])


def fresh_stamp(now, stamp, max_age):
    if not np.isfinite([now, stamp, max_age]).all() or max_age <= 0:
        raise ValueError("Invalid freshness parameters")
    if stamp <= 0 or stamp > now or now-stamp > max_age:
        raise ValueError("Missing, future or stale sensor timestamp")
