"""Frozen basic walker inference; all state here is per robot, never plant state."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

RELEASE = Path(
    "/home/tl/Downloads/r1_walking_bootstrap_20261027/release/r1-walk-basic-v0.1"
)
POLICY_SHA256 = "e8203c04b0d0b52bedb945069260bb77926e8c40212bb9ac848f7019c2e08d2c"


class WalkingRuntime:
    def __init__(self, release=RELEASE):
        release = Path(release)
        contract_hash = hashlib.sha256(
            (release / "runtime_contract.json").read_bytes()
        ).hexdigest()
        if (
            contract_hash
            != "8cc4748c7b8a7a212be674b9ae7c450bd467a620f66872ba54552c8a69843de6"
        ):
            raise ValueError("Unexpected walking runtime contract")
        self.contract = json.loads((release / "runtime_contract.json").read_text())
        if (
            hashlib.sha256((release / "policy.onnx").read_bytes()).hexdigest()
            != POLICY_SHA256
        ):
            raise ValueError("Unexpected walking policy")
        self.names = tuple(self.contract["joint_names"])
        if len(self.names) != 24 or len(set(self.names)) != 24:
            raise ValueError("Invalid walking joint contract")
        for field in (
            "kp",
            "kd",
            "effort_limit",
            "action_scale",
            "action_offset",
            "default_joint_pos",
        ):
            value = np.asarray(self.contract[field], float)
            if value.shape != (24,) or not np.isfinite(value).all():
                raise ValueError("Invalid walking contract: " + field)
            setattr(self, field, value)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(release / "policy.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape[-1] != 415:
            raise ValueError("Walking input shape mismatch")
        self.reset_history()

    def reset_history(self):
        self.history = None
        self.last_action = np.zeros(24, dtype=np.float32)
        self.step_index = 0

    @staticmethod
    def inverse_rotate(q, v):
        q = np.asarray(q, float)
        if (
            q.shape != (4,)
            or not np.isfinite(q).all()
            or abs(np.linalg.norm(q) - 1) > 1e-3
        ):
            raise ValueError("Invalid root quaternion")
        return (
            np.asarray(v)
            - 2 * q[0] * np.cross(q[1:], v)
            + 2 * np.cross(q[1:], np.cross(q[1:], v))
        )

    def observation(
        self, joint_names, joint_pos, joint_vel, root_quat, root_ang_vel, command
    ):
        names = tuple(joint_names)
        if len(names) != len(set(names)) or not set(self.names).issubset(names):
            raise ValueError("Missing or duplicated walking state joints")
        order = [names.index(n) for n in self.names]
        positions = np.asarray(joint_pos, float)
        velocities = np.asarray(joint_vel, float)
        angular = np.asarray(root_ang_vel, float)
        if (
            positions.shape != (len(names),)
            or velocities.shape != (len(names),)
            or angular.shape != (3,)
            or not all(np.isfinite(v).all() for v in (positions, velocities, angular))
        ):
            raise ValueError("Invalid walking sensor state")
        cmd = np.asarray(command, np.float32)
        bounds = self.contract["training_command_limits"]
        if cmd.shape != (3,) or not np.isfinite(cmd).all():
            raise ValueError("Invalid velocity command")
        if any(
            cmd[i] < bounds[n][0] - 1e-6 or cmd[i] > bounds[n][1] + 1e-6
            for i, n in enumerate(("vx", "vy", "wz"))
        ):
            raise ValueError("Velocity command exceeds frozen training envelope")
        phase = 2 * np.pi * ((self.step_index * 0.02) % 0.6) / 0.6
        if np.linalg.norm(cmd) < self.contract["standing_phase_freeze_threshold"]:
            phase = 0.0
        values = [
            self.inverse_rotate(root_quat, root_ang_vel)
            * self.contract["observations"][0]["scale"],
            self.inverse_rotate(root_quat, [0.0, 0.0, -1.0]),
            cmd,
            positions[order] - self.default_joint_pos,
            velocities[order] * self.contract["observations"][4]["scale"],
            self.last_action,
            np.array([np.sin(phase), np.cos(phase)]),
        ]
        if self.history is None:
            self.history = [np.tile(x, (5, 1)) for x in values]
        else:
            for history, value in zip(self.history, values):
                history[:-1] = history[1:]
                history[-1] = value
        obs = np.concatenate([h.ravel() for h in self.history]).astype(np.float32)
        if obs.shape != (415,) or not np.isfinite(obs).all():
            raise ValueError("Invalid walking observation")
        return obs

    def target(self, **state):
        observation = self.observation(**state)
        action = self.session.run(None, {self.input: observation[None]})[0][0]
        if action.shape != (24,) or not np.isfinite(action).all():
            raise ValueError("Invalid walking action")
        self.last_action = action.copy()
        self.step_index += 1
        return self.action_offset + self.action_scale * action
