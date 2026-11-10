"""Contract and execution tests for the independent walker and PD switching."""

import unittest
import numpy as np
from .walking_runtime import WalkingRuntime


class TestWalking(unittest.TestCase):
    def setUp(self):
        self.w = WalkingRuntime()
        self.state = dict(
            joint_names=self.w.names,
            joint_pos=self.w.default_joint_pos.copy(),
            joint_vel=np.zeros(24),
            root_quat=[1, 0, 0, 0],
            root_ang_vel=[1, 2, 3],
            command=[0, 0, 0],
        )

    def test_named_state_permutation_and_body_frame(self):
        ref = self.w.observation(**self.state)
        self.w.reset_history()
        order = np.random.default_rng(561).permutation(24)
        self.state.update(
            joint_names=[self.w.names[i] for i in order],
            joint_pos=self.w.default_joint_pos[order],
            root_quat=[2**-0.5, 0, 0, 2**-0.5],
            root_ang_vel=[-2, 1, 3],
        )
        np.testing.assert_allclose(self.w.observation(**self.state), ref, atol=1e-7)

    def test_term_history_and_stopped_phase(self):
        first = self.w.observation(**self.state)
        self.w.step_index = 7
        self.state["joint_pos"] = self.w.default_joint_pos + 1
        second = self.w.observation(**self.state)
        np.testing.assert_array_equal(
            second[45:165].reshape(5, 24)[:4], np.zeros((4, 24))
        )
        np.testing.assert_array_equal(second[45:165].reshape(5, 24)[4], np.ones(24))
        np.testing.assert_array_equal(
            second[405:].reshape(5, 2), np.tile([0, 1], (5, 1))
        )
        self.state["command"] = [0.2, 0, 0]
        third = self.w.observation(**self.state)
        self.assertGreater(abs(third[413]), 0.5)
        self.assertEqual(first.shape, (415,))

    def test_invalid_state_and_commands(self):
        for command in ([np.nan, 0, 0], [0.41, 0, 0], [0, 0, 0.51]):
            with self.assertRaises(ValueError):
                self.w.observation(**dict(self.state, command=command))
        with self.assertRaises(ValueError):
            self.w.observation(**dict(self.state, joint_names=self.w.names[:-1]))

    def test_physics_validates_and_applies_both_contracts(self):
        from .live_worker import LivePlant

        p = LivePlant(dict(motion=0, walking=True))
        r = p.robot
        cmd = dict(
            op="step",
            joint_names=self.w.names,
            positions=self.w.default_joint_pos.tolist(),
            kp=self.w.kp.tolist(),
            kd=self.w.kd.tolist(),
            max_torques=self.w.effort_limit.tolist(),
        )
        p.step(cmd)
        ids = [r.model.actuator(n).id for n in self.w.names]
        np.testing.assert_array_equal(r.model.actuator_gainprm[ids, 0], self.w.kp)
        np.testing.assert_array_equal(r.model.actuator_biasprm[ids, 2], -self.w.kd)
        cmd["kd"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            p.step(cmd)
        meta = p.meta
        active = meta["action_joint_ids"]
        cmd = dict(
            op="step",
            joint_names=meta["action_joint_names"],
            positions=r.data.qpos[r.qpos_ids][active].tolist(),
            kp=np.asarray(meta["stiffness"])[active].tolist(),
            kd=np.asarray(meta["damping"])[active].tolist(),
            max_torques=np.asarray(meta["effort_limits"])[active].tolist(),
        )
        p.step(cmd)
        np.testing.assert_array_equal(
            r.model.actuator_gainprm[r.action_ids, 0], cmd["kp"]
        )
        np.testing.assert_array_equal(
            r.model.actuator_biasprm[r.action_ids, 2], -np.asarray(cmd["kd"])
        )
        p2 = LivePlant(dict(motion=0))
        cmd.update(
            joint_names=self.w.names,
            kp=self.w.kp.tolist(),
            kd=self.w.kd.tolist(),
            max_torques=self.w.effort_limit.tolist(),
        )
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            p2.step(cmd)


if __name__ == "__main__":
    unittest.main()
