"""Offline checks: no ROS publisher or hardware access."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from exp.r1_bitbots_adapter.runtime import Reference, Runtime, anchored_goal, fresh_stamp

BUNDLE = Path("logs/r1_logic_fix/r1_mujoco_formal_20261017/artifacts")


class TestAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = Runtime(BUNDLE)

    def setUp(self):
        self.r = self.runtime
        self.r.reset()

    def test_joint_state_permutation(self):
        n = len(self.r.names)
        for seed in range(10):
            order = np.random.default_rng(seed).permutation(n)
            q, dq = self.r.named_joints([self.r.names[i] for i in order], order, -order)
            np.testing.assert_array_equal(q, np.arange(n))
            np.testing.assert_array_equal(dq, -np.arange(n))

    def test_missing_head_is_rejected(self):
        for head in ("head_pitch_joint", "head_yaw_joint"):
            with self.subTest(joint=head):
                names = [name for name in self.r.names if name != head]
                with self.assertRaisesRegex(ValueError, head):
                    self.r.named_joints(names, np.zeros(len(names)), np.zeros(len(names)))

    def test_duplicate_names_rejected(self):
        names = list(self.r.names)
        names[-1] = names[0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.r.named_joints(names, np.zeros(26), np.zeros(26))

    def test_nonfinite_joint_state_rejected(self):
        q = np.zeros(26)
        q[3] = np.nan
        with self.assertRaises(ValueError):
            self.r.named_joints(self.r.names, q, np.zeros(26))

    def test_target_order_and_gains(self):
        a = np.linspace(-1, 1, 24)
        target = self.r.targets(a)
        meta = self.r.meta
        for i, name in enumerate(target.joint_names):
            j = meta["joint_names"].index(name)
            self.assertEqual(target.kp[i], meta["stiffness"][j])
            self.assertEqual(target.kd[i], meta["damping"][j])
            self.assertEqual(target.effort_limits[i], meta["effort_limits"][j])
            self.assertEqual(target.positions[i], meta["action_offset"][i] + meta["action_scale"][i] * a[i])
        self.assertEqual(len(target.positions), 24)

    def test_invalid_action_rejected(self):
        for action in (np.zeros(26), np.full(24, np.nan)):
            with self.assertRaises(ValueError):
                self.r.targets(action)

    def test_reset_clears_history(self):
        self.r.step(np.zeros(143))
        self.r.reset()
        for state in (self.r.h, self.r.c, self.r.last_action):
            self.assertFalse(np.any(state))

    def test_bad_recurrent_output_not_committed(self):
        real = self.r.session
        h = self.r.h.copy()
        c = self.r.c.copy()
        previous = self.r.last_action.copy()
        self.r.session = SimpleNamespace(run=lambda *args: [np.zeros((1, 24), np.float32), np.full_like(h, np.nan), c])
        try:
            with self.assertRaises(ValueError):
                self.r.step(np.zeros(143))
            np.testing.assert_array_equal(self.r.h, h)
            np.testing.assert_array_equal(self.r.c, c)
            np.testing.assert_array_equal(self.r.last_action, previous)
        finally:
            self.r.session = real

    def test_reference_time_clamps_and_does_not_wrap(self):
        path = Path("motions/r1-corrected") / (self.r.meta["motion_names"][0] + ".npz")
        ref = Reference(path, self.r)
        for i in range(ref.frames + 30):
            self.assertEqual(ref.frame(i * self.r.dt), min(i, ref.frames - 1))
        self.assertTrue(ref.ended(ref.frames * self.r.dt))
        self.assertFalse(ref.ended((ref.frames - 1) * self.r.dt))
        with self.assertRaises(ValueError):
            ref.frame(-self.r.dt)

    def test_body_heading_anchors_in_world(self):
        half = np.pi / 4
        q = [np.cos(half), 0, 0, np.sin(half)]
        goal = anchored_goal([1, 0], q, [1, 2, 0.8], 5, 0.11)
        np.testing.assert_allclose(goal, [1, 7, 0.11], atol=1e-12)
        for direction in ([0, 0], [float("nan"), 0]):
            with self.assertRaises(ValueError):
                anchored_goal(direction, q, [1, 2, 0.8], 5, 0.11)

    def test_missing_future_stale_sensors_rejected(self):
        fresh_stamp(10, 9.95, 0.1)
        for stamp in (0, 10.01, 9.8, float("nan")):
            with self.assertRaises(ValueError):
                fresh_stamp(10, stamp, 0.1)

    def test_ball_position_uses_full_pelvis_rotation(self):
        args = dict(
            ref_joint_pos=np.zeros(26),
            ref_joint_vel=np.zeros(26),
            ref_ang_vel=np.zeros(3),
            root_pos=[1, 2, 0.8],
            root_quat=[1, 0, 0, 0],
            root_ang_vel=np.zeros(3),
            joint_pos=self.r.default,
            joint_vel=np.zeros(26),
            ball_pos=[2, 2, 0.11],
            goal_pos=[6, 2, 0.11],
        )
        obs = self.r.observation(**args)
        np.testing.assert_allclose(obs[-6:-3], [1, 0, -0.69], atol=1e-7)
        self.assertEqual(obs.shape, (143,))

    def test_invalid_metadata_contracts_rejected(self):
        import copy
        import tempfile

        mutations = [
            lambda m: m["action_joint_ids"].__setitem__(0, -1),
            lambda m: m["action_joint_ids"].__setitem__(0, 0.5),
            lambda m: m["action_joint_names"].reverse(),
            lambda m: m["joint_names"].__setitem__(0, m["joint_names"][1]),
            lambda m: m["action_scale"].__setitem__(0, 0),
            lambda m: m.__setitem__("control_dt", -1),
            lambda m: m["observation_terms"].reverse(),
        ]
        for mutate in mutations:
            meta = copy.deepcopy(self.r.meta)
            mutate(meta)
            with self.subTest(mutation=mutations.index(mutate)):
                with tempfile.TemporaryDirectory() as folder:
                    Path(folder, "metadata.json").write_text(json.dumps(meta))
                    with self.assertRaises(ValueError):
                        Runtime(folder)

    def test_nonfinite_observation_preserves_history(self):
        h, c = self.r.h.copy(), self.r.c.copy()
        with self.assertRaises(ValueError):
            self.r.step(np.full(143, np.inf))
        np.testing.assert_array_equal(self.r.h, h)
        np.testing.assert_array_equal(self.r.c, c)


if __name__ == "__main__":
    unittest.main()
