import unittest

from .readiness import Readiness, ReadinessConfig


class TestReadiness(unittest.TestCase):
    def setUp(self):
        self.monitor = Readiness(ReadinessConfig(), 0.75)
        self.good = dict(
            height=0.75,
            tilt=0.0,
            linear_speed=0.0,
            angular_speed=0.0,
            joint_speed=0.0,
            both_feet_loaded=True,
            body_ground_contact=False,
        )

    def feed(self, start, count, **changes):
        result = False
        for i in range(count):
            result = self.monitor.update(start + i * 0.02, self.good | changes)
        return result

    def test_dwell_requires_continuous_physical_samples(self):
        self.assertFalse(self.feed(0.0, 25))
        self.assertTrue(self.monitor.update(0.5, self.good))

    def test_motion_resets_recovery_window(self):
        self.feed(0.0, 24)
        self.assertFalse(self.monitor.update(0.48, self.good | {"linear_speed": 1.0}))
        self.assertFalse(self.feed(0.5, 25))
        self.assertTrue(self.monitor.update(1.0, self.good))

    def test_missing_support_and_body_contact_are_not_ready(self):
        for changes in ({"both_feet_loaded": False}, {"body_ground_contact": True}):
            self.monitor = Readiness(ReadinessConfig(), 0.75)
            self.assertFalse(self.feed(0.0, 100, **changes))

    def test_stale_gap_does_not_count_as_stable_time(self):
        self.monitor.update(0.0, self.good)
        self.assertFalse(self.monitor.update(1.0, self.good))
        self.assertEqual(self.monitor.elapsed, 0.0)

    def test_repeated_or_backward_time_cannot_complete_dwell(self):
        self.feed(0.0, 24)
        self.assertFalse(self.monitor.update(0.46, self.good))
        self.assertFalse(self.monitor.update(0.40, self.good))

    def test_nonfinite_state_cannot_be_ready(self):
        self.assertFalse(self.feed(0.0, 100, height=float("nan")))

    def test_preparation_checks_pose_in_addition_to_stability(self):
        for i in range(100):
            self.assertFalse(self.monitor.update(i * 0.02, self.good, pose_error=0.5))
        self.assertIn("preparation_pose", self.monitor.reasons)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            ReadinessConfig(dwell=0.0)


if __name__ == "__main__":
    unittest.main()
