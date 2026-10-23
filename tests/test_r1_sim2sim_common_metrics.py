"""Scoring boundaries and event ordering that affect published comparisons."""
import unittest
import numpy as np
from exp.r1_sim2sim.common_metrics import ShotMetrics, goal_crossed


def trial():
    return ShotMetrics(np.array([0., -5., .11]), np.array([0., -1.]), 1., .5)


class ScoringTests(unittest.TestCase):
    def test_directed_crossing_and_boundary(self):
        for x, expected in [(.5, True), (.500001, False), (-.5, True)]:
            self.assertEqual(goal_crossed([x,-4.9,.11], [x,-5.1,.11], [0,-5,.11], [0,-1], 1), expected)
        self.assertFalse(goal_crossed([0,-5.1,.11], [0,-4.9,.11], [0,-5,.11], [0,-1], 1))

    def test_interpolates_crossing_instead_of_endpoint(self):
        self.assertTrue(goal_crossed([0,-4,.11], [1,-6,.11], [0,-5,.11], [0,-1], 1))

    def test_normal_scale_does_not_change_geometry(self):
        self.assertEqual(goal_crossed([0,-4,.11], [0,-6,.11], [0,-5,.11], [0,-1], 1),
                         goal_crossed([0,-4,.11], [0,-6,.11], [0,-5,.11], [0,-7], 1))

    def test_unassisted_rolling_goal_is_identified(self):
        t = trial()
        t.observe([0,-4.9,.11], [0,-5.1,.11], [0,-.3,0], False, .1)
        r = t.result()
        self.assertTrue(r["goal"])
        self.assertFalse(r["kicked_goal"])
        self.assertIsNone(r["kick_accuracy_cosine"])

    def test_contact_in_crossing_sample_counts_as_kick(self):
        t = trial()
        t.observe([0,-4.9,.11], [0,-5.1,.11], [0,-1,0], True, .1)
        self.assertTrue(t.result()["tracking_valid_kicked_goal"])

    def test_tracking_failure_does_not_end_fixed_horizon_score(self):
        t = trial()
        t.observe([0,-1,.11], [0,-1.1,.11], [0,-1,0], True, .1)
        t.mark_tracking_failure(.2)
        t.observe([0,-4.9,.11], [0,-5.1,.11], [0,-1,0], False, .2)
        r = t.result()
        self.assertTrue(r["goal"] and r["kicked_goal"])
        self.assertFalse(r["tracking_valid_kicked_goal"])

    def test_goal_before_later_tracking_failure_is_retained(self):
        t = trial()
        t.observe([0,-4.9,.11], [0,-5.1,.11], [0,-1,0], True, .1)
        t.mark_tracking_failure(.2)
        self.assertTrue(t.result()["tracking_valid_kicked_goal"])
        self.assertTrue(t.result()["tracking_failed"])

    def test_accuracy_uses_impact_point_and_qualifying_speed(self):
        t = trial()
        t.observe([0,-1,.11], [0,-1.01,.11], [0,-.2,0], True, .1)
        self.assertIsNone(t.result()["kick_accuracy_cosine"])
        t.observe([0,-1.01,.11], [2,-2,.11], [0,-1,0], False, .2)
        self.assertAlmostEqual(t.result()["kick_accuracy_cosine"], 1.)
        t.observe([2,-2,.11], [2.1,-2,.11], [2,0,0], False, .3)
        self.assertAlmostEqual(t.result()["kick_accuracy_cosine"], 1.)
        self.assertAlmostEqual(t.result()["peak_speed_accuracy_cosine"], 0.)

    def test_numerical_failure_is_not_dropped_from_denominator(self):
        t = trial()
        t.observe([0,-4.9,.11], [0,-5.1,.11], [0,-1,0], True, .1)
        t.numerical_failure = True
        self.assertFalse(t.result()["goal"])
        self.assertIsNone(t.result()["kick_accuracy_cosine"])

    def test_failed_control_transition_is_excluded_as_a_whole(self):
        t = trial()
        t.mark_tracking_failure(.12)
        t.observe([0,-4.9,.11], [0,-5.1,.11], [0,-1,0], True, .101)
        self.assertTrue(t.result()["goal"])
        self.assertFalse(t.result()["tracking_valid_kicked_goal"])

    def test_geometry_is_invariant_under_shared_world_transform(self):
        angle = .73
        rotation = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle),np.cos(angle)]])
        offset = np.array([2.,-3.])
        def point(value):
            return np.r_[rotation @ np.asarray(value)[:2] + offset, value[2]]
        for x, expected in [(.3,True),(.7,False)]:
            self.assertEqual(goal_crossed(point([x,-4,.11]),point([x,-6,.11]),
                                          point([0,-5,.11]),rotation @ [0,-1],1.), expected)

    def test_invalid_inputs_and_time_are_rejected(self):
        with self.assertRaises(ValueError):
            ShotMetrics(np.zeros(3), np.zeros(2), 1., .5)
        t = trial()
        t.observe([0,0,0], [0,0,0], [0,0,0], False, 0.)
        with self.assertRaises(ValueError):
            t.observe([0,0,0], [0,0,0], [0,0,0], False, 0.)
        with self.assertRaises(ValueError):
            t.observe([np.nan,0,0], [0,0,0], [0,0,0], False, .1)


if __name__ == "__main__":
    unittest.main()
