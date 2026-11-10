"""Boundary semantics, fast crossings, goal entry and generated-model contracts."""

import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from .match_arena import build_scene, field_to_world, world_to_field
from .match_geometry import ArenaTracker, FieldGeometry, bitbots_parameters, load_config


class TestFieldGeometry(unittest.TestCase):
    def setUp(self):
        self.g = FieldGeometry()
        self.r = 0.07

    def test_touchline_requires_whole_ball(self):
        for y, expected in [
            (2.99, True),
            (3.02, True),
            (3.06999, True),
            (3.07001, False),
        ]:
            with self.subTest(y=y):
                self.assertEqual(self.g.overlaps_field([0, y, 0.07], self.r), expected)

    def test_goal_line_requires_whole_ball(self):
        self.assertTrue(self.g.overlaps_field([4.56999, 0, 0.07], self.r))
        self.assertFalse(self.g.overlaps_field([4.57001, 0, 0.07], self.r))

    def test_corner_projection_uses_circle_rectangle_distance(self):
        self.assertTrue(self.g.overlaps_field([4.54, 3.04, 4], self.r))
        self.assertFalse(self.g.overlaps_field([4.56, 3.06, 4], self.r))

    def test_airborne_projection_counts_for_out(self):
        self.assertTrue(self.g.overlaps_field([0, 3.02, 10], self.r))
        self.assertFalse(self.g.overlaps_field([0, 3.08, 10], self.r))

    def test_two_goals_and_fast_tunnelling(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                result = self.g.crossing(
                    [sign * 4, 0, 0.07], [sign * 7, 0, 0.07], self.r
                )
                self.assertEqual(result["kind"], "goal")
                self.assertEqual(result["end"], sign)
                self.assertAlmostEqual(abs(result["crossing_position"][0]), 4.57)

    def test_partial_ball_in_opening_is_not_goal(self):
        for y in (0.84, -0.84):
            self.assertEqual(
                self.g.crossing([4, y, 0.07], [5, y, 0.07], self.r)["kind"], "out"
            )
        self.assertEqual(
            self.g.crossing([4, 0.82, 0.07], [5, 0.82, 0.07], self.r)["kind"], "goal"
        )

    def test_crossbar_height_checks_whole_ball(self):
        self.assertEqual(
            self.g.crossing([4, 0, 1.17], [5, 0, 1.17], self.r)["kind"], "goal"
        )
        self.assertEqual(
            self.g.crossing([4, 0, 1.19], [5, 0, 1.19], self.r)["kind"], "out"
        )

    def test_ground_contact_penetration_tolerance_is_bounded(self):
        self.assertEqual(
            self.g.crossing([4, 0, 0.06999], [5, 0, 0.06999], self.r)["kind"], "goal"
        )
        self.assertEqual(
            self.g.crossing([4, 0, 0.065], [5, 0, 0.065], self.r)["kind"], "out"
        )

    def test_crossing_time_not_final_height_or_lateral(self):
        # Full crossing is at t=.57, z=1.14; final z=2 is above the bar.
        result = self.g.crossing([4, 0, 0], [5, 0, 2], self.r)
        self.assertEqual(result["kind"], "goal")
        self.assertAlmostEqual(result["crossing_position"][2], 1.14)
        # Crossing occurs outside the opening even though final sample is central.
        self.assertEqual(
            self.g.crossing([4, 3, 0.07], [5, 0, 0.07], self.r)["kind"], "out"
        )

    def test_no_goal_from_outside_or_side_entry(self):
        self.assertIsNone(self.g.crossing([5, 2, 0.07], [5, 0, 0.07], self.r))
        self.assertIsNone(self.g.crossing([5, 0, 0.07], [4, 0, 0.07], self.r))

    def test_in_goal_volume_is_not_entry(self):
        self.assertTrue(self.g.in_goal_volume([4.8, 0, 0.07], self.r))
        self.assertFalse(self.g.in_goal_volume([5.2, 0, 0.07], self.r))
        self.assertFalse(self.g.in_goal_volume([4.8, 0.86, 0.07], self.r))

    def test_exit_boundary_types(self):
        cases = [
            ([0, 3.1, 0.07], "touchline"),
            ([4.6, 2, 0.07], "goal_line"),
            ([4.6, 3.1, 0.07], "corner"),
        ]
        for after, boundary in cases:
            before = [0, 0, 0.07] if boundary != "corner" else [4.4, 2.9, 0.07]
            with self.subTest(boundary=boundary):
                self.assertEqual(
                    self.g.crossing(before, after, self.r)["boundary"], boundary
                )

    def test_no_event_while_in_play(self):
        self.assertIsNone(self.g.crossing([0, 0, 0.07], [4.53, 0, 0.07], self.r))

    def test_event_latches_until_external_restart(self):
        tracker = ArenaTracker(self.g, self.r, [4, 0, 0.07])
        self.assertEqual(tracker.observe([5, 0, 0.07], 1)["kind"], "goal")
        tracker.observe([4, 0, 0.07], 2)
        tracker.observe([5, 0, 0.07], 3)
        self.assertEqual(len(tracker.events), 1)

    def test_nonfinite_input_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.g.overlaps_field([value, 0, 0.07], self.r)
        for radius in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.g.overlaps_field([0, 0, 0.07], radius)

    def test_frame_roundtrip(self):
        for heading in (0, math.pi / 2, -math.pi / 2, math.pi, 0.37):
            value = world_to_field(field_to_world([3, -2, 0.07], heading), heading)
            for got, expected in zip(value, [3, -2, 0.07]):
                self.assertAlmostEqual(got, expected)

    def test_shared_config_adapts_clear_width(self):
        c = load_config()
        p = bitbots_parameters(c)
        self.assertEqual(p["field.size.x"], 9)
        self.assertAlmostEqual(p["field.goal.width"], 1.9)
        self.assertEqual(p["field.goal.depth"], 0.6)

    def test_builder_preserves_robot_and_contact_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "frozen" / "model.xml"
            source.parent.mkdir()
            source.write_text(
                '<mujoco><compiler meshdir="assets"/><option timestep=".001"/><worldbody><geom name="ground" type="plane" size="10 10 .1" friction="1 .005 .0001"/><body name="pelvis_link"><joint name="root" type="free"/><geom type="sphere" size=".1" mass="1"/></body><body name="soccer_ball"><freejoint/><geom name="soccer_ball_geom" type="sphere" size=".11" mass=".4" friction=".4 .005 .0001"/></body></worldbody><actuator/></mujoco>'
            )
            original = source.read_bytes()
            target = root / "arena" / "model.xml"
            result = build_scene(
                source,
                target,
                load_config(),
                ball_profile="small_candidate",
                heading=0.3,
            )
            tree = ET.parse(target)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(
                tree.find("worldbody/geom[@name='ground']").get("friction"),
                "1 .005 .0001",
            )
            self.assertEqual(
                tree.find("worldbody/body[@name='pelvis_link']/geom").get("mass"), "1"
            )
            self.assertEqual(
                tree.find("worldbody/body[@name='soccer_ball']/geom").get("size"),
                "0.07",
            )
            for name in result["visual_only_geometries"]:
                geom = tree.find(f".//geom[@name='{name}']")
                self.assertEqual(
                    (geom.get("contype"), geom.get("conaffinity")), ("0", "0")
                )
            self.assertEqual(len(tree.findall(".//geom[@type='cylinder']")), 9)

    def test_builder_cannot_overwrite_frozen_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                build_scene(
                    root / "model.xml",
                    root / "copy.xml",
                    load_config(),
                    ball_profile="small_candidate",
                    heading=0,
                )


if __name__ == "__main__":
    unittest.main()
