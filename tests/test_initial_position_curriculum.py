"""Check schedule boundaries, monotonicity, and invalid configuration."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "source/whole_body_tracking/soccer/tasks/tracking/mdp/initial_position_curriculum.py"
SPEC = importlib.util.spec_from_file_location("initial_position_curriculum", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
radius = MODULE.linear_initial_position_radius


class InitialPositionCurriculumTests(unittest.TestCase):
    def test_starts_at_initial_radius_and_clamps_before_start(self):
        for step in [-10, 0]:
            with self.subTest(step=step):
                self.assertEqual(radius(step, 0.05, 0.5, 100), 0.05)

    def test_progresses_monotonically_then_holds_target(self):
        values = [radius(step, 0.05, 0.5, 100) for step in range(201)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[50], 0.275)
        self.assertEqual(values[100], 0.5)
        self.assertEqual(values[-1], 0.5)

    def test_zero_radius_and_constant_schedule_are_valid(self):
        self.assertEqual(radius(100, 0.0, 0.0, 100), 0.0)
        self.assertEqual(radius(50, 0.25, 0.25, 100), 0.25)

    def test_rejects_invalid_radius_ranges(self):
        for start, end in [(-0.1, 0.5), (0.5, 0.1), (float("nan"), 0.5), (0.0, float("inf"))]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    radius(0, start, end, 100)

    def test_rejects_invalid_progress_configuration(self):
        for steps in [0, -1, 2.5]:
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    radius(0, 0.05, 0.5, steps)
        with self.assertRaises(ValueError):
            radius(float("nan"), 0.05, 0.5, 100)


@unittest.skipIf(torch is None, "Production reset checks require project PyTorch")
class ProductionResetTests(unittest.TestCase):
    def reset_function(self):
        source = ast.parse(HELPER.with_name("commands_multi_motion_soccer.py").read_text())
        function = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.FunctionDef) and node.name == "_resample_command"
        )
        for arg in function.args.args:
            arg.annotation = None
        function.returns = None

        def midpoint(low, high, shape, device):
            return (low + (high - low) / 2) + torch.zeros(shape, device=device)

        def identity_quaternion(x, y, z):
            result = torch.zeros((len(x), 4))
            result[:, 0] = 1
            return result

        namespace = {
            "torch": torch,
            "math": math,
            "sample_uniform": midpoint,
            "quat_from_euler_xyz": identity_quaternion,
            "quat_mul": lambda a, b: b,
            "linear_initial_position_radius": radius,
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(HELPER), "exec"), namespace)
        return namespace[function.name]

    def command(self, enabled=False, evaluation_radius=None):
        cfg = SimpleNamespace(
            train_init_position_curriculum=enabled,
            train_init_radius_start=0.05,
            train_init_radius_end=0.5,
            train_init_ramp_steps=100,
            eval_robot_init_radius=evaluation_radius,
            eval_match_motion_to_ball=False,
            eval_fixed_ball_pos=None,
            sampling_strategy="uniform",
            blind_distance_min_range=(0.0, 0.0),
            blind_distance_max_range=(10.0, 10.0),
            pose_range={"x": (0.05, 0.05), "y": (0.05, 0.05)},
            velocity_range={},
            joint_position_range=(0.0, 0.0),
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0]], [[-1.0, 1.0]]])
            ),
            write_joint_state_to_sim=lambda *args, **kwargs: None,
        )
        robot.write_root_state_to_sim = lambda state, **kwargs: setattr(
            robot, "saved_root", state.clone()
        )
        no_op = lambda *args, **kwargs: None
        return SimpleNamespace(
            cfg=cfg,
            device="cpu",
            num_envs=2,
            _env=SimpleNamespace(common_step_counter=100),
            metrics={},
            _state_prefix="_motion",
            _to_env_id_tensor=lambda ids: torch.tensor(ids, dtype=torch.long),
            _sample_soccer_offset=no_op,
            _uniform_sampling=no_op,
            _compute_soccer_ball_positions=no_op,
            _update_soccer_ball=no_op,
            _update_target_points=no_op,
            _update_destination_points=no_op,
            _update_motion_alignment=no_op,
            blind_distance_min=torch.zeros(2),
            blind_distance_max=torch.zeros(2),
            is_in_blind_zone=torch.zeros(2, dtype=torch.bool),
            last_visible_target_point_base=torch.zeros((2, 3)),
            body_pos_w=torch.tensor([[[0.0, 0.0, 0.82]], [[0.0, 0.0, 0.82]]]),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]),
            body_lin_vel_w=torch.zeros((2, 1, 3)),
            body_ang_vel_w=torch.zeros((2, 1, 3)),
            joint_pos=torch.zeros((2, 1)),
            joint_vel=torch.zeros((2, 1)),
            robot=robot,
        )

    def reset(self, command):
        def fixed_uniform(*shape, device=None, **kwargs):
            return torch.full(shape, 0.25, device=device)

        with patch.object(torch, "rand", side_effect=fixed_uniform):
            self.reset_function()(command, [0, 1])
        return command.robot.saved_root[:, :2]

    def test_disabled_curriculum_preserves_planar_box_offsets(self):
        command = self.command()
        actual = self.reset(command)
        torch.testing.assert_close(actual, torch.full((2, 2), 0.05))
        self.assertNotIn("initial_position_radius", command.metrics)

    def test_older_config_without_curriculum_fields_keeps_legacy_sampling(self):
        command = self.command()
        del command.cfg.train_init_position_curriculum
        actual = self.reset(command)
        torch.testing.assert_close(actual, torch.full((2, 2), 0.05))

    def test_curriculum_replaces_box_offsets_instead_of_adding_to_them(self):
        command = self.command(enabled=True)
        actual = self.reset(command)
        expected = torch.tensor([[0.0, 0.25], [0.0, 0.25]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(
            command.metrics["initial_position_radius"], torch.full((2,), 0.5)
        )

    def test_explicit_evaluation_radius_takes_precedence_over_curriculum(self):
        command = self.command(enabled=True, evaluation_radius=0.0)
        actual = self.reset(command)
        torch.testing.assert_close(actual, torch.full((2, 2), 0.05))
        self.assertNotIn("initial_position_radius", command.metrics)


if __name__ == "__main__":
    unittest.main()
