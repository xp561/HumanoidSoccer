"""Check noise geometry, random-stream isolation and actor-only wiring."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/whole_body_tracking/soccer/tasks/tracking/mdp"
spec = importlib.util.spec_from_file_location("ball_noise", MDP / "ball_observation_noise.py")
noise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(noise)


class BallNoiseTests(unittest.TestCase):
    def test_disabled_is_bitwise_identity_and_does_not_advance_rng(self):
        x = torch.randn(16, 3)
        g = torch.Generator().manual_seed(17)
        state = g.get_state().clone()
        self.assertIs(noise.add_ball_xy_noise(x, 0, g), x)
        self.assertTrue(torch.equal(state, g.get_state()))

    def test_invalid_std(self):
        for std in (-1, float("nan"), float("inf")):
            with self.subTest(std=std), self.assertRaises(ValueError):
                noise.add_ball_xy_noise(torch.zeros(2, 3), std)

    def test_invalid_shape_and_dtype(self):
        for x in (torch.zeros(2, 2), torch.tensor(0.), torch.zeros(2, 3, dtype=torch.int64)):
            with self.assertRaises(ValueError):noise.add_ball_xy_noise(x, .005)

    def test_statistics_geometry_and_no_mutation(self):
        x = torch.ones(200000, 3, dtype=torch.float64)
        y = noise.add_ball_xy_noise(x, .005, torch.Generator().manual_seed(23))
        d = y - x
        self.assertTrue(torch.equal(x, torch.ones_like(x)))
        self.assertTrue(torch.equal(y[:, 2], x[:, 2]))
        self.assertEqual(y.dtype, x.dtype)
        self.assertLess(float(d[:, :2].mean(0).abs().max()), .00005)
        self.assertLess(float((d[:, :2].std(0) - .005).abs().max()), .00005)
        self.assertLess(abs(float(torch.corrcoef(d[:, :2].T)[0, 1])), .01)

    def test_seed_reproduction_and_independent_axes(self):
        x = torch.zeros(100, 3)
        a = noise.add_ball_xy_noise(x, .005, torch.Generator().manual_seed(12))
        b = noise.add_ball_xy_noise(x, .005, torch.Generator().manual_seed(12))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a[:, 0], a[:, 1]))

    def test_generator_preserves_global_rng_and_is_reused(self):
        env = NS(cfg=NS(seed=12));x = torch.zeros(10, 3)
        state = torch.get_rng_state().clone()
        g = noise.ball_noise_generator(env, x)
        noise.add_ball_xy_noise(x, .005, g)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIs(noise.ball_noise_generator(env, x), g)

    def test_unset_seed_uses_process_seed_without_consuming_global_rng(self):
        env = NS(cfg=NS(seed=None));x = torch.zeros(10, 3)
        state = torch.get_rng_state().clone()
        with patch("torch.initial_seed", return_value=319):
            generator = noise.ball_noise_generator(env, x)
        expected = torch.Generator().manual_seed(319 + 917431)
        self.assertTrue(torch.equal(generator.get_state(), expected.get_state()))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_configured_training_seed_retains_original_stream(self):
        env = NS(cfg=NS(seed=20261022));x = torch.zeros(10, 3)
        with patch("torch.initial_seed", side_effect=AssertionError("Training must use configured seed")):
            generator = noise.ball_noise_generator(env, x)
        expected = torch.Generator().manual_seed(20261022 + 917431)
        actual = noise.add_ball_xy_noise(x, .005, generator)
        original = noise.add_ball_xy_noise(x, .005, expected)
        self.assertTrue(torch.equal(actual, original))

    def test_actor_only_config_and_no_aliasing(self):
        def constant_target_point_pos():pass
        shared = {"command_name": "motion"}
        actor = NS(func=constant_target_point_pos, params=shared)
        critic = NS(func=constant_target_point_pos, params=shared)
        cfg = NS(observations=NS(policy=NS(target_point_pos=actor), critic=NS(target_point_pos=critic)))
        result = noise.configure_ball_observation_noise(cfg, .005)
        self.assertEqual(actor.params["ball_position_noise_std"], .005)
        self.assertEqual(critic.params, {"command_name": "motion"})
        self.assertIsNot(actor.params, critic.params)
        self.assertTrue(result["actor_only"])

    def test_unsupported_observation_is_rejected(self):
        cfg = NS(observations=NS(policy=NS(target_point_pos=None)))
        with self.assertRaises(ValueError):noise.configure_ball_observation_noise(cfg, .005)

    def test_actual_observation_wrapper_keeps_default_clean(self):
        tree = ast.parse((MDP / "observations.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "constant_target_point_pos")
        position = torch.ones(10, 3)
        namespace = {"torch": torch, "validate_ball_noise_std": noise.validate_ball_noise_std,
                     "add_ball_xy_noise": noise.add_ball_xy_noise, "ball_noise_generator": noise.ball_noise_generator,
                     "get_target_point_base": lambda *_: position}
        exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
        f = namespace["constant_target_point_pos"];env = NS(cfg=NS(seed=4))
        self.assertIs(f(env), position)
        self.assertFalse(hasattr(env, "_ball_position_noise_generator"))
        actor = f(env, ball_position_noise_std=.005)
        critic = f(env)
        self.assertTrue(torch.equal(critic, position))
        self.assertTrue(torch.equal(actor[:, 2], position[:, 2]))
        self.assertFalse(torch.equal(actor[:, :2], position[:, :2]))


if __name__ == "__main__":unittest.main()
