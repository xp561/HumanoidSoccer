"""Regressions for legacy scoring and goal-distance angular precision."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HELPER = (
    ROOT
    / "source/whole_body_tracking/soccer/tasks/tracking/mdp/precision_alignment.py"
)
SPEC = importlib.util.spec_from_file_location("precision_alignment", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
direction_error_reward = MODULE.direction_error_reward


def test_default_preserves_legacy_values_exactly():
    angles = torch.tensor([0.0, 0.1, 0.4, math.pi])
    distances = torch.tensor([0.0, 1.0, 3.5, 9.0])
    expected = torch.exp(-angles.square() / (0.8**2))
    actual = direction_error_reward(angles.square(), distances, 0.8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_same_error_is_penalized_more_for_a_farther_goal():
    error = torch.full((2,), math.radians(10) ** 2)
    reward = direction_error_reward(error, torch.tensor([1.0, 4.0]), 0.8, 1.0)
    assert reward[0] > reward[1]


def test_wider_goal_relaxes_the_same_shot():
    error = torch.tensor([math.radians(10) ** 2])
    distance = torch.tensor([3.5])
    narrow = direction_error_reward(error, distance, 0.8, 1.0)
    wide = direction_error_reward(error, distance, 0.8, 2.0)
    assert narrow.item() < wide.item()


def test_shot_at_apparent_goal_post_has_unit_normalized_error():
    post_angle = math.atan2(0.5, 3.5)
    reward = direction_error_reward(
        torch.tensor([post_angle**2]), torch.tensor([3.5]), 0.8, 1.0
    )
    assert reward.item() == pytest.approx(math.exp(-1), rel=1e-6)


def test_observed_miss_receives_less_reward_than_legacy_scoring():
    error = torch.tensor([math.radians(10.44) ** 2])
    distance = torch.tensor([3.5])
    assert direction_error_reward(error, distance, 0.8).item() > 0.94
    assert direction_error_reward(error, distance, 0.8, 1.0).item() < 0.25


def test_perfect_shots_remain_finite_at_zero_and_long_distances():
    distance = torch.tensor([0.0, 1.0, 3.5, 1e6])
    reward = direction_error_reward(torch.zeros_like(distance), distance, 0.8, 1.0)
    assert torch.isfinite(reward).all()
    torch.testing.assert_close(reward, torch.ones_like(reward))


@pytest.mark.parametrize("width", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_goal_width_is_rejected(width):
    with pytest.raises(ValueError, match="Goal width"):
        direction_error_reward(torch.tensor([0.1]), torch.tensor([3.5]), 0.8, width)


def actual_reward_function():
    """Exercise the production reward without loading the Isaac application."""
    source = HELPER.with_name("rewards.py").read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "ball_velocity_direction_alignment"
    )
    for argument in function.args.args:
        argument.annotation = None
    function.returns = None
    namespace = {
        "torch": torch,
        "direction_error_reward": direction_error_reward,
        "_get_kick_tracker": lambda command: SimpleNamespace(
            detect=lambda *args: SimpleNamespace(new_contact=torch.tensor([False]))
        ),
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(HELPER), "exec"), namespace)
    return namespace[function.name]


def reward_environment(timer):
    angle = math.radians(10.44)
    command = SimpleNamespace(
        target_destination_pos=torch.tensor([[0.0, -5.0, 0.11]]),
        initial_target_point_pos=torch.tensor([[0.0, -1.5, 0.11]]),
        metrics={},
    )
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_w=torch.tensor(
                [[4.6 * math.sin(angle), -4.6 * math.cos(angle), 0.0]]
            )
        )
    )
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command),
        scene={"soccer_ball": ball},
        num_envs=1,
        device="cpu",
        _motion_dir_align_timer=torch.tensor([timer]),
    )


def test_production_reward_uses_adaptive_scoring_only_when_enabled():
    reward = actual_reward_function()
    legacy = reward(reward_environment(1), "motion", 0.8)
    adaptive = reward(
        reward_environment(1), "motion", 0.8, distance_adaptive=True
    )
    assert legacy.item() > 0.94
    assert adaptive.item() < 0.25


def test_adaptive_scoring_still_requires_post_contact_window():
    reward = actual_reward_function()
    actual = reward(reward_environment(0), "motion", 0.8, distance_adaptive=True)
    assert actual.item() == 0.0
