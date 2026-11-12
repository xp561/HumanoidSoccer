"""Direction rewards with an optional goal-distance angular tolerance."""

from __future__ import annotations

import math

import torch


def direction_error_reward(
    squared_angle: torch.Tensor,
    distance_to_goal: torch.Tensor,
    std: float,
    goal_width: float | None = None,
) -> torch.Tensor:
    """Score direction error, optionally deriving tolerance from goal geometry.

    With no goal width, preserve the original direction reward. Otherwise,
    use the apparent half-width of the goal as the angular tolerance, capped
    by the original tolerance. This approximates a goal perpendicular to the
    desired shot direction; it does not score an actual goal-plane crossing.
    """
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Direction reward std must be finite and positive")
    if goal_width is None:
        return torch.exp(-squared_angle / (std**2))
    if not math.isfinite(goal_width) or goal_width <= 0:
        raise ValueError("Goal width must be finite and positive")

    half_width = torch.full_like(distance_to_goal, goal_width / 2)
    angular_tolerance = torch.atan2(half_width, distance_to_goal.clamp_min(0))
    angular_tolerance = angular_tolerance.clamp(
        min=torch.finfo(distance_to_goal.dtype).eps, max=std
    )
    return torch.exp(-squared_angle / angular_tolerance.square())
