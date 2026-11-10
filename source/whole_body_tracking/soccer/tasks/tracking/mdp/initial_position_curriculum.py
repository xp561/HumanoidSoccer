"""A bounded schedule for initial planar position perturbations."""

from __future__ import annotations

import math


def linear_initial_position_radius(
    control_step: int,
    start_radius: float,
    end_radius: float,
    ramp_steps: int,
) -> float:
    """Increase the sampling radius with elapsed environment control steps.

    The schedule starts when a fresh environment is constructed. Resuming a
    policy does not restore an earlier environment's curriculum progress.
    """
    if not math.isfinite(control_step):
        raise ValueError("Control step must be finite")
    if not math.isfinite(start_radius) or start_radius < 0:
        raise ValueError("Start radius must be finite and nonnegative")
    if not math.isfinite(end_radius) or end_radius < start_radius:
        raise ValueError("End radius must be finite and at least the start radius")
    if not isinstance(ramp_steps, int) or ramp_steps <= 0:
        raise ValueError("Ramp steps must be a positive integer")

    progress = min(max(control_step, 0) / ramp_steps, 1.0)
    return start_radius + progress * (end_radius - start_radius)
