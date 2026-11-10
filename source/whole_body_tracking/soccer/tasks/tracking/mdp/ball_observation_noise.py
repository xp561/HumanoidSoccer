"""Opt-in actor-only ball-position noise in the policy observation frame."""
from __future__ import annotations

import math
import torch


def validate_ball_noise_std(std: float) -> float:
    value = float(std)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Ball observation noise standard deviation must be finite and nonnegative")
    return value


def add_ball_xy_noise(position: torch.Tensor, std: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Perturb policy-frame planar position without changing height or source data.

    A disabled term performs no random draws. Call once when constructing a
    fresh actor observation, then reuse that observation during inference.
    """
    std = validate_ball_noise_std(std)
    if position.ndim < 1 or position.shape[-1] != 3 or not position.is_floating_point():
        raise ValueError("Expected floating-point ball positions ending in xyz")
    if std == 0:
        return position
    output = position.clone()
    noise = torch.randn(position.shape[:-1] + (2,), dtype=position.dtype, device=position.device, generator=generator)
    output[..., :2] += noise * std
    return output


def ball_noise_generator(env, position: torch.Tensor) -> torch.Generator:
    """Keep sensor draws independent of task randomization and PPO sampling."""
    generator = getattr(env, "_ball_position_noise_generator", None)
    if generator is None:
        generator = torch.Generator(device=position.device)
        seed = getattr(env.cfg, "seed", None)
        if seed is None:
            # Reading the process seed does not consume the global RNG stream.
            seed = torch.initial_seed()
        generator.manual_seed((int(seed) + 917431) % (2**63 - 1))
        env._ball_position_noise_generator = generator
    return generator


def configure_ball_observation_noise(env_cfg, std: float) -> dict:
    """Change only the named actor term; keep critic and physical state clean."""
    std = validate_ball_noise_std(std)
    term = getattr(env_cfg.observations.policy, "target_point_pos", None)
    if term is None or getattr(term.func, "__name__", "") != "constant_target_point_pos":
        raise ValueError("Ball noise requires the unencoded constant_target_point_pos policy term")
    term.params = {**term.params, "ball_position_noise_std": std}
    return {"distribution": "normal", "mean_m": 0.0, "std_m": std,
            "axes": ["x", "y"], "frame": "policy_body_frame", "z_unchanged": True,
            "actor_only": True, "sampling": "per_fresh_policy_observation",
            "rng": "separate_seeded_torch_generator", "critic_and_physics_unchanged": True}
