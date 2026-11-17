"""Implicit PD actuator plus a Coulomb joint friction torque (MuJoCo-style frictionloss in N m).

PhysX's joint friction is a unitless coefficient, so the spur frictionloss values (N m) cannot be put there.
Instead the PD part stays an implicit PhysX drive (stable at 5 ms) and the friction is applied as a feed-forward
joint effort every physics step:

    tau_friction = -Fs * tanh(qd / v_s)

Fs is per joint and per env (domain randomization scales it at startup), v_s is the smoothing velocity.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class FrictionImplicitActuator(ImplicitActuator):
    cfg: FrictionImplicitActuatorCfg

    def __init__(self, cfg: FrictionImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_nominal = self._parse_joint_parameter(cfg.coulomb_friction, 0.0)
        self.friction_static = self.friction_nominal.clone()
        self.friction_vel = self._parse_joint_parameter(cfg.coulomb_friction_vel, 0.1)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        tau_f = -self.friction_static * torch.tanh(joint_vel / self.friction_vel)
        if control_action.joint_efforts is None:
            control_action.joint_efforts = tau_f
        else:
            control_action.joint_efforts = control_action.joint_efforts + tau_f
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class FrictionImplicitActuatorCfg(ImplicitActuatorCfg):
    class_type: type = FrictionImplicitActuator

    coulomb_friction: dict[str, float] | float = 0.0
    """Coulomb friction torque Fs [N m] (MuJoCo frictionloss)."""

    coulomb_friction_vel: dict[str, float] | float = 0.1
    """Smoothing velocity v_s [rad/s] of tanh(qd / v_s)."""


def randomize_coulomb_friction(env, env_ids: Sequence[int] | torch.Tensor | None, asset_cfg, scale_range):
    """Event: Fs = nominal * U(scale_range), per env and per joint, for every FrictionImplicitActuator."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    lo, hi = scale_range
    for act in asset.actuators.values():
        if isinstance(act, FrictionImplicitActuator):
            nom = act.friction_nominal[env_ids]
            act.friction_static[env_ids] = nom * (lo + (hi - lo) * torch.rand_like(nom))
