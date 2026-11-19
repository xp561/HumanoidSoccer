"""Deployment constraints from the FreiBots R1 training guidelines (R1_training_guidelines.pdf) and hsl-harness.

* GuidedJointPositionAction: PD targets clamped to the joint range (what the deploy node does) and a per-env
  action delay of 0-40 ms, resampled at every reset, applied at physics-substep resolution (5 ms).
  The unclamped target excess is kept for the target_beyond_range penalty (harness warning
  "target_beyond_joint_range").
* joint_near_hard_limit: termination when a measured joint comes within `margin` of its hard (URDF) limit.
  PhysX limits are rigid, so touching the stop here is what reads "out of range" on MuJoCo / the robot,
  where the deploy node stops with zero margin (harness "measured_out_of_range").
* pelvis_tilt: termination at absolute pelvis tilt > 0.8 rad (deploy node max_tilt, harness "tilt_over_0.8rad").
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class GuidedJointPositionAction(JointPositionAction):
    cfg: "GuidedJointPositionActionCfg"

    def __init__(self, cfg: "GuidedJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        lim = self._asset.data.joint_pos_limits[:, self._joint_ids].clone()  # hard limits, (N, J, 2)
        self._lo, self._hi = lim[..., 0], lim[..., 1]
        self._decimation = int(env.cfg.decimation)
        self._dmin, self._dmax = cfg.delay_substeps_range
        depth = math.ceil(self._dmax / self._decimation) + 1
        self._hist = torch.zeros(self.num_envs, depth, self.action_dim, device=self.device)
        self._delay = torch.randint(self._dmin, self._dmax + 1, (self.num_envs,), device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._substep = 0
        self._ar = torch.arange(self.num_envs, device=self.device)
        self.target_excess = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        p = self._processed_actions
        self.target_excess = torch.relu(p - self._hi) + torch.relu(self._lo - p)
        if self.cfg.clamp_to_joint_range:
            self._processed_actions = p = torch.clamp(p, self._lo, self._hi)
        self._hist = torch.roll(self._hist, 1, dims=1)
        self._hist[:, 0] = p
        if self._fresh.any():  # first action after a reset: no older target to hold
            self._hist[self._fresh] = p[self._fresh].unsqueeze(1)
            self._fresh[:] = False
        self._substep = 0

    def apply_actions(self):
        # the target issued j policy steps ago is (substep + j * decimation) substeps old;
        # use the newest one that is at least `delay` substeps old
        j = torch.clamp(torch.ceil((self._delay - self._substep) / self._decimation), min=0).long()
        j = torch.clamp(j, max=self._hist.shape[1] - 1)
        self._asset.set_joint_position_target(self._hist[self._ar, j], joint_ids=self._joint_ids)
        self._substep += 1

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        n = self._delay[env_ids].shape[0]
        self._delay[env_ids] = torch.randint(self._dmin, self._dmax + 1, (n,), device=self.device)
        self._fresh[env_ids] = True
        self.target_excess[env_ids] = 0.0


@configclass
class GuidedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = GuidedJointPositionAction
    clamp_to_joint_range: bool = True
    delay_substeps_range: tuple[int, int] = (0, 8)  # x sim.dt (5 ms) -> 0-40 ms


def target_beyond_range(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Sum over joints of how far the unclamped PD target lies outside the joint range [rad]."""
    return env.action_manager.get_term(action_name).target_excess.sum(dim=-1)


def _near_limit(env, asset_cfg: SceneEntityCfg, margin: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lim = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    return (q > lim[..., 1] - margin) | (q < lim[..., 0] + margin)


def joint_near_hard_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, margin: float, active_after_steps: int = 0
) -> torch.Tensor:
    """Termination; off until env.common_step_counter reaches active_after_steps (penalty-only warm-up)."""
    hit = torch.any(_near_limit(env, asset_cfg, margin), dim=-1)
    if env.common_step_counter < active_after_steps:
        return torch.zeros_like(hit)
    return hit


def hard_limit_touch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, margin: float) -> torch.Tensor:
    """Penalty: number of policy joints within `margin` of their hard limit."""
    return _near_limit(env, asset_cfg, margin).float().sum(dim=-1)


def pelvis_tilt(env: ManagerBasedRLEnv, command_name: str, limit_rad: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    g = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    pg = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, g)  # anchor = pelvis_link
    return torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)) > limit_rad

