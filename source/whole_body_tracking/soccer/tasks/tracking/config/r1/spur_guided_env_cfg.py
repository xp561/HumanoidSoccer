"""Spur R1 soccer task (spur_env_cfg) + deployment constraints from the FreiBots R1 training guidelines.

Robot physics, gains, stand pose, action scale, rewards and observations are the spur task's, so a spur
checkpoint resumes unchanged. Added on top (see soccer/tasks/tracking/mdp/guided.py):
  * PD targets clamped to the joint range + 0-40 ms action delay (per env, resampled on reset)
  * penalty on the unclamped target's excess beyond the joint range
  * penalty per joint touching its hard limit (margin 0.01 rad); termination on it after 2000 iterations
  * termination at absolute pelvis tilt > 0.8 rad
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from soccer.robots.r1_spur import R1_SPUR_ACTION_SCALE
from soccer.tasks.tracking.mdp import guided

from .spur_env_cfg import R1SpurFlatKickMovingEnvCfg

TARGET_EXCESS_WEIGHT = -2.0   # per rad of summed target excess (joint_limit on measured position is -10)
LIMIT_MARGIN = 0.01           # rad inside the hard limit
LIMIT_TOUCH_WEIGHT = -20.0    # per joint touching its hard limit, per step (x dt 0.02 -> -0.4)
# 12000 touches ankle roll in ~35 % and waist roll in ~17 % of episodes (diag_limits.py); terminating from
# iteration 0 would end most episodes, so the hard-limit termination switches on after 2000 iterations
# (24 env steps each) of penalty-only training.
LIMIT_TERM_AFTER_STEPS = 2000 * 24
TILT_LIMIT = 0.8              # rad


def _apply_guidelines(cfg):
    joints = list(R1_SPUR_ACTION_SCALE)
    old = cfg.actions.joint_pos
    cfg.actions.joint_pos = guided.GuidedJointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=old.scale,
        use_default_offset=old.use_default_offset,
        clamp_to_joint_range=True,
        delay_substeps_range=(0, 8),
    )
    cfg.rewards.target_beyond_range = RewTerm(
        func=guided.target_beyond_range, weight=TARGET_EXCESS_WEIGHT, params={"action_name": "joint_pos"}
    )
    cfg.rewards.hard_limit_touch = RewTerm(
        func=guided.hard_limit_touch,
        weight=LIMIT_TOUCH_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints), "margin": LIMIT_MARGIN},
    )
    cfg.terminations.joint_hard_limit = DoneTerm(
        func=guided.joint_near_hard_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=joints),
            "margin": LIMIT_MARGIN,
            "active_after_steps": LIMIT_TERM_AFTER_STEPS,
        },
    )
    cfg.terminations.pelvis_tilt = DoneTerm(
        func=guided.pelvis_tilt, params={"command_name": "motion", "limit_rad": TILT_LIMIT}
    )


@configclass
class R1SpurGuidedFlatKickMovingEnvCfg(R1SpurFlatKickMovingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_guidelines(self)
