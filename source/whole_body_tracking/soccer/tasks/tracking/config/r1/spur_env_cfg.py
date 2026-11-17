"""R1 soccer tasks with spur robot parameters (see soccer/robots/r1_spur.py).

Only the robot asset, action scale and robot-related domain randomization change. Rewards, observations,
motions and PPO settings are inherited unchanged from the original R1 tasks.
"""

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from soccer.robots.r1_spur import R1_SPUR_ACTION_SCALE, R1_SPUR_CFG
from soccer.robots.spur_actuator import randomize_coulomb_friction

from .soccer_flat_env_cfg import R1FlatKickMovingEnvCfg, R1TerrainMotionEnvCfg


class randomize_rigid_body_material_equal(isaac_mdp.randomize_rigid_body_material):
    """Isaac Lab's material randomization with dynamic friction == static friction (MuJoCo has one coefficient)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.material_buckets[:, 1] = self.material_buckets[:, 0]


def _apply_spur_robot(cfg):
    joints = list(R1_SPUR_ACTION_SCALE)
    cfg.scene.robot = R1_SPUR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.joint_names = joints
    cfg.actions.joint_pos.scale = R1_SPUR_ACTION_SCALE

    # Contact friction 0.6-1.4, static == dynamic (spur real_foot_contact range).
    cfg.events.physics_material.func = randomize_rigid_body_material_equal
    cfg.events.physics_material.params["static_friction_range"] = (0.6, 1.4)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.4)

    # Encoder offset +-0.015 rad (spur sensors.yaml q_bias_std range).
    cfg.events.add_joint_default_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=joints)
    cfg.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.015, 0.015)

    # Torso CoM +-5 cm per axis (spur train DR); body name kept from R1FlatEnvCfg (waist_yaw_link).
    cfg.events.base_com.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}

    # Joint friction x0.7-1.3 (spur HW1 ranges are +-30 %), armature x0.6-1.4 (spur legs 0.005-0.011 around 0.008).
    cfg.events.joint_coulomb_friction = EventTerm(
        func=randomize_coulomb_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": (0.7, 1.3)},
    )
    cfg.events.joint_armature = EventTerm(
        func=isaac_mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=joints),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )


@configclass
class R1SpurTerrainMotionEnvCfg(R1TerrainMotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_spur_robot(self)


@configclass
class R1SpurFlatKickMovingEnvCfg(R1FlatKickMovingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_spur_robot(self)
