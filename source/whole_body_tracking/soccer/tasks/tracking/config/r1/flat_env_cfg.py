from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from soccer.robots.r1 import R1_ACTION_SCALE, R1_CYLINDER_CFG
from soccer.tasks.tracking.config.r1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from soccer.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from soccer.tasks.tracking.mdp import commands_multi_motion as motion_cmds


@configclass
class R1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = R1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Keep the two head joints at their initial pose.  They are present in
        # the R1 articulation but intentionally excluded from policy actions.
        self.actions.joint_pos.joint_names = list(R1_ACTION_SCALE)
        self.actions.joint_pos.scale = R1_ACTION_SCALE
        self.events.add_joint_default_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(R1_ACTION_SCALE)
        )
        self.commands.motion.anchor_body_name = "pelvis_link"
        self.commands.motion.class_type = motion_cmds.MotionCommand
        self.commands.motion.body_names = [
            "pelvis_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "waist_yaw_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_roll_link",
        ]
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_roll_link$)(?!right_wrist_roll_link$).+$"
            ],
        )
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
        ]


@configclass
class R1FlatWoStateEstimationEnvCfg(R1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class R1FlatLowFreqEnvCfg(R1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
