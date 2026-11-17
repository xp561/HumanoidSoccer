"""R1 robot config with spur (github.com/Houman-HM/spur @8f34f3a) parameters, "mixed" choice:

* PD gains, effort limits, action scale: spur TRAIN profile (= what spur's deployed policy uses;
  models/provenance.yaml train_position_actuator_gains, policy_metadata_rounding).
* Joint armature, frictionloss, damping: spur REAL profile, measured on the robot (HW1, Oct 2026;
  sim/r1sim/model.py REAL_JOINT_*_HW1). Ankle armature 0.10 is spur's unidentified nominal.
  Frictionloss -> FrictionImplicitActuator (Coulomb feed-forward), damping folded into the drive Kd.
* Masses: spur REAL profile lumped into main_spur.urdf (Dex3 hands instead of fists, backpack 0.454 kg,
  rear mass 1.6 kg); total 31.997 kg (stock 28.836). Head stays a separate articulated chain.
* Default pose: spur POSE_STAND (measured stand lock = the live policy's default pose); base z 0.745.
* Head joints: held at 0 by a stiff non-policy actuator (the motion npz files carry 26 joints).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from soccer.assets import ASSET_DIR
from soccer.robots.spur_actuator import FrictionImplicitActuatorCfg

SIDES = ("left", "right")


def _both(*names):
    return [f"{s}_{n}_joint" for s in SIDES for n in names]


# --- spur real profile, per joint (sim/r1sim/model.py REAL_JOINT_FRICTIONLOSS_HW1 / ARMATURE_HW1 / DAMPING_HW1)
SPUR_FRICTIONLOSS = {
    "left_hip_pitch_joint": 0.585, "left_hip_roll_joint": 0.373, "left_hip_yaw_joint": 0.720, "left_knee_joint": 0.631,
    "left_ankle_pitch_joint": 2.516, "left_ankle_roll_joint": 1.445,
    "right_hip_pitch_joint": 0.585, "right_hip_roll_joint": 0.373, "right_hip_yaw_joint": 0.489,
    "right_knee_joint": 0.425, "right_ankle_pitch_joint": 2.074, "right_ankle_roll_joint": 1.181,
    "waist_roll_joint": 0.516, "waist_yaw_joint": 0.410,
    **{j: 0.1 for j in _both("shoulder_pitch", "shoulder_roll")},
    **{j: 0.03 for j in _both("shoulder_yaw", "elbow", "wrist_roll")},
}  # fmt: skip
SPUR_ARMATURE = {
    **{j: 0.008 for j in _both("hip_pitch", "hip_roll", "hip_yaw", "knee")},
    "waist_roll_joint": 0.008, "waist_yaw_joint": 0.008,
    **{j: 0.10 for j in _both("ankle_pitch", "ankle_roll")},
    **{j: 0.0025 for j in _both("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")},
}  # fmt: skip
SPUR_JOINT_DAMPING = {j: 0.05 for j in SPUR_ARMATURE} | {"left_elbow_joint": 1.0, "right_elbow_joint": 1.0}

# --- spur train profile gains: group -> (joints, kp, kd, effort [N m], velocity limit [rad/s] (R1.urdf))
SPUR_GROUPS = {
    "legs": (_both("hip_pitch", "hip_roll", "hip_yaw", "knee"), 84.30221295234487, 5.366845561980516, 60.0, 18.8),
    "waist": (["waist_roll_joint", "waist_yaw_joint"], 84.30221295234487, 5.366845561980516, 60.0, 18.8),
    "ankles": (_both("ankle_pitch", "ankle_roll"), 31.38534199546416, 1.9980529276831085, 50.0, 30.0),
    "shoulders": (_both("shoulder_pitch", "shoulder_roll"), 60.0, 2.0, 60.0, 18.8),
    "forearms": (_both("shoulder_yaw", "elbow"), 45.0, 2.0, 33.0, 33.4),
    "wrists": (_both("wrist_roll"), 30.0, 1.5, 33.0, 33.4),
}
FRICTION_VEL = 0.1  # rad/s, tanh smoothing of the Coulomb friction


def _make_actuators():
    acts = {}
    for name, (joints, kp, kd, effort, vel) in SPUR_GROUPS.items():
        acts[name] = FrictionImplicitActuatorCfg(
            joint_names_expr=list(joints),
            stiffness=kp,
            damping={j: kd + SPUR_JOINT_DAMPING[j] for j in joints},
            armature={j: SPUR_ARMATURE[j] for j in joints},
            effort_limit_sim=effort,
            velocity_limit_sim=vel,
            coulomb_friction={j: SPUR_FRICTIONLOSS[j] for j in joints},
            coulomb_friction_vel=FRICTION_VEL,
        )
    acts["head"] = ImplicitActuatorCfg(
        joint_names_expr=["head_pitch_joint", "head_yaw_joint"],
        stiffness=200.0, damping=5.0, armature=0.0025, effort_limit_sim=33.0, velocity_limit_sim=33.4,
    )
    return acts


# spur POSE_STAND (models/provenance.yaml pose_stand)
SPUR_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.3,
    "left_hip_roll_joint": 0.025, "right_hip_roll_joint": -0.025,
    "left_hip_yaw_joint": -0.130, "right_hip_yaw_joint": 0.130,
    ".*_knee_joint": 0.455,
    ".*_ankle_pitch_joint": -0.14,
    "left_ankle_roll_joint": -0.050, "right_ankle_roll_joint": 0.050,
    "waist_.*_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.22,
    "left_shoulder_roll_joint": 0.193, "right_shoulder_roll_joint": -0.194,
    "left_shoulder_yaw_joint": 0.039, "right_shoulder_yaw_joint": -0.033,
    ".*_elbow_joint": 0.76,
    ".*_wrist_roll_joint": 0.0,
    "head_.*_joint": 0.0,
}  # fmt: skip

R1_SPUR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/unitree_description/urdf/r1/main_spur.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.745),
        joint_pos=SPUR_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=_make_actuators(),
)

# action scale = 0.25 * effort / kp (spur: legs 0.178, ankles 0.398)
_scale = {j: 0.25 * effort / kp for joints, kp, _kd, effort, _v in SPUR_GROUPS.values() for j in joints}

from soccer.robots.r1 import R1_ACTION_SCALE as _OLD_SCALE  # noqa: E402

assert set(_OLD_SCALE) == set(_scale), "joint set mismatch vs old R1 config"
R1_SPUR_ACTION_SCALE = {j: _scale[j] for j in _OLD_SCALE}  # same order as the old R1 task
