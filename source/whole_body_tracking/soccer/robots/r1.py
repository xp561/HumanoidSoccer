import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from soccer.assets import ASSET_DIR

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# R1 uses the same actuator classes as the existing G1 configuration.  Keep
# R1-specific aliases so the joint-group assignment below remains explicit.
ARMATURE_R1_HIP14 = ARMATURE_7520_14
ARMATURE_R1_HIP22 = ARMATURE_7520_22
ARMATURE_R1_5020 = ARMATURE_5020
ARMATURE_R1_4010 = ARMATURE_4010
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

STIFFNESS_R1_HIP14 = STIFFNESS_7520_14
STIFFNESS_R1_HIP22 = STIFFNESS_7520_22
STIFFNESS_R1_5020 = STIFFNESS_5020
STIFFNESS_R1_4010 = STIFFNESS_4010

DAMPING_R1_HIP14 = DAMPING_7520_14
DAMPING_R1_HIP22 = DAMPING_7520_22
DAMPING_R1_5020 = DAMPING_5020
DAMPING_R1_4010 = DAMPING_4010

R1_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/unitree_description/urdf/r1/main.urdf",
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
    pos=(0.0, 0.0, 0.78), # 全伸直R1重心更高，z先设0.78，可视化微调
    joint_pos={
        "left_hip_pitch_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.0,
        "left_ankle_pitch_joint": 0.0,
        "left_ankle_roll_joint": 0.0,

        "right_hip_pitch_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.0,
        "right_ankle_pitch_joint": 0.0,
        "right_ankle_roll_joint": 0.0,

        "waist_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,

        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,

        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,

        "head_pitch_joint": 0.0,
        "head_yaw_joint": 0.0,
    },
    joint_vel={".*": 0.0},
),


    soft_joint_pos_limit_factor=0.9,


    actuators={
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
            "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
        ],
        effort_limit_sim={
            "left_hip_yaw_joint": 60.0,
            "left_hip_roll_joint": 60.0,
            "left_hip_pitch_joint": 60.0,
            "left_knee_joint": 60.0,
            "right_hip_yaw_joint": 60.0,
            "right_hip_roll_joint": 60.0,
            "right_hip_pitch_joint": 60.0,
            "right_knee_joint": 60.0,
        },
        velocity_limit_sim={
            "left_hip_yaw_joint": 18.8,
            "left_hip_roll_joint": 18.8,
            "left_hip_pitch_joint": 18.8,
            "left_knee_joint": 18.8,
            "right_hip_yaw_joint": 18.8,
            "right_hip_roll_joint": 18.8,
            "right_hip_pitch_joint": 18.8,
            "right_knee_joint": 18.8,
        },
        stiffness={
            "left_hip_pitch_joint": STIFFNESS_R1_HIP14,
            "left_hip_roll_joint": STIFFNESS_R1_HIP22,
            "left_hip_yaw_joint": STIFFNESS_R1_HIP14,
            "left_knee_joint": STIFFNESS_R1_HIP22,
            "right_hip_pitch_joint": STIFFNESS_R1_HIP14,
            "right_hip_roll_joint": STIFFNESS_R1_HIP22,
            "right_hip_yaw_joint": STIFFNESS_R1_HIP14,
            "right_knee_joint": STIFFNESS_R1_HIP22,
        },
        damping={
            "left_hip_pitch_joint": DAMPING_R1_HIP14,
            "left_hip_roll_joint": DAMPING_R1_HIP22,
            "left_hip_yaw_joint": DAMPING_R1_HIP14,
            "left_knee_joint": DAMPING_R1_HIP22,
            "right_hip_pitch_joint": DAMPING_R1_HIP14,
            "right_hip_roll_joint": DAMPING_R1_HIP22,
            "right_hip_yaw_joint": DAMPING_R1_HIP14,
            "right_knee_joint": DAMPING_R1_HIP22,
        },
        armature={
            "left_hip_pitch_joint": ARMATURE_R1_HIP14,
            "left_hip_roll_joint": ARMATURE_R1_HIP22,
            "left_hip_yaw_joint": ARMATURE_R1_HIP14,
            "left_knee_joint": ARMATURE_R1_HIP22,
            "right_hip_pitch_joint": ARMATURE_R1_HIP14,
            "right_hip_roll_joint": ARMATURE_R1_HIP22,
            "right_hip_yaw_joint": ARMATURE_R1_HIP14,
            "right_knee_joint": ARMATURE_R1_HIP22,
        },
    ),
    "feet": ImplicitActuatorCfg(
        effort_limit_sim=50.0,
        velocity_limit_sim=30.0,
        joint_names_expr=["left_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
        stiffness=2.0 * STIFFNESS_R1_5020,
        damping=2.0 * DAMPING_R1_5020,
        armature=2.0 * ARMATURE_R1_5020,
    ),
    "waist": ImplicitActuatorCfg(
        effort_limit_sim=60,
        velocity_limit_sim=18.8,
        joint_names_expr=["waist_roll_joint"],
        stiffness=2.0 * STIFFNESS_R1_5020,
        damping=2.0 * DAMPING_R1_5020,
        armature=2.0 * ARMATURE_R1_5020,
    ),
    "waist_yaw": ImplicitActuatorCfg(
        effort_limit_sim=60,
        velocity_limit_sim=18.8,
        joint_names_expr=["waist_yaw_joint"],
        stiffness=STIFFNESS_R1_HIP14,
        damping=DAMPING_R1_HIP14,
        armature=ARMATURE_R1_HIP14,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint",
        ],
        effort_limit_sim={
            "left_shoulder_pitch_joint": 60.0,
            "left_shoulder_roll_joint": 60.0,
            "left_shoulder_yaw_joint": 33.0,
            "left_elbow_joint": 33.0,
            "left_wrist_roll_joint": 33.0,
            "right_shoulder_pitch_joint": 60.0,
            "right_shoulder_roll_joint": 60.0,
            "right_shoulder_yaw_joint": 33.0,
            "right_elbow_joint": 33.0,
            "right_wrist_roll_joint": 33.0,
        },
        velocity_limit_sim={
            "left_shoulder_pitch_joint": 18.8,
            "left_shoulder_roll_joint": 18.8,
            "left_shoulder_yaw_joint": 33.4,
            "left_elbow_joint": 33.4,
            "left_wrist_roll_joint": 33.4,
            "right_shoulder_pitch_joint": 18.8,
            "right_shoulder_roll_joint": 18.8,
            "right_shoulder_yaw_joint": 33.4,
            "right_elbow_joint": 33.4,
            "right_wrist_roll_joint": 33.4,
        },
        stiffness={
            "left_shoulder_pitch_joint": STIFFNESS_R1_5020,
            "left_shoulder_roll_joint": STIFFNESS_R1_5020,
            "left_shoulder_yaw_joint": STIFFNESS_R1_5020,
            "left_elbow_joint": STIFFNESS_R1_5020,
            "left_wrist_roll_joint": STIFFNESS_R1_4010,
            "right_shoulder_pitch_joint": STIFFNESS_R1_5020,
            "right_shoulder_roll_joint": STIFFNESS_R1_5020,
            "right_shoulder_yaw_joint": STIFFNESS_R1_5020,
            "right_elbow_joint": STIFFNESS_R1_5020,
            "right_wrist_roll_joint": STIFFNESS_R1_4010,
        },
        damping={
            "left_shoulder_pitch_joint": DAMPING_R1_5020,
            "left_shoulder_roll_joint": DAMPING_R1_5020,
            "left_shoulder_yaw_joint": DAMPING_R1_5020,
            "left_elbow_joint": DAMPING_R1_5020,
            "left_wrist_roll_joint": DAMPING_R1_4010,
            "right_shoulder_pitch_joint": DAMPING_R1_5020,
            "right_shoulder_roll_joint": DAMPING_R1_5020,
            "right_shoulder_yaw_joint": DAMPING_R1_5020,
            "right_elbow_joint": DAMPING_R1_5020,
            "right_wrist_roll_joint": DAMPING_R1_4010,
        },
        armature={
            "left_shoulder_pitch_joint": ARMATURE_R1_5020,
            "left_shoulder_roll_joint": ARMATURE_R1_5020,
            "left_shoulder_yaw_joint": ARMATURE_R1_5020,
            "left_elbow_joint": ARMATURE_R1_5020,
            "left_wrist_roll_joint": ARMATURE_R1_4010,
            "right_shoulder_pitch_joint": ARMATURE_R1_5020,
            "right_shoulder_roll_joint": ARMATURE_R1_5020,
            "right_shoulder_yaw_joint": ARMATURE_R1_5020,
            "right_elbow_joint": ARMATURE_R1_5020,
            "right_wrist_roll_joint": ARMATURE_R1_4010,
        },
    ),
},





)

R1_ACTION_SCALE = {}
for a in R1_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            R1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
# G1_ACTION_SCALE = {}
# for a in G1_CYLINDER_CFG.actuators.values():
#     e = a.effort_limit_sim
#     s = a.stiffness
#     names = a.joint_names_expr
#     if not isinstance(e, dict):
#         e = {n: e for n in names}
#     if not isinstance(s, dict):
#         s = {n: s for n in names}
#     for n in names:
#         if n in e and n in s and s[n]:
#             G1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
