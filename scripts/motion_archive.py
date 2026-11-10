"""Validate and map released motion archives without starting a simulator."""

from pathlib import Path

import numpy as np

# Archives without metadata use the articulation order written by the G1 exporter.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
R1_CONTROLLED_JOINT_NAMES = tuple(
    name for name in G1_JOINT_NAMES
    if name not in {"waist_pitch_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                    "right_wrist_pitch_joint", "right_wrist_yaw_joint"}
)


def load_g1_motion(path: Path) -> dict[str, np.ndarray]:
    required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w", "kick_leg"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing required fields: {sorted(missing)}")
        motion = {name: archive[name] for name in required}
        motion["joint_names"] = (
            archive["joint_names"] if "joint_names" in archive.files else np.asarray(G1_JOINT_NAMES)
        )
    names = motion["joint_names"].tolist()
    if len(names) != len(set(names)) or set(names) != set(G1_JOINT_NAMES):
        raise ValueError(f"{path} has invalid G1 joint-name metadata")
    positions = motion["joint_pos"]
    if positions.ndim != 2 or positions.shape[1] != len(names) or not len(positions):
        raise ValueError(f"{path} has incompatible joint position shape")
    if motion["joint_vel"].shape != positions.shape:
        raise ValueError(f"{path} has incompatible joint velocity shape")
    for name in required - {"fps", "kick_leg"}:
        if len(motion[name]) != len(positions) or not np.isfinite(motion[name]).all():
            raise ValueError(f"{path} has invalid {name} frames")
    fps = np.asarray(motion["fps"]).reshape(-1)
    if fps.size != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
        raise ValueError(f"{path} has an invalid frame rate")
    label = str(motion["kick_leg"].item()).strip().lower()
    if label not in {"left", "right"}:
        raise ValueError(f"{path} has an invalid kick_leg label")
    motion["kick_leg"] = np.asarray(label)
    if motion["body_pos_w"].ndim != 3 or motion["body_pos_w"].shape[1:] != (30, 3):
        raise ValueError(f"{path} must contain the G1 exporter body state")
    return motion


def source_joint_indices(motion: dict[str, np.ndarray], target_names: tuple[str, ...]) -> list[int]:
    names = motion["joint_names"].tolist()
    return [names.index(name) for name in target_names]


def validate_joint_limits(positions: np.ndarray, limits: np.ndarray, names: tuple[str, ...]) -> None:
    """Reject infeasible reference poses instead of silently clipping a trajectory."""
    invalid = (positions < limits[:, 0] - 1e-6) | (positions > limits[:, 1] + 1e-6)
    if np.any(invalid):
        bad_names = [name for index, name in enumerate(names) if invalid[:, index].any()]
        raise ValueError(f"Reference poses exceed R1 joint limits: {bad_names}")
