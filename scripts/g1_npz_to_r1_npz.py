"""Convert G1 motion archives into R1-reference motion archives.

The released motion archives record G1's joint and body state.  R1 has a
different joint set and body layout, so this tool replays the common joints on
R1 and records R1's own kinematic state.  The two R1 head joints are left at
their default pose and are not part of the policy action space.

Run this script with the project's ``soccer`` environment and Isaac Sim.  It
does not advance physics; it only renders the articulated kinematic pose.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Convert G1 NPZ motions to R1 NPZ motions.")
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-file", type=Path, help="One G1 motion archive.")
input_group.add_argument("--input-dir", type=Path, help="Directory of G1 motion archives.")
parser.add_argument("--output-dir", type=Path, required=True, help="Directory for converted R1 archives.")
parser.add_argument("--output-name", type=str, help="Output name when converting a single input file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from soccer.robots.r1 import R1_CYLINDER_CFG


# This is the order written by csv_to_npz.py and pkl_to_npz.py for G1.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# R1 shares these 24 controllable joints.  The R1 head joints remain at their
# zero default state and are still recorded in the 26-dimensional archive.
R1_CONTROLLED_JOINT_NAMES = tuple(
    name
    for name in G1_JOINT_NAMES
    if name not in {"waist_pitch_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"}
)


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot = R1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def collect_inputs() -> list[Path]:
    if args_cli.input_file is not None:
        return [args_cli.input_file]
    return sorted(path for path in args_cli.input_dir.rglob("*.npz") if path.is_file())


def load_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing required fields: {sorted(missing)}")
        motion = {name: archive[name] for name in required}

    if motion["joint_pos"].ndim != 2 or motion["joint_pos"].shape[1] != len(G1_JOINT_NAMES):
        raise ValueError(f"{path} must contain {len(G1_JOINT_NAMES)} G1 joint positions per frame")
    if motion["joint_vel"].shape != motion["joint_pos"].shape:
        raise ValueError(f"{path} has incompatible joint velocity shape")
    if motion["body_pos_w"].ndim != 3 or motion["body_pos_w"].shape[1:] != (30, 3):
        raise ValueError(f"{path} must contain the 30-body G1 state written by the existing exporter")
    return motion


def convert_motion(
    sim: SimulationContext,
    scene: InteractiveScene,
    path: Path,
    output_path: Path,
    source_joint_indices: torch.Tensor,
    target_joint_indices: torch.Tensor,
) -> None:
    motion = load_motion(path)
    robot: Articulation = scene["robot"]
    frame_count = motion["joint_pos"].shape[0]
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if fps <= 0:
        raise ValueError(f"{path} has a non-positive frame rate")

    source_joint_pos = torch.as_tensor(motion["joint_pos"], device=sim.device, dtype=torch.float32)
    source_joint_vel = torch.as_tensor(motion["joint_vel"], device=sim.device, dtype=torch.float32)
    source_root_pos = torch.as_tensor(motion["body_pos_w"][:, 0], device=sim.device, dtype=torch.float32)
    source_root_quat = torch.as_tensor(motion["body_quat_w"][:, 0], device=sim.device, dtype=torch.float32)
    source_root_lin_vel = torch.as_tensor(motion["body_lin_vel_w"][:, 0], device=sim.device, dtype=torch.float32)
    source_root_ang_vel = torch.as_tensor(motion["body_ang_vel_w"][:, 0], device=sim.device, dtype=torch.float32)

    log = {
        "fps": np.asarray([fps]),
        # Preserve the simulator's ordering with the archive, rather than
        # requiring downstream checks to infer it from the URDF.
        "joint_names": np.asarray(robot.joint_names),
        "body_names": np.asarray(robot.body_names),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    frame_dt = 1.0 / fps

    for frame_idx in range(frame_count):
        frame_start = time.perf_counter()
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = source_root_pos[frame_idx]
        root_state[:, 3:7] = source_root_quat[frame_idx]
        root_state[:, 7:10] = source_root_lin_vel[frame_idx]
        root_state[:, 10:13] = source_root_ang_vel[frame_idx]
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, target_joint_indices] = source_joint_pos[frame_idx, source_joint_indices]
        joint_vel[:, target_joint_indices] = source_joint_vel[frame_idx, source_joint_indices]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        elapsed = time.perf_counter() - frame_start
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **{name: np.stack(values) if isinstance(values, list) else values for name, values in log.items()})
    print(f"[INFO] Converted {path} -> {output_path}")


def main() -> None:
    input_paths = collect_inputs()
    if not input_paths:
        raise SystemExit("No NPZ motions found.")

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot: Articulation = scene["robot"]

    source_index_by_name = {name: index for index, name in enumerate(G1_JOINT_NAMES)}
    source_joint_indices = torch.tensor(
        [source_index_by_name[name] for name in R1_CONTROLLED_JOINT_NAMES], dtype=torch.long, device=sim.device
    )
    target_joint_indices = torch.tensor(
        robot.find_joints(list(R1_CONTROLLED_JOINT_NAMES), preserve_order=True)[0], dtype=torch.long, device=sim.device
    )
    if target_joint_indices.numel() != len(R1_CONTROLLED_JOINT_NAMES):
        raise RuntimeError("The R1 articulation does not contain every configured controlled joint")

    single_input = args_cli.input_file is not None
    for input_path in input_paths:
        output_name = args_cli.output_name if single_input and args_cli.output_name else input_path.stem
        convert_motion(
            sim,
            scene,
            input_path,
            args_cli.output_dir / f"{output_name}.npz",
            source_joint_indices,
            target_joint_indices,
        )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        # Isaac Sim 4.5 can stall in SimulationApp.close() on this headless
        # host after conversion has completed.  This is a one-shot utility, so
        # flush its result and let process exit reclaim the simulator resources.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
