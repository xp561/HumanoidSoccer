"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a single motion file. When specified, only this motion is played and exported.")
parser.add_argument("--motion_path", type=str, default=None, help="The path to the directory containing motion files for random sampling (no export).")

parser.add_argument("--export_motion_name", type=str, default=None, help="Select one motion for exporter (required when --motion_file is used).")
parser.add_argument("--eval_steps", type=int, default=0, help="Run a bounded soccer evaluation and write one JSON summary; 0 disables evaluation.")
parser.add_argument("--eval_output", type=str, default=None, help="Path for the optional soccer evaluation JSON summary.")
parser.add_argument("--eval_ball_forward", type=float, default=None, help="Fixed ball forward distance for evaluation, in meters.")
parser.add_argument("--eval_ball_lateral", type=float, default=0.0, help="Fixed ball lateral offset for evaluation, in meters.")
parser.add_argument("--eval_ball_speed", type=float, default=0.0, help="Fixed initial ball speed for evaluation, in m/s.")
parser.add_argument("--eval_ball_direction_deg", type=float, default=0.0, help="Initial ball direction: 0 points toward the goal.")
parser.add_argument("--eval_ground_friction", type=float, default=None, help="Override ground friction for evaluation before simulation starts.")
parser.add_argument("--eval_ground_restitution", type=float, default=None, help="Override ground restitution for evaluation before simulation starts.")
parser.add_argument("--eval_ball_obs_noise", type=float, default=0.0, help="Gaussian standard deviation applied only to the observed ball target position.")
parser.add_argument("--eval_initial_yaw_deg", type=float, default=None, help="Fixed initial yaw offset for evaluation, in degrees.")
parser.add_argument("--eval_push_speed", type=float, default=0.0, help="One-time lateral root-velocity perturbation for evaluation, in m/s.")
parser.add_argument("--eval_seed", type=int, default=None, help="Deterministic seed for one evaluation trial.")
parser.add_argument("--eval_robot_init_radius", type=float, default=None, help="Sample the robot initial xy position uniformly in this-radius disk.")
parser.add_argument("--eval_ball_speed_range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Independently sample each environment's ball speed uniformly in [MIN, MAX].")
parser.add_argument("--tracking_eval_steps", type=int, default=0, help="Collect body/joint tracking metrics for this many policy steps; 0 disables it.")
parser.add_argument("--tracking_eval_output", type=str, default=None, help="Write per-environment R1 tracking metrics as JSON after --tracking_eval_steps.")
parser.add_argument("--tracking_trace_output", type=str, default=None, help="Optional .npz path for per-frame tracking diagnostics (single environment only).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import glob
import pathlib
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import soccer.tasks  # noqa: F401
from soccer.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from r1_eval_metrics import KickTrial
from r1_tracking_metrics import evaluate_tracking, tracking_trace

def get_motion_files(motion_path: str) -> list[str]:
    """
    Get a list of motion files.
    
    Args:
        motion_path: File path or directory path.
        
    Returns:
        List of motion file paths.
    """
    if os.path.isfile(motion_path):
        # Single-file input.
        return [motion_path]
    elif os.path.isdir(motion_path):
        # Directory input: collect all .npz files.
        motion_files = glob.glob(os.path.join(motion_path, "*.npz"))
        if not motion_files:
            raise ValueError(f"No .npz files found in directory: {motion_path}")
        motion_files.sort()
        print(f"Found {len(motion_files)} motion files in {motion_path}")
        for file in motion_files:
            print(f"  - {os.path.basename(file)}")
        return motion_files
    else:
        raise ValueError(f"Invalid path: {motion_path}. Must be a file or directory.")

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.eval_steps and args_cli.eval_ball_forward is not None:
        angle = math.radians(args_cli.eval_ball_direction_deg)
        env_cfg.commands.motion.eval_fixed_ball_pos = (
            args_cli.eval_ball_lateral,
            -args_cli.eval_ball_forward,
            0.11,
        )
        env_cfg.commands.motion.eval_fixed_ball_lin_vel = (
            args_cli.eval_ball_speed * math.sin(angle),
            -args_cli.eval_ball_speed * math.cos(angle),
            0.0,
        )
        env_cfg.commands.motion.eval_fixed_target_destination = (0.0, -5.0, 0.11)
    if args_cli.eval_steps and args_cli.eval_initial_yaw_deg is not None:
        yaw = math.radians(args_cli.eval_initial_yaw_deg)
        env_cfg.commands.motion.pose_range = {**env_cfg.commands.motion.pose_range, "yaw": (yaw, yaw)}
    if args_cli.eval_steps and args_cli.eval_robot_init_radius is not None:
        env_cfg.commands.motion.eval_robot_init_radius = args_cli.eval_robot_init_radius
    if args_cli.eval_steps and args_cli.eval_ball_speed_range is not None:
        env_cfg.commands.motion.eval_ball_speed_range = tuple(args_cli.eval_ball_speed_range)
    if args_cli.eval_steps and (args_cli.eval_ground_friction is not None or args_cli.eval_ground_restitution is not None):
        material = env_cfg.scene.terrain.physics_material
        env_cfg.scene.terrain.physics_material = material.replace(
            static_friction=args_cli.eval_ground_friction if args_cli.eval_ground_friction is not None else material.static_friction,
            dynamic_friction=args_cli.eval_ground_friction if args_cli.eval_ground_friction is not None else material.dynamic_friction,
            restitution=args_cli.eval_ground_restitution if args_cli.eval_ground_restitution is not None else material.restitution,
        )
        env_cfg.sim.physics_material = env_cfg.scene.terrain.physics_material

    # Preserve the task's robot-following camera for recorded videos.  The
    # global viewer is still preferable for non-video batch playback.
    if not args_cli.video:
        env_cfg.viewer.origin_type = None
        env_cfg.viewer.asset_name = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    motion_files: list[str] = []

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
            motion_files = [args_cli.motion_file]

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        # Select single-motion or multi-motion mode from CLI args.
        if args_cli.motion_file is not None:
            # Single-motion mode: play and export.
            motion_files = [args_cli.motion_file]
            print(f"[INFO]: Using single motion file: {args_cli.motion_file}")
        elif args_cli.motion_path is not None:
            # Multi-motion mode: random sampling for playback (no export by default).
            motion_files = get_motion_files(args_cli.motion_path)
        else:
            raise ValueError("Either --motion_file or --motion_path must be specified.")
        
        env_cfg.commands.motion.motion_files = motion_files
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.eval_seed is not None:
        env.reset(seed=args_cli.eval_seed)
        torch.manual_seed(args_cli.eval_seed)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_targets: list[tuple[str, str]] = []

    if args_cli.motion_file is not None and args_cli.export_motion_name is not None:
        # A single motion is exported only when requested explicitly.
        export_targets.append((args_cli.motion_file, args_cli.export_motion_name))
    elif args_cli.motion_path is not None and args_cli.export_motion_name is not None:
        # Directory mode: export by matching names from export_motion_name.
        if args_cli.export_motion_name.strip().lower() == "all":
            export_targets = [(mf, os.path.basename(mf)) for mf in motion_files]
        else:
            requested_names = [n.strip() for n in args_cli.export_motion_name.split(",") if n.strip()]
            for name in requested_names:
                match = next(
                    (
                        mf
                        for mf in motion_files
                        if os.path.splitext(os.path.basename(mf))[0] == os.path.splitext(name)[0]
                        or os.path.basename(mf) == name
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(f"Requested export motion '{name}' not found in {args_cli.motion_path}.")
                export_targets.append((match, name))

    if export_targets:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        ckpt = args_cli.checkpoint.split('_')[1].split('.')[0]

        for motion_file, export_name in export_targets:
            export_stem = os.path.splitext(export_name)[0]
            filename = f"policy_{ckpt}_{export_stem}.onnx"
            export_motion_policy_as_onnx(
                env.unwrapped,
                ppo_runner.alg.policy,
                normalizer=ppo_runner.obs_normalizer,
                path=export_model_dir,
                filename=filename,
                motion_name=export_name,
            )
            attach_onnx_metadata(
                env.unwrapped,
                args_cli.wandb_path if args_cli.wandb_path else "none",
                export_model_dir,
                filename=filename,
            )
            print(f"[INFO]: Exported policy for {export_name} to: {os.path.join(export_model_dir, filename)}")
    else:
        print("[INFO]: Skipping policy export (set --export_motion_name to enable export).")
    
    # reset environment
    # breakpoint()
    obs, _ = env.get_observations()
    trials = None
    tracking = None
    previous_ball = None
    goal_center = None
    goal_normal = None
    if args_cli.eval_steps:
        command = env.unwrapped.command_manager.get_term("motion")
        ball = env.unwrapped.scene["soccer_ball"]
        previous_ball = ball.data.root_pos_w.detach().cpu().numpy().copy()
        goal_center = (command.target_destination_pos + env.unwrapped.scene.env_origins).detach().cpu().numpy()
        goal_normal = previous_ball[:, :2] - goal_center[:, :2]
        goal_normal /= np.linalg.norm(goal_normal, axis=1, keepdims=True)
        trials = [KickTrial() for _ in range(env.unwrapped.num_envs)]
    if args_cli.tracking_eval_steps:
        tracking = {"policy_body": [], "reference_body": [], "policy_joint": [], "reference_joint": []}
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actor_obs = obs
            if args_cli.eval_ball_obs_noise:
                actor_obs = obs.clone()
                # Soccer configurations append target_point_pos followed by
                # target_destination_pos_local to the policy observation.
                actor_obs[:, -6:-3] += torch.randn_like(actor_obs[:, -6:-3]) * args_cli.eval_ball_obs_noise
            actions = policy(actor_obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if trials is not None:
            ball = env.unwrapped.scene["soccer_ball"]
            current_ball = ball.data.root_pos_w.detach().cpu().numpy().copy()
            velocity = ball.data.root_lin_vel_w.detach().cpu().numpy().copy()
            for index, trial in enumerate(trials):
                trial.observe(previous_ball[index], current_ball[index], velocity[index], goal_center[index], goal_normal[index], 1.0, 0.5)
            previous_ball = current_ball
            if args_cli.eval_push_speed and timestep == args_cli.eval_steps // 4:
                robot = env.unwrapped.scene["robot"]
                root_velocity = robot.data.root_vel_w.clone()
                root_velocity[:, 0] += args_cli.eval_push_speed
                robot.write_root_velocity_to_sim(root_velocity)
        if tracking is not None:
            command = env.unwrapped.command_manager.get_term("motion")
            tracking["policy_body"].append(command.robot_body_pos_w.detach().cpu().numpy().copy())
            tracking["reference_body"].append(command.body_pos_w.detach().cpu().numpy().copy())
            tracking["policy_joint"].append(command.robot_joint_pos.detach().cpu().numpy().copy())
            tracking["reference_joint"].append(command.joint_pos.detach().cpu().numpy().copy())
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if trials is not None:
            timestep += 1
            if timestep >= args_cli.eval_steps:
                output = args_cli.eval_output or os.path.join(os.path.dirname(resume_path), "evaluation_single.json")
                with open(output, "w", encoding="utf-8") as stream:
                    valid = [trial.kick_accuracy_cosine for trial in trials if np.isfinite(trial.kick_accuracy_cosine)]
                    json.dump({"seed": args_cli.eval_seed, "trials": len(trials), "success_goal_rate": float(np.mean([trial.success for trial in trials])), "first_contact_rate": float(np.mean([trial.contacted for trial in trials])), "kick_accuracy_cosine": float(np.mean(valid)) if valid else None, "kick_accuracy_valid_trials": len(valid), "max_ball_speed": float(np.mean([trial.max_ball_speed for trial in trials]))}, stream, indent=2)
                print(f"[INFO]: Wrote soccer evaluation to: {output}")
                break
        if tracking is not None:
            timestep += 1
            if timestep >= args_cli.tracking_eval_steps:
                output = args_cli.tracking_eval_output or os.path.join(os.path.dirname(resume_path), "tracking_evaluation.json")
                values = {key: np.stack(sequence, axis=0) for key, sequence in tracking.items()}
                payload = {
                    "motion_files": [os.path.basename(path) for path in motion_files],
                    "metrics": [
                        evaluate_tracking(
                            values["policy_body"][:, env_id], values["reference_body"][:, env_id],
                            values["policy_joint"][:, env_id], values["reference_joint"][:, env_id],
                        )
                        for env_id in range(env.unwrapped.num_envs)
                    ],
                }
                with open(output, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2)
                if args_cli.tracking_trace_output is not None:
                    if env.unwrapped.num_envs != 1:
                        raise ValueError("--tracking_trace_output requires --num_envs 1")
                    np.savez(
                        args_cli.tracking_trace_output,
                        **tracking_trace(values["policy_body"][:, 0], values["reference_body"][:, 0]),
                    )
                print(f"[INFO]: Wrote R1 tracking evaluation to: {output}")
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
