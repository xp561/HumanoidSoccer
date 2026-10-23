"""Run a minimal headless reset-and-step check for the R1 flat tracking task.

Example:
    python scripts/smoke_test_r1_flat_env.py \
        --motion-file motions/r1-converted/soccer-standard-001_right.npz --headless

This script creates a small environment, resets it, and applies zero actions
for a few steps.  It does not construct an RL runner or write training logs.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test the R1 flat tracking environment.")
parser.add_argument("--task", type=str, default="Tracking-Flat-R1-v0", help="Registered R1 tracking task to test.")
parser.add_argument("--motion-file", type=Path, required=True, help="Converted R1 NPZ motion archive.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=8, help="Number of zero-action environment steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num-envs must be positive")
if args_cli.steps < 1:
    parser.error("--steps must be positive")
if not args_cli.motion_file.is_file():
    parser.error(f"motion file does not exist: {args_cli.motion_file}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import soccer.tasks  # noqa: F401 - imports Gym registrations
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def assert_finite(value: object, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"{label} contains non-finite values")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{label}.{key}")


def run() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_files = [str(args_cli.motion_file.resolve())]
    env_cfg.commands.motion.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        observations, _ = env.reset()
        assert_finite(observations, "reset observations")

        raw_env = env.unwrapped
        action_dim = raw_env.action_manager.total_action_dim
        actions = torch.zeros((raw_env.num_envs, action_dim), dtype=torch.float32, device=raw_env.device)
        for _ in range(args_cli.steps):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            assert_finite(observations, "step observations")
            assert_finite(rewards, "step rewards")
            if not isinstance(terminated, torch.Tensor) or not isinstance(truncated, torch.Tensor):
                raise RuntimeError("environment returned non-tensor termination flags")

        print(f"[PASS] {args_cli.task} reset and stepped successfully", flush=True)
        print(f"[PASS] num_envs={raw_env.num_envs}, action_dim={action_dim}, zero_action_steps={args_cli.steps}", flush=True)
        print(f"[PASS] motion_file={args_cli.motion_file}", flush=True)
    finally:
        env.close()


exit_code = 0
try:
    run()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    # See verify_r1_isaacsim_load.py for why the one-shot diagnostics bypass
    # SimulationApp.close() on this headless host.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
