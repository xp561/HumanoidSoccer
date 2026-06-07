# MuJoCo Sim2Sim Rollout

This folder contains standalone MuJoCo sim2sim rollout code for the stage-2 soccer policy.

The rollout samples robot and ball initial states, selects a kicking motion from the released motion set, runs the exported policy in MuJoCo, and writes per-trial outputs under `exp/results/`. Generated outputs are ignored by git.

Install the ONNX runtime before using `--policy` or `--policy-dir`:

```bash
python -m pip install onnxruntime
```

## Run

Run directly with the released ONNX policy:

```bash
python exp/mujoco_soccer_experiment.py \
  --policy ckp/policy_30000.onnx \
  --motion-path motions/soccer-standard \
  --num-trials 100
```

Add `--visualize` or `--render` to open a MuJoCo viewer window:

```bash
python exp/mujoco_soccer_experiment.py \
  --policy ckp/policy_30000.onnx \
  --motion-path motions/soccer-standard \
  --num-trials 1 \
  --visualize
```

Useful options:

```bash
python exp/mujoco_soccer_experiment.py \
  --experiment-name test \
  --policy ckp/policy_30000.onnx \
  --motion-path motions/soccer-standard \
  --num-trials 1000 \
  --birth-radius 0.25 \
  --ball-forward-range 0.5 3.0 \
  --ball-lateral-range -1.5 1.5 \
  --goal 5.0 0.0 \
  --goal-width 2.0 \
  --enable-ball-vel \
  --ball-speed-range 0.1 0.3 \
  --ball-spawn-max-attempts 128
```

To export a custom policy from an Isaac Lab checkpoint, use the existing play script:

```bash
python scripts/rsl_rl/play_multi.py \
  --task Tracking-Flat-G1-SoccerMoving-RNN-v0 \
  --motion_path motions/soccer-standard \
  --load_run <run-name> \
  --checkpoint <checkpoint-file> \
  --export_motion_name all \
  --num_envs 1 \
  --headless
```

The ball spawn ranges and goal coordinates are interpreted in the robot initial local frame by default. Thus the defaults sample the ball in front of the robot with local `x` in `[0.5, 5.0]` and local `y` in `[-2.5, 2.5]`, and put the goal center at `(6, 0)`.

## Code Layout

- `mujoco_soccer_experiment.py`: thin CLI entry point
- `mujoco_soccer/cli.py`: argument parsing and top-level orchestration
- `mujoco_soccer/runner.py`: rollout execution and spawn validation
- `mujoco_soccer/scene.py`: MuJoCo XML generation and simulator wrapper
- `mujoco_soccer/motion.py`: motion loading and selection
- `mujoco_soccer/policy.py`: ONNX policy adapter
