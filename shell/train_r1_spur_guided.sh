#!/usr/bin/env bash
# Resume the spur Stage-2 run (4090 run 2026-11-17_01-42-54_r1_spur_v3, model_12000.pt) with the FreiBots R1
# training-guideline deployment constraints (Tracking-Flat-R1-SoccerMoving-SpurGuided-RNN-v0) for 10000 iterations.
# This produced model_21999.pt (china4070, RTX 4070 16 GB: 8192 envs; 12288 envs ran out of PhysX GPU memory).
#   bash shell/train_r1_spur_guided.sh <run_name> [num_envs=8192] [max_iterations=10000]
# The resumed run must be under logs/rsl_rl/r1_flat/ (load_run below).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH} PYTHONUNBUFFERED=1
python scripts/rsl_rl/train_multi.py --task Tracking-Flat-R1-SoccerMoving-SpurGuided-RNN-v0 \
  --motion_path motions/r1-corrected --load_run 2026-11-17_01-42-54_r1_spur_v3 --checkpoint model_12000.pt \
  --resume True --run_name "${1:-r1_spur_guided}" --num_envs "${2:-8192}" --max_iterations "${3:-10000}" --headless
