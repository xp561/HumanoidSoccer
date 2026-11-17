#!/usr/bin/env bash

# Train the R1 curriculum in two stages.  Override the variables below from
# the command line when running a smaller experiment, for example:
# NUM_ENVS=256 MIMIC_MAX_ITERATIONS=4000 SOCCER_MAX_ITERATIONS=10000 \
#     bash shell/progressive_soccer_train_r1.sh trial

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/r1_flat"

RUN_NAME="${RUN_NAME:-${1:-r1_spur}}"
MOTION_PATH="${MOTION_PATH:-motions/r1-corrected}"
# Stage 1 = the old policy's 2048; Stage 2 = 12288 (old: 16384, which OOMs on the 16 GB 4070)
MIMIC_NUM_ENVS="${MIMIC_NUM_ENVS:-2048}"
SOCCER_NUM_ENVS="${SOCCER_NUM_ENVS:-12288}"
MIMIC_MAX_ITERATIONS="${MIMIC_MAX_ITERATIONS:-4000}"
SOCCER_MAX_ITERATIONS="${SOCCER_MAX_ITERATIONS:-18000}"

cd "${REPO_ROOT}"

python scripts/rsl_rl/train_multi.py --task Tracking-Terrain-R1-Spur-RNN-v0 \
    --motion_path "${MOTION_PATH}" \
    --run_name "${RUN_NAME}" \
    --num_envs "${MIMIC_NUM_ENVS}" \
    --max_iterations "${MIMIC_MAX_ITERATIONS}" \
    --headless

LOAD_RUN="${LOAD_RUN:-$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 | xargs -r basename)}"

if [[ -z "${LOAD_RUN}" ]]; then
    echo "Failed to resolve load_run from ${EXPERIMENT_DIR}" >&2
    exit 1
fi

echo "Resolved load_run=${LOAD_RUN}"
echo "Run name=${RUN_NAME}"

python scripts/rsl_rl/train_multi.py --task Tracking-Flat-R1-SoccerMoving-Spur-RNN-v0 \
    --motion_path "${MOTION_PATH}" \
    --load_run "${LOAD_RUN}" \
    --run_name "${RUN_NAME}" \
    --num_envs "${SOCCER_NUM_ENVS}" \
    --max_iterations "${SOCCER_MAX_ITERATIONS}" \
    --resume True \
    --headless
