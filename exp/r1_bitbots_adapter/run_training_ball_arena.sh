#!/bin/bash
# Development baseline: original frozen ball (radius 0.11 m, mass 0.4 kg),
# validated 5.2 s reference-end handoff, native navigation and measured recovery.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo 'Usage: run_training_ball_arena.sh OUTPUT_DIRECTORY [autonomy arguments]' >&2
    exit 2
fi
run_output=$1
shift
cd /home/tl/Downloads/r1_ros_integration/bitbots_source
exec /home/tl/Downloads/r1_ros_integration/tools/pixi run --frozen -e default \
    env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=183 \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    bash -c '
        set -euo pipefail
        cd /home/tl/Downloads/HumanoidSoccer
        run_output=$1
        shift
        if [[ -e "$run_output/result.json" || -e "$run_output/scene/model.xml" ]]; then
            echo "Choose a new output directory; existing evidence is preserved." >&2
            exit 2
        fi
        python -u -m exp.r1_bitbots_adapter.match_arena \
            --ball-profile frozen_reference --output "$run_output/scene"
        exec python -u -m exp.r1_bitbots_adapter.capture_timed_kick \
            --arena-manifest "$run_output/scene/arena_manifest.json" \
            --output "$run_output" --duration 5.2 "$@"
    ' r1-training-ball "$run_output" "$@"
