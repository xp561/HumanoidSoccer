#!/bin/bash
# Native GameState-driven simulation. Without --scenario, waits for external ROS input.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo 'Usage: run_match_flow.sh OUTPUT_DIRECTORY [--scenario SCENARIO] [--seconds N] [--record]' >&2
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
        exec python -u -m exp.r1_bitbots_adapter.validate_match_flow \
            --arena-manifest "$run_output/scene/arena_manifest.json" \
            --output "$run_output" "$@"
    ' r1-match-flow "$run_output" "$@"
