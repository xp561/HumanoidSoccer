#!/bin/bash
set -euo pipefail
cd /home/tl/Downloads/r1_ros_integration/bitbots_source
exec /home/tl/Downloads/r1_ros_integration/tools/pixi run --frozen -e default \
 env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=183 \
 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 bash -c 'cd /home/tl/Downloads/HumanoidSoccer && exec python -u -m exp.r1_bitbots_adapter.match_session "$@"' r1-match-arena "$@"
