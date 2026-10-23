#!/bin/bash
set -euo pipefail
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
cd /home/tl/Downloads/r1_ros_integration/bitbots_source
exec /home/tl/Downloads/r1_ros_integration/tools/pixi run --frozen -e default \
 env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=178 \
 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST OPENBLAS_NUM_THREADS=1 \
 bash -c 'cd /home/tl/Downloads/HumanoidSoccer && exec python -u -m exp.r1_bitbots_adapter.live_session "$@"' r1-live "$@"
