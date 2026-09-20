#!/usr/bin/env bash
# Source only: select system ROS/Python even from a Conda terminal.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
set +u
source /opt/ros/jazzy/setup.bash
set -u
# Image/point-cloud messages exceed the default shared-memory transport size.
# Keep this scoped to this shell and its children; respect explicit overrides.
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-LARGE_DATA?max_msg_size=8MB&sockets_size=16MB&non_blocking=true}"
