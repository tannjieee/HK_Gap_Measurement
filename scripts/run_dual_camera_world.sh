#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/gemini335l_ros_env.sh"
unset QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR
exec /usr/bin/python3 "$PROJECT_ROOT/dual_camera_world.py" --rviz "$@"
