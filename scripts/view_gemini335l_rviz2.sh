#!/usr/bin/env bash
# View an already-running Gemini 335L camera; keep the camera driver terminal open.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/scripts/gemini335l_ros_env.sh"
# RViz uses the system Qt libraries; ignore plugin paths set by OpenCV/Conda.
unset QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR
exec rviz2 -d "$project_root/config/gemini335l.rviz" "$@"
