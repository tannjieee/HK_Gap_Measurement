#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/scripts/gemini335l_ros_env.sh"
if ! ros2 pkg prefix orbbec_camera >/dev/null 2>&1; then
    echo "尚未安装 Orbbec 驱动，请执行：sudo bash $project_root/scripts/setup_gemini335l_system.sh" >&2
    exit 1
fi
exec ros2 launch "$project_root/launch/gemini335l.launch.py" "$@"
