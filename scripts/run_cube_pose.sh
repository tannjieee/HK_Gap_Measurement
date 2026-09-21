#!/usr/bin/env bash
set -euo pipefail
hk_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${HK_PYTHON:-/home/tan/miniconda3/envs/revo3_ros/bin/python}"
unset QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR
cd "$hk_dir"
exec "$python_bin" "$hk_dir/cube_pose_app.py" "$@"
