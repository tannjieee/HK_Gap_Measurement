#!/usr/bin/env bash
set -euo pipefail
hk_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${HK_PYTHON:-/home/tan/miniconda3/envs/revo3_ros/bin/python}"
unset QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR
cd "$hk_dir"
exec "$python_bin" "$hk_dir/hikrobot_calibration.py" \
    --serial DB1488042 --camera-name CH120-10GM-1 \
    --squares 12x9 --square-size-mm 15 \
    --minimum-samples 12 --target-samples 20 "$@"
