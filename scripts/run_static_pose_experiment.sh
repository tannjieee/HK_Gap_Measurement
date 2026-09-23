#!/usr/bin/env bash
set -euo pipefail

hk_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${HK_PYTHON:-/home/tan/miniconda3/envs/revo3_ros/bin/python}"
frames="${1:-60}"
if [[ ! "$frames" =~ ^[1-9][0-9]*$ ]]; then
    echo "帧数必须是正整数" >&2
    exit 2
fi

run_dir="$hk_dir/cube_pose_runs/static_repeatability_CH120_0304_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"

bash "$hk_dir/scripts/run_cube_pose.sh" \
    --headless --frames "$frames" \
    --record "$run_dir/frames.jsonl" \
    --output "$run_dir/final_snapshot"

"$python_bin" "$hk_dir/static_pose_experiment.py" \
    "$run_dir/frames.jsonl" --output-dir "$run_dir"

echo "静态重复性实验结果：$run_dir"
