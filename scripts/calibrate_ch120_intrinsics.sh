#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

exec conda run --no-capture-output -n revo3_ros python \
    hikrobot_calibration.py \
    --squares 12x9 \
    --square-size-mm 15 \
    --minimum-samples 12 \
    --target-samples 20 \
    --serial DB1488042 \
    --camera-name CH120-10GM-1
