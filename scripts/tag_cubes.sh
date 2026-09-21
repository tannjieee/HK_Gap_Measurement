#!/usr/bin/env bash
set -euo pipefail
scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--calibrate" ]]; then
    shift
    exec bash "$scripts_dir/calibrate_cube_camera.sh" "$@"
fi
exec bash "$scripts_dir/run_cube_pose.sh" "$@"
