#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if python3 -c 'import PyQt5' >/dev/null 2>&1; then
    exec python3 hikrobot_gui.py "$@"
fi

if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output -n revo3_ros python hikrobot_gui.py "$@"
fi

echo "未找到 PyQt5。请先运行：python3 -m pip install -r requirements.txt" >&2
exit 1
