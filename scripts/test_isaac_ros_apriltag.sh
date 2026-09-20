#!/usr/bin/env bash
set -euo pipefail

# Read-only validation for the native NVIDIA Isaac ROS AprilTag path.
# It starts the component in an isolated ROS domain, waits for cuAprilTag to
# instantiate, then terminates the complete process group.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_PYTHON="/usr/bin/python3"
# Fast DDS reserves domain IDs above 232 because their calculated UDP ports
# overflow; keep the smoke test in a high but valid isolated domain.
ROS_DOMAIN="${ROS_DOMAIN_ID_TEST:-220}"
LOG_DIR="$(mktemp -d /tmp/hk_isaac_ros_test.XXXXXX)"
trap 'rm -rf "$LOG_DIR"' EXIT

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    set -u
else
    echo "未找到 /opt/ros/jazzy/setup.bash" >&2
    exit 2
fi

echo "[1/4] 检查 NVIDIA GPU"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

echo "[2/4] 检查 ROS/动态库"
ros2 pkg prefix isaac_ros_apriltag >/dev/null
ldd /opt/ros/jazzy/lib/libapriltag_node.so | tee "$LOG_DIR/ldd.txt"
if grep -q 'not found' "$LOG_DIR/ldd.txt"; then
    echo "Isaac ROS AprilTag 仍有未满足的动态库依赖" >&2
    exit 1
fi

"$ROS_PYTHON" - <<'PY'
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
print("AprilTagDetectionArray import: OK")
PY

echo "[3/4] 隔离启动 cuAprilTag 组件"
"$ROS_PYTHON" - "$PROJECT_ROOT" "$ROS_DOMAIN" "$LOG_DIR" <<'PY'
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
domain = sys.argv[2]
log_dir = Path(sys.argv[3])
env = os.environ.copy()
env.update(
    {
        "ROS_DOMAIN_ID": domain,
        "ROS_LOG_DIR": str(log_dir),
        "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4",
        "RMW_FASTRTPS_USE_SHM": "0",
        "HIKROBOT_APRILTAG_SIZE_M": "0.04835",
        "HIKROBOT_APRILTAG_FAMILY": "tag36h11",
        "HIKROBOT_APRILTAG_BACKENDS": "CUDA",
    }
)
command = [sys.executable, str(root / "launch" / "isaac_ros_apriltag_bridge.launch.py")]
process = subprocess.Popen(
    command,
    env=env,
    cwd=str(root),
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    start_new_session=True,
)
ready = False
lines = []
deadline = time.monotonic() + 10.0
try:
    while time.monotonic() < deadline:
        line = process.stdout.readline() if process.stdout is not None else ""
        if line:
            lines.append(line.rstrip())
            print(line, end="")
            if "Using cuAprilTag implementation" in line:
                ready = True
                break
            if "Failed to load library" in line:
                break
        elif process.poll() is not None:
            break
        else:
            time.sleep(0.05)
    if not ready:
        print("cuAprilTag 组件未在测试窗口内就绪", file=sys.stderr)
        raise SystemExit(1)
finally:
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=4.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2.0)
PY

echo "[4/4] 通过：CUDA/VPI 动态库、ROS 接口和 cuAprilTag 组件均正常"
