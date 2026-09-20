#!/usr/bin/env bash
set -euo pipefail

# Install the system-side NVIDIA Isaac ROS AprilTag component used by
# nvidia_apriltag_backend.py.  This is deliberately separate from Conda:
# Isaac ROS is a ROS 2 binary package and needs to be visible to /opt/ros/jazzy.

ISAAC_ROS_RELEASE="${ISAAC_ROS_RELEASE:-4.0}"
ISAAC_ROS_MIRROR="${ISAAC_ROS_MIRROR:-}"
KEYRING="/usr/share/keyrings/nvidia-isaac-ros.gpg"
SOURCE_FILE="/etc/apt/sources.list.d/nvidia-isaac-ros.list"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
VPI_KEY_URL="https://repo.download.nvidia.com/jetson/jetson-ota-public.asc"
VPI_KEYRING="/usr/share/keyrings/nvidia-vpi.gpg"
VPI_SOURCE_FILE="/etc/apt/sources.list.d/nvidia-vpi.list"

has_cuda_runtime() {
    ldconfig -p 2>/dev/null | grep -q 'libcudart.so.13' || \
        find /usr/local/cuda* /usr/lib/x86_64-linux-gnu \
            -type f -name 'libcudart.so.13*' -print -quit 2>/dev/null | grep -q .
}

has_vpi_runtime() {
    ldconfig -p 2>/dev/null | grep -q 'libnvvpi.so.4' || \
        find /opt/nvidia/vpi4 /usr/lib /usr/local \
            -type f -name 'libnvvpi.so.4*' -print -quit 2>/dev/null | grep -q .
}

if [[ "${1:-}" == "--check" ]]; then
    echo "== NVIDIA / ROS 检查 =="
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    else
        echo "nvidia-smi: 未找到"
    fi
    if has_cuda_runtime; then
        echo "libcudart.so.13: OK"
    else
        echo "libcudart.so.13: 未找到"
    fi
    if has_vpi_runtime; then
        echo "libnvvpi.so.4: OK"
    else
        echo "libnvvpi.so.4: 未找到"
    fi
    if [[ -f /opt/ros/jazzy/setup.bash ]]; then
        # shellcheck disable=SC1091
        set +u
        source /opt/ros/jazzy/setup.bash
        set -u
    fi
    if command -v ros2 >/dev/null 2>&1; then
        ros2 pkg prefix isaac_ros_apriltag 2>/dev/null || echo "isaac_ros_apriltag: 未安装"
    else
        echo "ros2: 未找到"
    fi
    exit 0
fi

if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    echo "需要 sudo 才能安装系统级 ROS/Isaac ROS 包。" >&2
    exit 2
fi

if [[ ! -r /etc/os-release ]]; then
    echo "无法识别 Ubuntu 发行版。" >&2
    exit 2
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_CODENAME:-}" != "noble" ]]; then
    echo "当前脚本针对 Ubuntu 24.04 (noble) + ROS Jazzy；检测到 ${ID:-unknown} ${VERSION_CODENAME:-unknown}。" >&2
    exit 2
fi

if [[ -z "$ISAAC_ROS_MIRROR" ]]; then
    # The .cn endpoint is normally faster from mainland China.  Override with
    # ISAAC_ROS_MIRROR=https://isaac.download.nvidia.com when needed.
    ISAAC_ROS_MIRROR="https://isaac.download.nvidia.cn"
fi

tmp_key="$(mktemp)"
tmp_cuda_keyring="$(mktemp --suffix=.deb)"
trap 'rm -f "$tmp_key" "$tmp_key.gpg" "$tmp_cuda_keyring"' EXIT
echo "使用 Isaac ROS release ${ISAAC_ROS_RELEASE}，镜像 ${ISAAC_ROS_MIRROR}"

if ! has_cuda_runtime; then
    echo "未找到 libcudart.so.13，安装 CUDA 13 runtime"
    if ! dpkg-query -W -f='${Status}' cuda-keyring 2>/dev/null | grep -q 'install ok installed'; then
        curl --fail --silent --show-error --location \
            "$CUDA_KEYRING_URL" -o "$tmp_cuda_keyring"
        ${SUDO} dpkg --install "$tmp_cuda_keyring"
    fi
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y cuda-runtime-13-0
else
    echo "已找到 libcudart.so.13，跳过 CUDA runtime 安装"
fi

if ! has_vpi_runtime; then
    echo "未找到 libnvvpi.so.4，安装 NVIDIA VPI 4 runtime"
    vpi_key_tmp="$(mktemp)"
    trap 'rm -f "$tmp_key" "$tmp_key.gpg" "$tmp_cuda_keyring" "$vpi_key_tmp"' EXIT
    curl --fail --silent --show-error --location "$VPI_KEY_URL" -o "$vpi_key_tmp"
    ${SUDO} install -d -m 0755 "$(dirname "$VPI_KEYRING")"
    gpg --batch --yes --dearmor -o "$vpi_key_tmp.gpg" "$vpi_key_tmp"
    ${SUDO} install -m 0644 "$vpi_key_tmp.gpg" "$VPI_KEYRING"
    printf 'deb [arch=%s signed-by=%s] https://repo.download.nvidia.com/jetson/x86_64/noble r39.2 main\n' \
        "$(dpkg --print-architecture)" "$VPI_KEYRING" | \
        ${SUDO} tee "$VPI_SOURCE_FILE" >/dev/null
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y libnvvpi4
else
    echo "已找到 libnvvpi.so.4，跳过 VPI runtime 安装"
fi

curl --fail --silent --show-error --location \
    "${ISAAC_ROS_MIRROR}/isaac-ros/repos.key" -o "$tmp_key"

${SUDO} install -d -m 0755 "$(dirname "$KEYRING")"
gpg --batch --yes --dearmor -o "$tmp_key.gpg" "$tmp_key"
${SUDO} install -m 0644 "$tmp_key.gpg" "$KEYRING"

arch="$(dpkg --print-architecture)"
printf 'deb [arch=%s signed-by=%s] %s/isaac-ros/release-%s %s main\n' \
    "$arch" "$KEYRING" "$ISAAC_ROS_MIRROR" "$ISAAC_ROS_RELEASE" \
    "$VERSION_CODENAME" | ${SUDO} tee "$SOURCE_FILE" >/dev/null

${SUDO} apt-get update
${SUDO} apt-get install -y ros-jazzy-isaac-ros-apriltag

echo
echo "安装完成。新终端或当前终端执行："
echo "  source /opt/ros/jazzy/setup.bash"
echo "  ros2 pkg prefix isaac_ros_apriltag"
echo "然后在 Conda 环境 revo3_ros 中启动 GUI；配置 backend=auto 会优先使用 CUDA。"
