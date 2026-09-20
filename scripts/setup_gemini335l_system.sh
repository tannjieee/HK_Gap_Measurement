#!/usr/bin/env bash
# Install Orbbec ROS 2 Jazzy with a local packaging fix and its USB access rules.
set -euo pipefail

if (( EUID != 0 )); then
    echo "请在本机终端执行：sudo bash $0" >&2
    exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_CODENAME:-}" != noble ]]; then
    echo "此脚本用于 Ubuntu 24.04 (noble) + ROS 2 Jazzy。" >&2
    exit 1
fi
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
    echo "未找到现有 ROS 2 Jazzy：/opt/ros/jazzy/setup.bash" >&2
    exit 1
fi

# Original packages were verified against APT metadata. The camera package
# has a documented local fix: remove a header already owned by the installed
# ros-jazzy-magic-enum package, and declare that package as a dependency.
# Driver binaries are unchanged; see install-manifest.json for provenance.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_dir="$project_root/.cache/orbbec-packages"
if [[ ! -f "$package_dir/INSTALL_SHA256SUMS" ]]; then
    echo "缺少已校验的驱动安装包：$package_dir" >&2
    exit 1
fi
(cd "$package_dir" && sha256sum -c INSTALL_SHA256SUMS)
if [[ "$(dpkg-query -W -f='${Status}' ros-jazzy-magic-enum 2>/dev/null)" != "install ok installed" ]]; then
    echo "此修复包依赖本机已有的 ros-jazzy-magic-enum，请先检查该包的安装状态。" >&2
    exit 1
fi
packages=()
while read -r checksum filename; do
    packages+=("$package_dir/$filename")
done < "$package_dir/INSTALL_SHA256SUMS"
# dpkg uses exactly the verified local files. APT may fetch an equal-version
# dependency from its configured mirror even when given the local .deb path.
dpkg --install "${packages[@]}"

rules=/opt/ros/jazzy/share/orbbec_camera/udev/99-obsensor-libusb.rules
if [[ ! -f "$rules" ]]; then
    echo "驱动包中缺少官方 udev 规则：$rules" >&2
    exit 1
fi
install -m 0644 "$rules" /etc/udev/rules.d/99-obsensor-libusb.rules
udevadm control --reload-rules
# Restrict the hotplug refresh to Orbbec devices, leaving the Hikrobot camera
# and other USB peripherals alone.
udevadm trigger --action=change --attr-match=idVendor=2bc5
udevadm settle --timeout=10
echo "Orbbec 驱动和 ROS 2 接口已安装，USB 规则已刷新。"
dpkg-query -W ros-jazzy-orbbec-camera ros-jazzy-orbbec-camera-msgs ros-jazzy-orbbec-description
