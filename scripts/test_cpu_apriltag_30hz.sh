#!/usr/bin/env bash
set -euo pipefail

# Real-camera benchmark for the CPU AprilTag path.  It intentionally forces
# backend=opencv so a working CUDA installation cannot hide a slow CPU path.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REVO3_PYTHON:-/home/tan/miniconda3/envs/revo3_ros/bin/python}"
SERIAL="${HIKROBOT_SERIAL:-}"

if [[ ! -x "$PYTHON" ]]; then
    echo "找不到 Conda Python：$PYTHON（可通过 REVO3_PYTHON 覆盖）" >&2
    exit 2
fi

"$PYTHON" - "$PROJECT_ROOT" "$SERIAL" <<'PY'
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

import hikrobot_camera as sdk
from apriltag_pose_gui import PoseEstimator, load_intrinsics, load_pose_config
from hikrobot_gui import CameraController


root = Path(sys.argv[1]).resolve()
serial_hint = sys.argv[2].strip() or None
config_path = root / "config" / "apriltag_pose.yaml"
config = load_pose_config(config_path)
config.backend = "opencv"
camera_matrix, distortion, source_size = load_intrinsics(config.intrinsics_file)

sdk.require_ok("MV_CC_Initialize", sdk.MvCamera.MV_CC_Initialize())
controller: CameraController | None = None
estimator: PoseEstimator | None = None
try:
    _device_list, devices = sdk.enumerate_devices()
    if not devices:
        raise RuntimeError("没有找到 HIKROBOT 相机")
    selected = devices[0]
    if serial_hint:
        for device in devices:
            if sdk.device_identity(device)[2] == serial_hint:
                selected = device
                break
        else:
            raise RuntimeError(f"没有找到序列号为 {serial_hint!r} 的相机")

    estimator = PoseEstimator(
        camera_matrix,
        distortion,
        config,
        source_size,
    )
    controller = CameraController()
    controller.open_camera(selected)
    controller.start_grabbing()

    # Warm up the SDK and detector before timing the steady-state pipeline.
    warmup = 0
    while warmup < 5:
        packet = controller.read_rgb_frame()
        if packet is None:
            continue
        data, width, height, _frame_number = packet
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        estimator.detect(gray, image_size=(width, height))
        warmup += 1

    count = 0
    detected = 0
    start = time.perf_counter()
    while count < 60:
        packet = controller.read_rgb_frame()
        if packet is None:
            continue
        data, width, height, _frame_number = packet
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        poses = estimator.detect(gray, image_size=(width, height))
        detected += bool(poses)
        count += 1

    elapsed = time.perf_counter() - start
    hz = count / elapsed
    print(f"backend={estimator.backend_name}")
    print(f"quad_decimate={config.quad_decimate}")
    print(f"frames={count} detected_frames={detected}")
    print(f"pipeline_hz={hz:.2f} mean_frame_ms={elapsed * 1000.0 / count:.2f}")
    if hz < 30.0:
        raise RuntimeError(f"CPU AprilTag pipeline 低于 30 Hz：{hz:.2f} Hz")
finally:
    if controller is not None:
        controller.close()
    if estimator is not None:
        estimator.close()
    sdk.MvCamera.MV_CC_Finalize()
PY
