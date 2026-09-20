#!/usr/bin/env python3
"""Optional NVIDIA Isaac ROS CUDA AprilTag adapter.

The HIKROBOT GUI owns the camera and publishes each frame to the official
``isaac_ros_apriltag`` ROS 2 component.  Isaac ROS performs the tag decode on
CUDA and returns corners plus a metric pose.  This module deliberately keeps
all ROS imports lazy: a normal ``revo3_ros`` installation can continue to use
the tested OpenCV backend when Isaac ROS is not installed.

The NVIDIA node expects a rectified RGB/BGR image for its CUDA backend.  The
adapter rectifies the frame before publishing and projects the returned pose
back onto the original (distorted) image for GUI overlays.
"""

from __future__ import annotations

import ctypes.util
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2  # type: ignore
import numpy as np

from apriltag_coordinates import remap_tag_rotation


class NvidiaAprilTagUnavailable(RuntimeError):
    """The Isaac ROS CUDA detector is not available in this environment."""


class NvidiaAprilTagRuntimeError(RuntimeError):
    """The Isaac ROS node stopped or did not answer a frame request."""


def _cuda_library_paths() -> list[str]:
    """Return likely CUDA runtime directories without requiring ldconfig."""

    roots: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    roots.extend(
        Path(path)
        for path in (
            "/usr/local/cuda-13.0",
            "/usr/local/cuda",
            "/usr/local/cuda-12.9",
        )
    )
    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib")
    for root in roots:
        candidates.extend(
            (root / "lib64", root / "targets" / "x86_64-linux" / "lib")
        )
    candidates.extend(
        (
            Path("/opt/nvidia/vpi4/lib/x86_64-linux-gnu"),
            Path("/opt/nvidia/vpi4/lib"),
        )
    )
    return [str(path) for path in candidates if path.is_dir()]


def _with_cuda_library_path(environment: Mapping[str, str]) -> dict[str, str]:
    """Add CUDA/Conda library directories to a child ROS process."""

    result = dict(environment)
    paths = _cuda_library_paths()
    current = result.get("LD_LIBRARY_PATH", "")
    result["LD_LIBRARY_PATH"] = ":".join(
        list(dict.fromkeys(paths + ([current] if current else [])))
    )
    return result


def _cuda_runtime_available() -> bool:
    """Check for CUDA 13 and VPI 4 before spawning a ROS component."""

    library = ctypes.util.find_library("cudart")
    cuda_found = bool(library and "libcudart.so.13" in library)
    vpi_found = False
    for directory in _cuda_library_paths():
        try:
            if any(Path(directory).glob("libcudart.so.13*")):
                cuda_found = True
            if any(Path(directory).glob("libnvvpi.so.4*")):
                vpi_found = True
        except OSError:
            continue
        if cuda_found and vpi_found:
            return True
    # ldconfig may know VPI even when its directory is not one of the
    # conventional CUDA paths above.
    vpi_library = ctypes.util.find_library("nvvpi")
    return cuda_found and (
        vpi_found or bool(vpi_library and "libnvvpi.so.4" in vpi_library)
    )


def _require_cuda_runtime() -> None:
    if not _cuda_runtime_available():
        raise NvidiaAprilTagUnavailable(
            "未找到 CUDA 13/VPI 4 runtime（需要 libcudart.so.13 和 libnvvpi.so.4）。"
            "请执行 scripts/setup_isaac_ros_apriltag.sh 后再启用 NVIDIA 后端。"
        )


@dataclass
class NvidiaPoseRecord:
    """Detector result with the attributes consumed by the GUI adapter."""

    tag_id: int
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray
    euler_xyz_deg: np.ndarray
    translation_mm: np.ndarray
    reprojection_error_px: float
    decision_margin: float = float("nan")
    family: str = "tag36h11"


def _rotation_to_euler_xyz(rotation: np.ndarray) -> np.ndarray:
    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if sy >= 1e-9:
        x = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        y = math.atan2(-float(rotation[2, 0]), sy)
        z = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        x = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        y = math.atan2(-float(rotation[2, 0]), sy)
        z = 0.0
    return np.degrees([x, y, z]).astype(np.float64)


def _quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("NVIDIA AprilTag 返回了零长度四元数")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _stamp_key(stamp: Any) -> tuple[int, int]:
    return int(getattr(stamp, "sec", 0)), int(getattr(stamp, "nanosec", 0))


def _nested(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _square_object_points(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) / 2.0
    # cuAprilTags publishes corners in the order corresponding to
    # ``(-X,-Y), (+X,-Y), (+X,+Y), (-X,+Y)`` in its tag frame.  Keep this
    # order for the reprojection check; using the OpenCV/IPPE order here
    # rotates the square by 180 degrees and incorrectly rejects good poses.
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )


class NvidiaAprilTagEstimator:
    """Bridge live frames to NVIDIA Isaac ROS's CUDA AprilTag component.

    The official component is loaded separately by a ROS launch process.  By
    default the bridge starts the project-local launch file automatically when
    ``auto_start`` is true; set it false when a system-level Isaac ROS graph is
    already running on the same ``image``/``camera_info`` topics.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        config: Any,
        source_size: tuple[int, int] | None = None,
        *,
        auto_start: bool = True,
        timeout_s: float = 0.25,
        max_tags: int = 64,
        tile_size: int = 4,
        launch_file: str | Path | None = None,
        image_topic: str = "image",
        camera_info_topic: str = "camera_info",
        detection_topic: str = "tag_detections",
    ) -> None:
        family = str(getattr(config, "family", "tag36h11")).strip().lower()
        if family != "tag36h11":
            raise NvidiaAprilTagUnavailable(
                "NVIDIA cuAprilTags CUDA 后端目前只支持 tag36h11；"
                f"当前配置为 {family!r}，请切换 tag36h11 或使用 CPU 后端"
            )

        try:
            import rclpy  # type: ignore
            from rclpy.executors import SingleThreadedExecutor  # type: ignore
            from rclpy.qos import QoSProfile  # type: ignore
            from sensor_msgs.msg import CameraInfo, Image  # type: ignore
            from isaac_ros_apriltag_interfaces.msg import (  # type: ignore
                AprilTagDetectionArray,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise NvidiaAprilTagUnavailable(
                "未找到 NVIDIA Isaac ROS AprilTag 接口。请在 ROS Jazzy/Isaac ROS 环境中安装 "
                "ros-jazzy-isaac-ros-apriltag，并 source 对应 workspace；"
                "当前程序会自动回退到 OpenCV CPU 后端。"
            ) from exc

        self._rclpy = rclpy
        self._camera_info_type = CameraInfo
        self._image_type = Image
        self._family = family
        self._config = config
        self._camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self._distortion = np.asarray(distortion, dtype=np.float64).reshape(1, -1)
        self._source_size = source_size
        self._timeout_s = max(0.03, float(timeout_s))
        self._max_tags = max(1, int(max_tags))
        self._tile_size = max(1, int(tile_size))
        self._image_topic = str(image_topic)
        self._camera_info_topic = str(camera_info_topic)
        self._detection_topic = str(detection_topic)
        self._condition = threading.Condition()
        self._messages: dict[tuple[int, int], Any] = {}
        self._latest_message: Any | None = None
        self._latest_key: tuple[int, int] | None = None
        self._closed = False
        self._last_error = ""
        self._maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._process: subprocess.Popen[Any] | None = None
        self._owns_context = False

        if auto_start:
            # Fail before rclpy.init() so an ``auto`` configuration falls back
            # cleanly instead of creating a ROS log/node when CUDA is absent.
            _require_cuda_runtime()

        try:
            # Fast DDS shared-memory ports can be left locked by a component
            # container that was interrupted while the GUI was closing.  The
            # bridge is local-only and UDPv4 is reliable here, while avoiding
            # the stale ``fastrtps_portXXXX`` lock failure on the next start.
            os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
            os.environ.setdefault("RMW_FASTRTPS_USE_SHM", "0")
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_context = True
            # A settings reload can briefly overlap the old estimator while
            # the new CUDA component is loading.  Give each bridge instance
            # its own ROS name so rosout does not report a duplicate publisher
            # and the old node can finish its shutdown cleanly.
            instance_token = f"{os.getpid()}_{id(self) % 1000000}"
            self._node = rclpy.create_node(
                f"hikrobot_apriltag_cuda_bridge_{instance_token}"
            )
            qos = QoSProfile(depth=10)
            self._image_pub = self._node.create_publisher(
                Image, self._image_topic, qos
            )
            self._camera_info_pub = self._node.create_publisher(
                CameraInfo, self._camera_info_topic, qos
            )
            self._subscription = self._node.create_subscription(
                AprilTagDetectionArray,
                self._detection_topic,
                self._detection_callback,
                qos,
            )
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._spin, name="isaac-ros-apriltag-spin", daemon=True
            )
            self._spin_thread.start()
        except Exception:
            self.close()
            raise

        if auto_start:
            try:
                self._start_ros_component(launch_file)
            except Exception:
                self.close()
                raise

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def process_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as exc:  # pragma: no cover - depends on ROS shutdown
            if not self._closed:
                self._last_error = str(exc)

    def _start_ros_component(self, launch_file: str | Path | None) -> None:
        _require_cuda_runtime()
        if launch_file:
            launch_path = Path(launch_file).expanduser().resolve()
        else:
            launch_path = Path(__file__).resolve().parent / "launch" / (
                "isaac_ros_apriltag_bridge.launch.py"
            )
        if not launch_path.is_file():
            raise NvidiaAprilTagUnavailable(f"NVIDIA AprilTag launch 文件不存在：{launch_path}")
        # ROS Jazzy's Python launch stack is installed for the system Python;
        # the Conda interpreter intentionally remains free of ROS launch-only
        # dependencies such as ``lark``.
        ros_python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
        command = [ros_python, str(launch_path)]
        environment = _with_cuda_library_path(os.environ)
        environment.update(
            {
                "HIKROBOT_APRILTAG_SIZE_M": f"{float(getattr(self._config, 'tag_size_m', 0.04835)):.9g}",
                "HIKROBOT_APRILTAG_FAMILY": self._family,
                "HIKROBOT_APRILTAG_BACKENDS": "CUDA",
                "HIKROBOT_APRILTAG_MAX_TAGS": str(self._max_tags),
                "HIKROBOT_APRILTAG_TILE_SIZE": str(self._tile_size),
                # The launch description uses this suffix for the component
                # container and node as well.  It prevents a short overlap
                # during a GUI detector/settings rebuild from colliding with
                # an existing fixed ROS name.
                "HIKROBOT_APRILTAG_INSTANCE": f"{os.getpid()}_{id(self) % 1000000}",
            }
        )
        try:
            self._process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                start_new_session=True,
            )
        except OSError as exc:
            raise NvidiaAprilTagUnavailable(f"启动 NVIDIA AprilTag ROS 节点失败：{exc}") from exc

        # The launch process returns before the composable node has finished
        # loading.  Do not publish the first camera frame until the CUDA node
        # has created its image subscription; otherwise the first 250 ms
        # timeout is mistaken for a dead detector and closes a healthy node.
        deadline = time.monotonic() + max(3.0, self._timeout_s * 12.0)
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise NvidiaAprilTagRuntimeError(
                    "NVIDIA AprilTag launch 进程在订阅 image 前退出"
                )
            try:
                if int(self._node.count_subscribers(self._image_topic)) > 0:
                    return
            except Exception:
                pass
            time.sleep(0.05)
        raise NvidiaAprilTagRuntimeError(
            "NVIDIA AprilTag 节点未在 {:.1f} s 内订阅 image".format(
                max(3.0, self._timeout_s * 12.0)
            )
        )

    def _scaled_matrix(self, image_size: tuple[int, int]) -> np.ndarray:
        result = self._camera_matrix.copy()
        if self._source_size is None or self._source_size == image_size:
            return result
        source_width, source_height = self._source_size
        width, height = image_size
        if source_width > 0 and source_height > 0:
            result[0, 0] *= width / source_width
            result[0, 2] *= width / source_width
            result[1, 1] *= height / source_height
            result[1, 2] *= height / source_height
        return result

    def _rectify(self, rgb: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = rgb.shape[:2]
        zeros = np.zeros_like(self._distortion)
        if np.allclose(self._distortion, 0.0):
            return np.ascontiguousarray(rgb), zeros
        key = (width, height)
        maps = self._maps.get(key)
        if maps is None:
            map_x, map_y = cv2.initUndistortRectifyMap(
                matrix,
                self._distortion,
                None,
                matrix,
                (width, height),
                cv2.CV_32FC1,
            )
            maps = (map_x, map_y)
            self._maps[key] = maps
        return cv2.remap(rgb, maps[0], maps[1], cv2.INTER_LINEAR), zeros

    def _camera_info(self, stamp: Any, matrix: np.ndarray, width: int, height: int) -> Any:
        msg = self._camera_info_type()
        msg.header.stamp = stamp
        msg.header.frame_id = "hikrobot_camera_optical_frame"
        msg.width = int(width)
        msg.height = int(height)
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0] * 5
        msg.k = matrix.reshape(-1).tolist()
        msg.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()
        msg.p = [
            float(matrix[0, 0]),
            float(matrix[0, 1]),
            float(matrix[0, 2]),
            0.0,
            0.0,
            float(matrix[1, 1]),
            float(matrix[1, 2]),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return msg

    def _image(self, stamp: Any, rgb: np.ndarray) -> Any:
        msg = self._image_type()
        msg.header.stamp = stamp
        msg.header.frame_id = "hikrobot_camera_optical_frame"
        msg.height, msg.width = int(rgb.shape[0]), int(rgb.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(rgb.shape[1] * 3)
        msg.data = np.ascontiguousarray(rgb).tobytes()
        return msg

    def _detection_callback(self, message: Any) -> None:
        key = _stamp_key(message.header.stamp)
        with self._condition:
            self._messages[key] = message
            self._latest_message = message
            self._latest_key = key
            # Only one request is in flight, but keep the map bounded if a
            # ROS graph republishes stale detections.
            if len(self._messages) > 8:
                oldest = next(iter(self._messages))
                self._messages.pop(oldest, None)
            self._condition.notify_all()

    def detect(
        self,
        gray: np.ndarray,
        *,
        image_size: tuple[int, int] | None = None,
        rgb: np.ndarray | None = None,
    ) -> list[NvidiaPoseRecord]:
        del gray
        if self._closed:
            raise NvidiaAprilTagRuntimeError("NVIDIA AprilTag 后端已关闭")
        if rgb is None:
            raise ValueError("NVIDIA AprilTag 后端需要 RGB 图像")
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        actual_size = (width, height) if image_size is None else image_size
        matrix = self._scaled_matrix(actual_size)
        rectified, rectified_distortion = self._rectify(rgb, matrix)
        stamp = self._node.get_clock().now().to_msg()
        key = _stamp_key(stamp)
        with self._condition:
            self._messages.pop(key, None)
        self._camera_info_pub.publish(
            self._camera_info(stamp, matrix, width, height)
        )
        self._image_pub.publish(self._image(stamp, rectified))

        deadline = time.monotonic() + self._timeout_s
        message = None
        with self._condition:
            while not self._closed:
                message = self._messages.pop(key, None)
                if message is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
        if message is None:
            if self._process is not None and self._process.poll() is not None:
                raise NvidiaAprilTagRuntimeError(
                    "NVIDIA AprilTag ROS 节点已退出；请检查 Isaac ROS/CUDA/VPI 日志"
                )
            subscribers = 0
            try:
                subscribers = int(self._node.count_subscribers(self._image_topic))
            except Exception:
                pass
            self._last_error = (
                "NVIDIA AprilTag 节点未在 {:.0f} ms 内返回结果（image subscribers={}）".format(
                    self._timeout_s * 1000.0, subscribers
                )
            )
            raise NvidiaAprilTagRuntimeError(self._last_error)
        return self._to_records(message, matrix, rectified_distortion, actual_size)

    def _to_records(
        self,
        message: Any,
        matrix: np.ndarray,
        rectified_distortion: np.ndarray,
        image_size: tuple[int, int],
    ) -> list[NvidiaPoseRecord]:
        records: list[NvidiaPoseRecord] = []
        allowed = set(getattr(self._config, "tag_ids", ()))
        object_points = _square_object_points(float(getattr(self._config, "tag_size_m", 0.04835)))
        for detection in getattr(message, "detections", []):
            tag_id = int(getattr(detection, "id", -1))
            if allowed and tag_id not in allowed:
                continue
            corners = np.array(
                [
                    [float(point.x), float(point.y)]
                    for point in getattr(detection, "corners", [])
                ],
                dtype=np.float64,
            ).reshape(-1, 2)
            if corners.shape != (4, 2):
                continue
            pose_msg = getattr(getattr(getattr(detection, "pose", None), "pose", None), "pose", None)
            if pose_msg is None:
                continue
            position = pose_msg.position
            orientation = pose_msg.orientation
            tvec = np.array(
                [[float(position.x)], [float(position.y)], [float(position.z)]],
                dtype=np.float64,
            )
            rotation = _quaternion_to_matrix(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
            rvec, _ = cv2.Rodrigues(rotation)
            projected_rectified, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                matrix,
                rectified_distortion,
            )
            error = float(
                np.mean(
                    np.linalg.norm(
                        projected_rectified.reshape(-1, 2) - corners, axis=1
                    )
                )
            )
            max_error = getattr(self._config, "max_reprojection_error_px", None)
            if max_error is not None and error > float(max_error):
                continue
            projected_raw, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                matrix,
                self._distortion,
            )
            rvec, rotation = remap_tag_rotation(rotation)
            records.append(
                NvidiaPoseRecord(
                    tag_id=tag_id,
                    corners=projected_raw.reshape(4, 2).astype(np.float32),
                    rvec=rvec,
                    tvec=tvec,
                    rotation_matrix=rotation,
                    euler_xyz_deg=_rotation_to_euler_xyz(rotation),
                    translation_mm=tvec.reshape(3) * 1000.0,
                    reprojection_error_px=error,
                    family=self._family,
                )
            )
        return records

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        with getattr(self, "_condition", threading.Condition()):
            self._condition.notify_all()
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            # ``start_new_session=True`` gives the launch process and its
            # component container a private process group.  Terminating only
            # the Python launch parent leaves component_container_mt orphaned
            # and causes the next GUI start to collide on DDS shared memory.
            try:
                process_group = os.getpgid(process.pid)
            except OSError:
                process_group = None
            try:
                if process_group is not None:
                    os.killpg(process_group, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if process_group is not None:
                        os.killpg(process_group, signal.SIGKILL)
                    else:
                        process.kill()
                except OSError:
                    pass
        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=1.0)
            except TypeError:
                executor.shutdown()
            except Exception:
                pass
        thread = getattr(self, "_spin_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        node = getattr(self, "_node", None)
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if getattr(self, "_owns_context", False):
            try:
                if self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:
                pass

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown order
        try:
            self.close()
        except Exception:
            pass
