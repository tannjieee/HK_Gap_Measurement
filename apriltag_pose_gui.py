#!/usr/bin/env python3
"""HIKROBOT AprilTag 6D pose viewer.

The application deliberately keeps camera ownership in :mod:`hikrobot_gui`'s
``CameraController``.  AprilTag detection runs in a worker thread and only
publishes immutable NumPy frames to the Qt thread, so feature writes from the
optional GenICam panel cannot race an SDK call.  OpenCV's built-in AprilTag
dictionary is used when ``apriltag_core`` is not available; this is included in
the installed OpenCV wheel and avoids a second native detector dependency.
When ``detector.backend`` is ``auto`` or ``nvidia``, an installed NVIDIA Isaac
ROS AprilTag CUDA component is used first; the automatic mode falls back to
OpenCV CPU when the ROS/CUDA component is unavailable.

Example::

    conda run -n revo3_ros python apriltag_pose_gui.py \
        --config config/apriltag_pose.yaml

The pose convention is OpenCV's camera frame (X right, Y down, Z forward).
Translation is shown in millimetres and Euler angles are XYZ degrees.  When a
reference ID is configured, ``T_reference_tag`` is also shown for every other
detected tag.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# opencv-python bundles a Qt plugin path which can override the system/PyQt5
# plugin.  Save and restore it around the OpenCV import, as the calibration GUI
# does, because this application uses QImage/QPainter rather than cv2.imshow.
_QT_ENV_KEYS = ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR")
_qt_env = {key: os.environ.get(key) for key in _QT_ENV_KEYS}

import cv2  # type: ignore
import numpy as np
import yaml

for _key, _value in _qt_env.items():
    if _value is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _value

from PyQt5.QtCore import QRect, QSignalBlocker, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

cv2.setNumThreads(2)

from apriltag_coordinates import remap_tag_rotation


def _as_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "" or str(value).strip().lower() in {
        "none",
        "null",
        "all",
    }:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "on", "1", "y"}:
            return True
        if text in {"false", "no", "off", "0", "n"}:
            return False
    return bool(value) if value is not None else default


def _nested(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, Mapping) else {}


@dataclass
class PoseGuiConfig:
    """Normalized settings accepted by the GUI and detector."""

    config_path: Path
    intrinsics_file: Path
    family: str = "tag36h11"
    tag_size_mm: float = 48.35
    tag_ids: tuple[int, ...] = ()
    reference_tag_id: int | None = None
    axis_length_mm: float = 24.175
    draw_axis: bool = True
    draw_corners: bool = True
    show_text: bool = True
    show_reprojection_error: bool = True
    relative_enabled: bool = True
    backend: str = "auto"
    nthreads: int = 2
    nvidia_auto_start: bool = True
    nvidia_timeout_ms: float = 250.0
    nvidia_max_tags: int = 64
    nvidia_tile_size: int = 4
    nvidia_launch_file: str = ""
    max_reprojection_error_px: float | None = 5.0
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    max_hamming: int = 0
    camera_name: str = ""
    camera_serial: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def tag_size_m(self) -> float:
        return self.tag_size_mm / 1000.0

    @property
    def axis_length_m(self) -> float:
        return self.axis_length_mm / 1000.0


def _resolve_path(path_value: Any, config_path: Path) -> Path:
    path = Path(str(path_value or ""))
    if path.is_absolute():
        return path.expanduser().resolve()
    # Configuration files are commonly kept under ``config/`` while paths are
    # written relative to the project root (``calibration_results/...``).
    # Accept both that convention and paths relative to the YAML itself.
    candidates = [
        config_path.parent / path,
        config_path.parent.parent / path,
        Path.cwd() / path,
        Path(__file__).resolve().parent / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.expanduser().resolve()
    return candidates[0].expanduser().resolve()


def load_pose_config(path: str | Path) -> PoseGuiConfig:
    """Load flat or nested AprilTag YAML configuration.

    Both the original ``tag_family``/``intrinsics_file`` schema and the newer
    ``tag.family``/``camera.intrinsics_file`` schema are accepted.  This makes
    the GUI safe to use with configuration files copied from earlier runs.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"AprilTag 配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("AprilTag 配置必须是 YAML 映射")

    tag = _nested(raw, "tag")
    camera = _nested(raw, "camera")
    detector = _nested(raw, "detector")
    nvidia = _nested(raw, "nvidia") or _nested(detector, "nvidia")
    pose = _nested(raw, "pose")
    display = _nested(raw, "display") or pose
    relative = _nested(raw, "relative_pose") or _nested(raw, "relative")

    backend = str(
        detector.get("backend", raw.get("backend", "auto"))
    ).strip().lower() or "auto"

    family = str(
        tag.get("family", raw.get("tag_family", raw.get("family", "tag36h11")))
    ).strip()
    size = _as_float(tag.get("size_mm", raw.get("tag_size_mm", 48.35)), 48.35)
    if size <= 0:
        raise ValueError("tag_size_mm 必须为正数")

    intrinsic_value = camera.get(
        "intrinsics_file", raw.get("intrinsics_file", "")
    )
    if not intrinsic_value:
        raise ValueError("配置中缺少 camera.intrinsics_file")
    tag_ids_value = raw.get("tag_ids", tag.get("ids", ()))
    if tag_ids_value is None:
        tag_ids_value = ()
    if isinstance(tag_ids_value, (str, int)):
        tag_ids_value = [tag_ids_value]
    try:
        tag_ids = tuple(sorted({int(item) for item in tag_ids_value}))
    except (TypeError, ValueError) as exc:
        raise ValueError("tag_ids 必须是整数列表") from exc

    reference_value = relative.get(
        "reference_tag_id",
        raw.get("reference_tag_id", _nested(raw, "relative").get("reference_tag_id")),
    )
    pose_reference = _as_int_or_none(reference_value)
    max_error_value = detector.get("max_reprojection_error_px", 5.0)
    max_error = None if max_error_value is None else _as_float(max_error_value, 5.0)
    if max_error is not None and max_error <= 0:
        max_error = None

    return PoseGuiConfig(
        config_path=config_path,
        intrinsics_file=_resolve_path(intrinsic_value, config_path),
        family=family,
        tag_size_mm=size,
        tag_ids=tag_ids,
        reference_tag_id=pose_reference,
        axis_length_mm=max(
            0.001,
            _as_float(display.get("axis_length_mm", size * 0.5), size * 0.5),
        ),
        draw_axis=_as_bool(display.get("draw_axis", True), True),
        draw_corners=_as_bool(display.get("draw_corners", True), True),
        show_text=_as_bool(display.get("show_text", True), True),
        show_reprojection_error=_as_bool(
            display.get("show_reprojection_error", True), True
        ),
        relative_enabled=_as_bool(
            relative.get("enabled", True), True
        ),
        backend=backend,
        nthreads=max(1, int(_as_float(detector.get("nthreads", 2), 2))),
        nvidia_auto_start=_as_bool(nvidia.get("auto_start", True), True),
        nvidia_timeout_ms=max(30.0, _as_float(nvidia.get("timeout_ms", 250.0), 250.0)),
        nvidia_max_tags=max(1, int(_as_float(nvidia.get("max_tags", 64), 64))),
        nvidia_tile_size=max(1, int(_as_float(nvidia.get("tile_size", 4), 4))),
        nvidia_launch_file=str(nvidia.get("launch_file", "") or ""),
        max_reprojection_error_px=max_error,
        quad_decimate=max(0.1, _as_float(detector.get("quad_decimate", 1.0), 1.0)),
        quad_sigma=_as_float(detector.get("quad_sigma", 0.0), 0.0),
        refine_edges=_as_bool(detector.get("refine_edges", True), True),
        decode_sharpening=_as_float(
            detector.get("decode_sharpening", 0.25), 0.25
        ),
        max_hamming=max(0, int(_as_float(detector.get("max_hamming", 0), 0))),
        camera_name=str(camera.get("name", "")),
        camera_serial=str(camera.get("serial", "")),
        raw=dict(raw),
    )


def _matrix_from_document(document: Mapping[str, Any], key: str) -> np.ndarray:
    value = document.get(key)
    if isinstance(value, Mapping):
        value = value.get("data")
    if value is None:
        raise ValueError(f"内参文件缺少 {key}")
    array = np.asarray(value, dtype=np.float64)
    if key == "camera_matrix" and array.size != 9:
        raise ValueError("camera_matrix 必须包含 9 个数")
    if key == "distortion_coefficients" and array.size < 4:
        raise ValueError("distortion_coefficients 至少需要 4 个数")
    return array.reshape(3, 3) if key == "camera_matrix" else array.reshape(1, -1)


def load_intrinsics(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    """Read calibration_core YAML/JSON and return K, D, and source size."""

    intrinsic_path = Path(path).expanduser().resolve()
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"内参文件不存在：{intrinsic_path}")
    try:
        with intrinsic_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"无法解析内参 YAML：{exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("内参文件必须是 YAML/JSON 映射")
    camera_matrix = _matrix_from_document(document, "camera_matrix")
    distortion = _matrix_from_document(document, "distortion_coefficients")
    width = document.get("image_width")
    height = document.get("image_height")
    source_size = None
    if width is not None and height is not None:
        try:
            source_size = (int(width), int(height))
        except (TypeError, ValueError):
            source_size = None
    return camera_matrix, distortion, source_size


def scale_camera_matrix(
    camera_matrix: np.ndarray,
    source_size: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Scale calibration K when the camera is streaming another resolution."""

    if source_size is None or source_size == image_size:
        return np.asarray(camera_matrix, dtype=np.float64).copy()
    source_width, source_height = source_size
    width, height = image_size
    if source_width <= 0 or source_height <= 0:
        return np.asarray(camera_matrix, dtype=np.float64).copy()
    result = np.asarray(camera_matrix, dtype=np.float64).copy()
    sx, sy = width / source_width, height / source_height
    result[0, 0] *= sx
    result[0, 2] *= sx
    result[1, 1] *= sy
    result[1, 2] *= sy
    return result


def _rotation_to_euler_xyz(rotation: np.ndarray) -> np.ndarray:
    """Return one continuous XYZ Euler decomposition in degrees."""

    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    singular = sy < 1e-9
    if not singular:
        x = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        y = math.atan2(-float(rotation[2, 0]), sy)
        z = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        x = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        y = math.atan2(-float(rotation[2, 0]), sy)
        z = 0.0
    return np.degrees([x, y, z]).astype(np.float64)


@dataclass
class PoseRecord:
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


@dataclass
class RelativeRecord:
    tag_id: int
    reference_id: int
    translation_mm: np.ndarray
    rvec: np.ndarray
    euler_xyz_deg: np.ndarray
    rotation_matrix: np.ndarray


def _object_corners(tag_size_m: float) -> np.ndarray:
    half = tag_size_m / 2.0
    # OpenCV ArUco corner order is top-left, top-right, bottom-right,
    # bottom-left. These are native solver coordinates; the published tag
    # frame is remapped only after solving and checking reprojection error.
    # SOLVEPNP_IPPE_SQUARE requires this exact clockwise order.  It matches the
    # canonical TL/TR/BR/BL image corner order used by apriltag_pose_core.
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


_APRILTAG_DICTIONARIES = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}


class OpenCVAprilTagEstimator:
    """Small, dependency-free AprilTag pose backend based on cv2.aruco."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        config: PoseGuiConfig,
    ) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "当前 OpenCV 没有 cv2.aruco；请安装 opencv-contrib-python "
                "（不要与 opencv-python 同时安装）"
            )
        family = config.family.strip().lower()
        dictionary_name = _APRILTAG_DICTIONARIES.get(family)
        if dictionary_name is None:
            raise ValueError(
                f"不支持的 AprilTag family {config.family!r}；可用："
                + ", ".join(sorted(_APRILTAG_DICTIONARIES))
            )
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            # OpenCV has changed h/H capitalization across releases.
            dictionary_id = getattr(
                cv2.aruco, dictionary_name.replace("h", "H"), None
            )
        if dictionary_id is None:
            raise RuntimeError("当前 OpenCV 未编译 AprilTag 字典支持")
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        # OpenCV 4.6's Python constructor is not the factory and can create
        # an invalid native object. ROS Jazzy's system build needs _create().
        factory = getattr(cv2.aruco, "DetectorParameters_create", None)
        self.parameters = factory() if factory else cv2.aruco.DetectorParameters()
        for name, value in (
            ("aprilTagQuadDecimate", config.quad_decimate),
            ("aprilTagQuadSigma", config.quad_sigma),
            ("aprilTagDecodeSharpening", config.decode_sharpening),
            ("aprilTagMaxLineFitMse", 10.0),
        ):
            if hasattr(self.parameters, name):
                try:
                    setattr(self.parameters, name, float(value))
                except (TypeError, ValueError):
                    pass
        if hasattr(self.parameters, "cornerRefinementMethod") and config.refine_edges:
            # CORNER_REFINE_APRILTAG exists in OpenCV >= 4.7.  Leave the
            # default untouched if a vendor build omits it.
            refine = getattr(cv2.aruco, "CORNER_REFINE_APRILTAG", None)
            if refine is not None:
                self.parameters.cornerRefinementMethod = refine
        self.detector = (cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
                         if hasattr(cv2.aruco, "ArucoDetector") else None)
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(distortion, dtype=np.float64)
        self.config = config
        self.object_points = _object_corners(config.tag_size_m)

    def detect(
        self,
        gray: np.ndarray,
        *,
        camera_matrix: np.ndarray | None = None,
    ) -> list[PoseRecord]:
        if gray.ndim != 2:
            raise ValueError("AprilTag detector expects a grayscale image")
        matrix = self.camera_matrix if camera_matrix is None else np.asarray(
            camera_matrix, dtype=np.float64
        )
        if self.detector is None:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters)
        else:
            corners, ids, _rejected = self.detector.detectMarkers(gray)
        if ids is None or not corners:
            return []
        accepted = set(self.config.tag_ids)
        records: list[PoseRecord] = []
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            tag_id = int(marker_id)
            if accepted and tag_id not in accepted:
                continue
            points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            success, rvec, tvec = cv2.solvePnP(
                self.object_points,
                points,
                matrix,
                self.distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                success, rvec, tvec = cv2.solvePnP(
                    self.object_points,
                    points,
                    matrix,
                    self.distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            if not success:
                continue
            if float(np.asarray(tvec).reshape(-1)[2]) <= 0:
                continue
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            rotation, _ = cv2.Rodrigues(rvec)
            projected, _ = cv2.projectPoints(
                self.object_points,
                rvec,
                tvec,
                matrix,
                self.distortion,
            )
            error = float(
                np.mean(
                    np.linalg.norm(
                        projected.reshape(-1, 2) - points.reshape(-1, 2), axis=1
                    )
                )
            )
            # Some OpenCV releases return a high-error IPPE solution for an
            # exact, fronto-parallel square.  Re-run ITERATIVE in that case;
            # otherwise a valid tag can be discarded by the reprojection gate.
            if error > 1.0:
                try:
                    iterative_ok, iterative_rvec, iterative_tvec = cv2.solvePnP(
                        self.object_points,
                        points,
                        matrix,
                        self.distortion,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                except cv2.error:
                    iterative_ok = False
                if iterative_ok:
                    iterative_projected, _ = cv2.projectPoints(
                        self.object_points,
                        iterative_rvec,
                        iterative_tvec,
                        matrix,
                        self.distortion,
                    )
                    iterative_error = float(
                        np.mean(
                            np.linalg.norm(
                                iterative_projected.reshape(-1, 2)
                                - points.reshape(-1, 2),
                                axis=1,
                            )
                        )
                    )
                    if iterative_error < error:
                        rvec, tvec = iterative_rvec, iterative_tvec
                        rotation, _ = cv2.Rodrigues(rvec)
                        error = iterative_error
            if (
                self.config.max_reprojection_error_px is not None
                and error > self.config.max_reprojection_error_px
            ):
                continue
            rvec, rotation = remap_tag_rotation(rotation)
            records.append(
                PoseRecord(
                    tag_id=tag_id,
                    corners=points.astype(np.float32),
                    rvec=rvec,
                    tvec=tvec,
                    rotation_matrix=rotation,
                    euler_xyz_deg=_rotation_to_euler_xyz(rotation),
                    translation_mm=tvec.reshape(3) * 1000.0,
                    reprojection_error_px=error,
                    family=self.config.family,
                )
            )
        return records


def _coerce_pose_record(value: Any, family: str) -> PoseRecord | None:
    """Adapt an ``apriltag_core.TagPose`` without coupling its exact class."""

    try:
        tag_id = int(getattr(value, "tag_id"))
        corners = np.asarray(getattr(value, "corners"), dtype=np.float32).reshape(4, 2)
        rvec = np.asarray(getattr(value, "rvec"), dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(getattr(value, "tvec"), dtype=np.float64).reshape(3, 1)
    except (AttributeError, TypeError, ValueError):
        return None
    rotation = getattr(value, "rotation_matrix", None)
    if rotation is None:
        rotation, _ = cv2.Rodrigues(rvec)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    euler = getattr(value, "euler_xyz_deg", None)
    if euler is None:
        euler = _rotation_to_euler_xyz(rotation)
    translation = getattr(value, "translation_mm", None)
    if translation is None:
        translation = tvec.reshape(3) * 1000.0
    error = getattr(value, "reprojection_error_px", float("nan"))
    try:
        error = float(error) if error is not None else float("nan")
    except (TypeError, ValueError):
        error = float("nan")
    margin_value = getattr(value, "decision_margin", float("nan"))
    try:
        margin = float(margin_value) if margin_value is not None else float("nan")
    except (TypeError, ValueError):
        margin = float("nan")
    return PoseRecord(
        tag_id=tag_id,
        corners=corners,
        rvec=rvec,
        tvec=tvec,
        rotation_matrix=rotation,
        euler_xyz_deg=np.asarray(euler, dtype=np.float64).reshape(3),
        translation_mm=np.asarray(translation, dtype=np.float64).reshape(3),
        reprojection_error_px=error,
        decision_margin=margin,
        family=str(getattr(value, "family", family)),
    )


class PoseEstimator:
    """Prefer the project core API, with a tested OpenCV fallback."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        config: PoseGuiConfig,
        source_size: tuple[int, int] | None = None,
    ) -> None:
        self.config = config
        self.source_size = source_size
        self._nvidia: Any | None = None
        self._nvidia_error: Exception | None = None
        self._fallback: OpenCVAprilTagEstimator | None = None
        self._fallback_error: Exception | None = None

        requested_backend = str(config.backend).strip().lower()
        wants_nvidia = requested_backend in {
            "auto",
            "nvidia",
            "cuda",
            "isaac_ros",
            "isaac-ros",
        }
        forced_nvidia = requested_backend in {
            "nvidia",
            "cuda",
            "isaac_ros",
            "isaac-ros",
        }
        if wants_nvidia:
            try:
                from nvidia_apriltag_backend import NvidiaAprilTagEstimator

                self._nvidia = NvidiaAprilTagEstimator(
                    camera_matrix,
                    distortion,
                    config,
                    source_size,
                    auto_start=config.nvidia_auto_start,
                    timeout_s=config.nvidia_timeout_ms / 1000.0,
                    max_tags=config.nvidia_max_tags,
                    tile_size=config.nvidia_tile_size,
                    launch_file=config.nvidia_launch_file or None,
                )
            except Exception as exc:
                self._nvidia_error = exc
                self._nvidia = None
                if forced_nvidia:
                    raise RuntimeError(f"NVIDIA CUDA AprilTag 后端不可用：{exc}") from exc

        try:
            self._fallback = OpenCVAprilTagEstimator(camera_matrix, distortion, config)
        except Exception as exc:
            # The core may still be usable through pupil_apriltags/apriltag on
            # systems whose OpenCV build has no aruco module.
            self._fallback_error = exc
        self._core: Any | None = None
        try:
            # The project core is named apriltag_pose_core.  Keep the shorter
            # name as a compatibility alias for downstream revisions.
            try:
                import apriltag_pose_core as apriltag_core  # type: ignore
            except ImportError:
                import apriltag_core  # type: ignore

            estimator_type = getattr(apriltag_core, "AprilTagPoseEstimator", None)
            if estimator_type is not None:
                attempts: list[Any] = []
                # Construct a core config from the *current GUI values* first.
                # Loading the YAML directly would silently ignore edits made
                # to family/tag size/ID controls until the next file reload.
                core_config_type = getattr(apriltag_core, "AprilTagPoseConfig", None)
                core_camera_type = getattr(apriltag_core, "CameraIntrinsics", None)
                if core_config_type is not None and core_camera_type is not None:
                    try:
                        core_camera = core_camera_type(
                            camera_matrix,
                            distortion,
                            image_size=source_size,
                        )
                        core_config = core_config_type(
                            camera=core_camera,
                            tag_size_m=config.tag_size_m,
                            family=config.family,
                            backend=config.backend,
                            quad_decimate=config.quad_decimate,
                            quad_sigma=config.quad_sigma,
                            nthreads=config.nthreads,
                            refine_edges=config.refine_edges,
                            decode_sharpening=config.decode_sharpening,
                            max_hamming=config.max_hamming,
                            allowed_ids=(config.tag_ids or None),
                            max_reprojection_error_px=config.max_reprojection_error_px,
                        )
                        attempts.append(lambda cfg=core_config: estimator_type(cfg))
                    except (TypeError, ValueError, RuntimeError):
                        pass
                # Compatibility with an older core implementation and with
                # downstream users that expose only a path-based constructor.
                if hasattr(apriltag_core, "load_pose_config"):
                    attempts.append(
                        lambda: estimator_type(
                            apriltag_core.load_pose_config(config.config_path)
                        )
                    )
                attempts.extend(
                    [
                        lambda: estimator_type(config.config_path),
                        lambda: estimator_type(
                            camera_matrix,
                            distortion,
                            tag_size_m=config.tag_size_m,
                            family=config.family,
                            tag_ids=config.tag_ids,
                            detector_config={
                                "quad_decimate": config.quad_decimate,
                                "quad_sigma": config.quad_sigma,
                                "refine_edges": config.refine_edges,
                                "decode_sharpening": config.decode_sharpening,
                                "max_hamming": config.max_hamming,
                                "max_reprojection_error_px": config.max_reprojection_error_px,
                            },
                        ),
                        lambda: estimator_type(
                            camera_matrix=camera_matrix,
                            distortion_coefficients=distortion,
                            tag_size_m=config.tag_size_m,
                            family=config.family,
                        ),
                        lambda: estimator_type(camera_matrix, distortion, config.tag_size_m),
                    ]
                )
                for attempt in attempts:
                    try:
                        self._core = attempt()
                        break
                    except (TypeError, ValueError, RuntimeError, OSError):
                        continue
        except (ImportError, AttributeError, OSError):
            self._core = None
        if self._core is None and self._fallback is None:
            detail = str(self._fallback_error) if self._fallback_error else "未知错误"
            raise RuntimeError(f"没有可用的 AprilTag 检测后端：{detail}")

    @property
    def backend_name(self) -> str:
        if self._nvidia is not None:
            return "NVIDIA CUDA (Isaac ROS)"
        if self._core is not None:
            if self._nvidia_error is not None:
                return "AprilTag core（CUDA 回退）"
            return "AprilTag core"
        if self._fallback is not None:
            return "OpenCV CPU"
        return "不可用"

    def update_config(self, config: PoseGuiConfig) -> None:
        """Apply display/relative settings without restarting a GPU node."""

        self.config = config
        if self._nvidia is not None:
            self._nvidia._config = config
        if self._fallback is not None:
            self._fallback.config = config

    def close(self) -> None:
        nvidia = self._nvidia
        self._nvidia = None
        if nvidia is not None:
            try:
                nvidia.close()
            except Exception:
                pass

    def detect(
        self,
        gray: np.ndarray,
        *,
        image_size: tuple[int, int] | None = None,
        rgb: np.ndarray | None = None,
    ) -> list[PoseRecord]:
        if self._nvidia is not None:
            try:
                raw = self._nvidia.detect(
                    gray,
                    image_size=image_size,
                    rgb=rgb,
                )
                return [
                    record
                    for value in raw
                    if (record := _coerce_pose_record(value, self.config.family))
                    is not None
                ]
            except Exception as exc:
                forced_nvidia = str(self.config.backend).strip().lower() in {
                    "nvidia",
                    "cuda",
                    "isaac_ros",
                    "isaac-ros",
                }
                if forced_nvidia:
                    raise
                # ``auto`` must not make the live camera unusable when Isaac
                # ROS is installed but its component cannot start (for
                # example, a missing VPI/CUDA runtime).  Fall back once and
                # keep the error available to the status bar/debugger.
                self._nvidia_error = exc
                self.close()
        if self._core is not None:
            try:
                try:
                    raw = self._core.detect(gray, image_size=image_size)
                except TypeError:
                    raw = self._core.detect(gray)
                records = [
                    record
                    for value in raw
                    if (record := _coerce_pose_record(value, self.config.family))
                    is not None
                ]
                if records or not raw:
                    return records
            except (AttributeError, TypeError, ValueError, RuntimeError):
                # A partially compatible core should not make the live view
                # unusable; fall back to OpenCV and continue showing frames.
                pass
        if self._fallback is None:
            raise RuntimeError("没有可用的 AprilTag 检测后端")
        matrix = self._fallback.camera_matrix
        if image_size is not None:
            matrix = scale_camera_matrix(
                matrix,
                self.source_size,
                image_size,
            )
        return self._fallback.detect(gray, camera_matrix=matrix)


def relative_poses(
    poses: Sequence[PoseRecord], reference_id: int | None
) -> dict[int, RelativeRecord]:
    if reference_id is None and len(poses) >= 2:
        # A configured reference is preferred, but showing a deterministic
        # pairwise result is more useful than silently omitting relative pose
        # when the user leaves the field blank.  The lowest visible ID is the
        # documented automatic reference.
        reference_id = min(pose.tag_id for pose in poses)
    if reference_id is None:
        return {}
    reference = next((pose for pose in poses if pose.tag_id == reference_id), None)
    if reference is None:
        return {}
    reference_transform = np.eye(4, dtype=np.float64)
    reference_transform[:3, :3] = reference.rotation_matrix
    reference_transform[:3, 3] = reference.tvec.reshape(3)
    inverse_reference = np.linalg.inv(reference_transform)
    result: dict[int, RelativeRecord] = {}
    for pose in poses:
        if pose.tag_id == reference_id:
            continue
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = pose.rotation_matrix
        transform[:3, 3] = pose.tvec.reshape(3)
        relative = inverse_reference @ transform
        relative_rvec, _ = cv2.Rodrigues(relative[:3, :3])
        result[pose.tag_id] = RelativeRecord(
            tag_id=pose.tag_id,
            reference_id=reference_id,
            translation_mm=relative[:3, 3] * 1000.0,
            rvec=relative_rvec.reshape(3),
            euler_xyz_deg=_rotation_to_euler_xyz(relative[:3, :3]),
            rotation_matrix=relative[:3, :3],
        )
    return result


def _format_vector(values: Iterable[float], unit: str = "") -> str:
    numbers = [float(value) for value in values]
    suffix = unit
    return "[" + ", ".join(f"{value:+.1f}" for value in numbers) + f"]{suffix}"


def _format_xyz(values: Iterable[float]) -> str:
    return "  ".join(
        f"{axis}={float(value):+.1f}" for axis, value in zip("XYZ", values)
    ) + " mm"


def draw_pose_overlay(
    bgr: np.ndarray,
    poses: Sequence[PoseRecord],
    relatives: Mapping[int, RelativeRecord],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    config: PoseGuiConfig,
) -> np.ndarray:
    """Draw corners, axes, and compact 6D pose labels onto a BGR frame."""

    image = np.ascontiguousarray(bgr)
    height, width = image.shape[:2]
    axis_labels = []
    for index, pose in enumerate(poses):
        points = np.round(pose.corners).astype(np.int32).reshape(-1, 1, 2)
        if config.draw_corners:
            cv2.polylines(image, [points], True, (0, 220, 255), 2, cv2.LINE_AA)
            center = tuple(np.round(pose.corners.mean(axis=0)).astype(int))
            cv2.circle(image, center, 4, (0, 220, 255), -1, cv2.LINE_AA)
        if config.draw_axis:
            try:
                cv2.drawFrameAxes(
                    image,
                    camera_matrix,
                    distortion,
                    pose.rvec,
                    pose.tvec,
                    config.axis_length_m,
                    2,
                )
                endpoints, _ = cv2.projectPoints(
                    np.eye(3, dtype=np.float64) * config.axis_length_m,
                    pose.rvec,
                    pose.tvec,
                    camera_matrix,
                    distortion,
                )
                for axis, point, color in zip(
                    "XYZ",
                    endpoints.reshape(3, 2),
                    ((0, 0, 255), (0, 255, 0), (255, 0, 0)),
                ):
                    if not np.isfinite(point).all():
                        continue
                    (label_width, label_height), baseline = cv2.getTextSize(
                        axis, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                    )
                    origin = (
                        int(np.clip(point[0] + 6, 0, max(0, width - label_width - 2))),
                        int(np.clip(point[1] - 6, label_height + 2, max(label_height + 2, height - baseline - 2))),
                    )
                    # Draw after text panels so they cannot cover an axis name.
                    axis_labels.append((axis, origin, color))
            except cv2.error:
                pass
        if not config.show_text:
            continue
        x, y = np.round(pose.corners.min(axis=0)).astype(int)
        translation = _format_xyz(pose.translation_mm)
        euler = _format_vector(pose.euler_xyz_deg, " deg")
        lines = [f"ID {pose.tag_id}  camera (mm)", translation, f"rpy={euler}"]
        if config.show_reprojection_error and math.isfinite(
            pose.reprojection_error_px
        ):
            lines.append(f"err={pose.reprojection_error_px:.2f}px")
        relative = relatives.get(pose.tag_id)
        if relative is not None:
            lines.append(
                f"rel({relative.reference_id}) "
                + _format_xyz(relative.translation_mm)
            )
            lines.append("rel rpy=" + _format_vector(relative.euler_xyz_deg, " deg"))
        block_width = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            for line in lines
        ) + 6
        block_height = len(lines) * 18
        x = int(np.clip(x, 0, max(0, width - block_width)))
        # Keep the complete label above the tag, or below it near the top edge.
        text_top = y - block_height - 12
        if text_top < 0:
            text_top = int(np.max(pose.corners[:, 1])) + 12
        y = int(np.clip(text_top, 0, max(0, height - block_height - 8))) + 18
        cv2.rectangle(
            image,
            (x, max(0, y - 18)),
            (min(width - 1, x + block_width), min(height - 1, y + (len(lines) - 1) * 18 + 6)),
            (0, 0, 0),
            -1,
        )
        for line_index, line in enumerate(lines):
            text_y = y + line_index * 18
            cv2.putText(
                image,
                line,
                (x + 3, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    for axis, origin, color in axis_labels:
        cv2.putText(image, axis, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, axis, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return image


@dataclass
class PoseFrame:
    rgb: np.ndarray
    poses: list[PoseRecord]
    relatives: dict[int, RelativeRecord]
    frame_number: int
    elapsed_ms: float
    image_size: tuple[int, int]


class LatestPoseFrameBuffer:
    """Bounded one-slot buffer used to decouple detection from Qt painting.

    The detector can run faster than the GUI event loop.  Queuing every full
    1440x1080 RGB frame makes Qt retain several megabytes per pending signal
    and eventually produces visible pauses.  Replacing the previous frame is
    intentional: an old display frame is never more valuable than the latest
    camera frame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: PoseFrame | None = None

    def put(self, frame: PoseFrame) -> None:
        with self._lock:
            self._frame = frame

    def take(self) -> PoseFrame | None:
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame

    def clear(self) -> None:
        with self._lock:
            self._frame = None


class PoseCaptureThread(QThread):
    frame_ready = pyqtSignal(object)
    statistics = pyqtSignal(float, int, int, int)
    failed = pyqtSignal(str)

    def __init__(
        self,
        controller: Any,
        estimator: PoseEstimator,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        source_size: tuple[int, int] | None,
        config: PoseGuiConfig,
        display_buffer: LatestPoseFrameBuffer | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.estimator = estimator
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(distortion, dtype=np.float64)
        self.source_size = source_size
        self.config = config
        self.display_buffer = display_buffer
        self.stop_event = threading.Event()
        self.settings_lock = threading.RLock()

    def update_settings(
        self,
        estimator: PoseEstimator,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        source_size: tuple[int, int] | None,
        config: PoseGuiConfig,
    ) -> None:
        """Atomically switch detector/settings at a frame boundary."""

        with self.settings_lock:
            self.estimator = estimator
            self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
            self.distortion = np.asarray(distortion, dtype=np.float64)
            self.source_size = source_size
            self.config = config

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        frame_count = 0
        sample_start = time.monotonic()
        try:
            while not self.stop_event.is_set():
                packet = self.controller.read_rgb_frame()
                if packet is None:
                    continue
                rgb_bytes, width, height, frame_number = packet
                rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                    height, width, 3
                ).copy()
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                with self.settings_lock:
                    estimator = self.estimator
                    camera_matrix = self.camera_matrix.copy()
                    distortion = self.distortion.copy()
                    source_size = self.source_size
                    # Take a value snapshot so a simultaneous GUI edit cannot
                    # change overlay flags halfway through this frame.
                    config = replace(self.config)
                matrix = scale_camera_matrix(camera_matrix, source_size, (width, height))
                started = time.perf_counter()
                poses = estimator.detect(
                    gray,
                    image_size=(width, height),
                    rgb=rgb,
                )
                relatives = (
                    relative_poses(poses, config.reference_tag_id)
                    if config.relative_enabled
                    else {}
                )
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                display_bgr = draw_pose_overlay(
                    bgr,
                    poses,
                    relatives,
                    matrix,
                    distortion,
                    config,
                )
                display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                frame = PoseFrame(
                    rgb=np.ascontiguousarray(display_rgb),
                    poses=poses,
                    relatives=relatives,
                    frame_number=int(frame_number),
                    elapsed_ms=elapsed_ms,
                    image_size=(width, height),
                )
                if self.display_buffer is not None:
                    self.display_buffer.put(frame)
                else:  # backwards-compatible use outside the main GUI
                    self.frame_ready.emit(frame)
                frame_count += 1
                now = time.monotonic()
                elapsed = now - sample_start
                if elapsed >= 1.0:
                    self.statistics.emit(
                        frame_count / elapsed,
                        width,
                        height,
                        len(poses),
                    )
                    frame_count = 0
                    sample_start = now
        except Exception as exc:  # propagated to GUI, never crash silently
            self.failed.emit(str(exc))


class VideoWidget(QWidget):
    """Aspect-ratio-preserving QImage display."""

    def __init__(self) -> None:
        super().__init__()
        self.image = QImage()
        self.setMinimumSize(720, 576)
        self.setStyleSheet("background: #111;")

    def set_image(self, image: QImage) -> None:
        self.image = image
        self.update()

    def clear(self) -> None:
        self.image = QImage()
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        if self.image.isNull():
            painter.setPen(QColor("#aaaaaa"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待视频流")
            return
        size = self.image.size()
        size.scale(self.size(), Qt.KeepAspectRatio)
        target = QRect(
            (self.width() - size.width()) // 2,
            (self.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(target, self.image)


class MainWindow(QMainWindow):
    def __init__(
        self,
        config_path: str | Path,
        *,
        device_index: int = 0,
        serial: str | None = None,
        auto_connect: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("HIKROBOT AprilTag 6D 位姿")
        self.resize(1560, 900)
        self.config_path = Path(config_path).expanduser().resolve()
        self.pose_config = load_pose_config(self.config_path)
        self.camera_matrix, self.distortion, self.source_size = load_intrinsics(
            self.pose_config.intrinsics_file
        )
        self.estimator = PoseEstimator(
            self.camera_matrix,
            self.distortion,
            self.pose_config,
            self.source_size,
        )
        self.device_index_hint = device_index
        self.serial_override = serial
        # Prefer an explicit CLI serial, otherwise honor the camera identity
        # stored beside the AprilTag priors.  This prevents silently applying
        # one camera's calibration to another device in a multi-camera setup.
        self.serial_hint = serial or self.pose_config.camera_serial or None
        self.auto_connect = auto_connect
        self.controller: Any | None = None
        self.sdk: Any | None = None
        self.devices: list[Any] = []
        self.device_list: Any | None = None
        self.capture_thread: PoseCaptureThread | None = None
        self.display_buffer = LatestPoseFrameBuffer()
        self.display_timer = QTimer(self)
        self.display_timer.setInterval(16)
        self.display_timer.timeout.connect(self._drain_display_frame)
        self.latest_frame: PoseFrame | None = None
        self.latest_image = QImage()
        self._last_info_update = 0.0
        self._last_info_text = ""
        self.loading_ui = False
        self.feature_panel: Any | None = None
        self._feature_stream_was_running = False
        self._feature_stream_restart_allowed = False

        self.video = VideoWidget()
        self.device_combo = QComboBox()
        self.config_edit = QLineEdit(str(self.config_path))
        self.config_edit.setReadOnly(True)
        self.refresh_devices_button = QPushButton("刷新设备")
        self.connect_button = QPushButton("连接")
        self.stream_button = QPushButton("开始取流")
        self.reload_config_button = QPushButton("重新加载")
        self.browse_config_button = QPushButton("选择…")
        self.save_button = QPushButton("保存叠加帧")
        self.save_button.setEnabled(False)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("自动（CUDA 优先，失败回退 CPU）", "auto")
        self.backend_combo.addItem("NVIDIA CUDA（Isaac ROS）", "nvidia")
        self.backend_combo.addItem("OpenCV CPU", "opencv")
        self.family_combo = QComboBox()
        self.family_combo.addItems(sorted(_APRILTAG_DICTIONARIES))
        self.family_combo.setCurrentText(self.pose_config.family.lower())
        self.tag_size_spin = QDoubleSpinBox()
        self.tag_size_spin.setRange(0.1, 10000.0)
        self.tag_size_spin.setDecimals(3)
        self.tag_size_spin.setSuffix(" mm")
        self.axis_spin = QDoubleSpinBox()
        self.axis_spin.setRange(0.1, 10000.0)
        self.axis_spin.setDecimals(3)
        self.axis_spin.setSuffix(" mm")
        self.relative_check = QCheckBox("显示相对位姿")
        self.reference_check = QCheckBox("固定参考 Tag")
        self.reference_spin = QSpinBox()
        self.reference_spin.setRange(0, 1000000)
        self.reference_spin.setSpecialValueText("0")
        self.draw_axis_check = QCheckBox("坐标轴")
        self.draw_corners_check = QCheckBox("角点")
        self.show_text_check = QCheckBox("姿态文字")
        self.show_error_check = QCheckBox("重投影误差")
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(230)
        self.info_text.setStyleSheet("font-family: monospace;")
        self._load_config_into_widgets()
        self._build_ui()
        self._connect_signals()
        self._set_connected_state(False)
        self._load_optional_feature_panel()
        self.refresh_devices()
        if (
            self.auto_connect
            and self.devices
            and self.device_combo.currentData() is not None
        ):
            QTimer.singleShot(0, self.connect_camera)

    def _load_config_into_widgets(self) -> None:
        config = self.pose_config
        with QSignalBlocker(self.backend_combo):
            index = self.backend_combo.findData(config.backend.lower())
            self.backend_combo.setCurrentIndex(max(0, index))
        with QSignalBlocker(self.family_combo):
            self.family_combo.setCurrentText(config.family.lower())
        with QSignalBlocker(self.tag_size_spin):
            self.tag_size_spin.setValue(config.tag_size_mm)
        with QSignalBlocker(self.axis_spin):
            self.axis_spin.setValue(config.axis_length_mm)
        with QSignalBlocker(self.reference_check):
            self.reference_check.setChecked(config.reference_tag_id is not None)
        with QSignalBlocker(self.relative_check):
            self.relative_check.setChecked(config.relative_enabled)
        with QSignalBlocker(self.reference_spin):
            self.reference_spin.setValue(config.reference_tag_id or 0)
        for checkbox, value in (
            (self.draw_axis_check, config.draw_axis),
            (self.draw_corners_check, config.draw_corners),
            (self.show_text_check, config.show_text),
            (self.show_error_check, config.show_reprojection_error),
        ):
            with QSignalBlocker(checkbox):
                checkbox.setChecked(value)
        self.reference_spin.setEnabled(
            config.reference_tag_id is not None and config.relative_enabled
        )

    def _build_ui(self) -> None:
        connection_box = QGroupBox("相机")
        connection_layout = QVBoxLayout(connection_box)
        connection_layout.addWidget(self.device_combo)
        buttons = QHBoxLayout()
        buttons.addWidget(self.refresh_devices_button)
        buttons.addWidget(self.connect_button)
        connection_layout.addLayout(buttons)
        connection_layout.addWidget(self.stream_button)

        config_box = QGroupBox("先验参数与检测")
        config_layout = QFormLayout(config_box)
        path_row = QHBoxLayout()
        path_row.addWidget(self.config_edit, 1)
        path_row.addWidget(self.browse_config_button)
        config_layout.addRow("配置文件", path_row)
        config_layout.addRow("检测后端", self.backend_combo)
        config_layout.addRow("AprilTag family", self.family_combo)
        config_layout.addRow("Tag 外边长", self.tag_size_spin)
        config_layout.addRow("坐标轴长度", self.axis_spin)
        ref_row = QHBoxLayout()
        ref_row.addWidget(self.relative_check)
        ref_row.addWidget(self.reference_check)
        ref_row.addWidget(QLabel("ID"))
        ref_row.addWidget(self.reference_spin)
        config_layout.addRow("相对位姿", ref_row)
        draw_row = QHBoxLayout()
        draw_row.addWidget(self.draw_axis_check)
        draw_row.addWidget(self.draw_corners_check)
        draw_row.addWidget(self.show_text_check)
        draw_row.addWidget(self.show_error_check)
        config_layout.addRow("叠加显示", draw_row)
        config_layout.addRow(self.reload_config_button)

        info_box = QGroupBox("实时位姿（相机坐标系：X右/Y下/Z前）")
        info_layout = QVBoxLayout(info_box)
        info_layout.addWidget(self.info_text)

        basic_page = QWidget()
        basic_layout = QVBoxLayout(basic_page)
        basic_layout.addWidget(connection_box)
        basic_layout.addWidget(config_box)
        basic_layout.addWidget(info_box, 1)
        basic_layout.addWidget(self.save_button)

        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(basic_page, "AprilTag")
        self.right_tabs.setMinimumWidth(500)
        self.right_tabs.setMaximumWidth(660)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video, 1)
        main_layout.addWidget(self.right_tabs)
        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("未连接")

    def _connect_signals(self) -> None:
        self.refresh_devices_button.clicked.connect(self.refresh_devices)
        self.connect_button.clicked.connect(self._toggle_connection)
        self.stream_button.clicked.connect(self._toggle_stream)
        self.save_button.clicked.connect(self.save_current_frame)
        self.browse_config_button.clicked.connect(self.browse_config)
        self.reload_config_button.clicked.connect(self.reload_config)
        self.reference_check.toggled.connect(self._reference_toggled)
        for widget in (
            self.family_combo,
            self.tag_size_spin,
            self.axis_spin,
            self.reference_spin,
            self.relative_check,
            self.draw_axis_check,
            self.draw_corners_check,
            self.show_text_check,
            self.show_error_check,
        ):
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._settings_changed)
            elif isinstance(widget, QSpinBox | QDoubleSpinBox):
                widget.valueChanged.connect(self._settings_changed)
            else:
                widget.toggled.connect(self._settings_changed)

    def _load_optional_feature_panel(self) -> None:
        try:
            from camera_features import CameraFeaturePanel

            self.feature_panel = CameraFeaturePanel()
            self.feature_panel.before_change = self._before_feature_change
            self.feature_panel.after_change = self._after_feature_change
            self.feature_panel.status_changed.connect(self._feature_status)
            self.feature_panel.feature_changed.connect(self._feature_changed)
            self.right_tabs.addTab(self.feature_panel, "相机参数")
        except (ImportError, RuntimeError, AttributeError) as exc:
            self.feature_panel = None
            self.statusBar().showMessage(f"相机参数页不可用：{exc}")

    def _feature_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _feature_changed(self, spec: Any) -> None:
        """Warn when a live GenICam write invalidates the loaded calibration."""

        try:
            from camera_features import is_structural_feature
        except ImportError:
            return
        if is_structural_feature(str(getattr(spec, "name", ""))):
            self.statusBar().showMessage(
                "成像几何已改变；当前内参只对应原标定分辨率/ROI，建议重新标定或加载匹配内参"
            )

    def _before_feature_change(self, spec: Any) -> bool:
        """Pause acquisition around geometry changes and confirm risky nodes."""

        try:
            from camera_features import is_dangerous_feature, is_structural_feature
        except ImportError:
            is_dangerous_feature = lambda _name: False
            is_structural_feature = lambda _name: False
        name = str(getattr(spec, "name", ""))
        if is_dangerous_feature(name):
            answer = QMessageBox.warning(
                self,
                "确认相机命令",
                f"{name} 可能复位设备、改变持久化配置或中断取流。\n确定执行吗？",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return False
        if is_structural_feature(name) and self.capture_thread is not None:
            self._feature_stream_was_running = self.capture_thread.isRunning()
            self._feature_stream_restart_allowed = not is_dangerous_feature(name)
            if self._feature_stream_was_running:
                self.stop_stream()
        else:
            self._feature_stream_was_running = False
            self._feature_stream_restart_allowed = False
        return True

    def _after_feature_change(self, _spec: Any) -> None:
        if self._feature_stream_was_running and self._feature_stream_restart_allowed:
            self.start_stream()
        self._feature_stream_was_running = False
        self._feature_stream_restart_allowed = False
        if self.feature_panel is not None:
            QTimer.singleShot(0, self.feature_panel.refresh_features)

    def _set_connected_state(self, connected: bool) -> None:
        self.connect_button.setText("断开" if connected else "连接")
        self.device_combo.setEnabled(not connected)
        self.refresh_devices_button.setEnabled(not connected)
        self.stream_button.setEnabled(connected)
        self.save_button.setEnabled(connected and self.latest_frame is not None)
        if not connected:
            self.stream_button.setText("开始取流")
        if self.feature_panel is not None and self.controller is not None:
            self.feature_panel.bind_controller(self.controller if connected else None)

    def _import_camera_modules(self) -> tuple[Any, Any]:
        if self.sdk is None or self.controller is None:
            import hikrobot_camera as sdk
            from hikrobot_gui import CameraController

            self.sdk = sdk
            self.controller = CameraController()
        return self.sdk, self.controller

    def refresh_devices(self) -> None:
        if self.controller is not None and self.controller.device_open:
            return
        try:
            sdk, _controller = self._import_camera_modules()
            self.device_list, self.devices = sdk.enumerate_devices()
            self.device_combo.clear()
            for index, device in enumerate(self.devices):
                transport, model, serial = sdk.device_identity(device)
                self.device_combo.addItem(
                    f"[{index}] {model} ({serial}) - {transport}", index
                )
            if self.serial_hint:
                matched = False
                for index, device in enumerate(self.devices):
                    if sdk.device_identity(device)[2] == self.serial_hint:
                        self.device_combo.setCurrentIndex(index)
                        matched = True
                        break
                if not matched:
                    self.device_combo.setCurrentIndex(-1)
                    self.statusBar().showMessage(
                        f"发现 {len(self.devices)} 台相机，但未找到序列号 {self.serial_hint}"
                    )
                    return
            elif self.devices:
                self.device_combo.setCurrentIndex(
                    min(max(self.device_index_hint, 0), len(self.devices) - 1)
                )
            self.statusBar().showMessage(f"发现 {len(self.devices)} 台相机")
        except Exception as exc:
            self._show_error("刷新设备失败", exc)

    def _toggle_connection(self) -> None:
        if self.controller is not None and self.controller.device_open:
            self.disconnect_camera()
        else:
            self.connect_camera()

    def connect_camera(self) -> None:
        if self.controller is None:
            try:
                self._import_camera_modules()
            except Exception as exc:
                self._show_error("加载相机 SDK 失败", exc)
                return
        index = self.device_combo.currentData()
        if index is None or not self.devices:
            self._show_error("连接失败", RuntimeError("没有可连接的相机"))
            return
        try:
            self.controller.open_camera(self.devices[int(index)])
            self._set_connected_state(True)
            self.start_stream()
            self.statusBar().showMessage("相机已连接")
        except Exception as exc:
            if self.controller is not None:
                self.controller.close()
            self._set_connected_state(False)
            self._show_error("连接相机失败", exc)

    def disconnect_camera(self) -> None:
        self.stop_stream()
        if self.controller is not None:
            self.controller.close()
        if self.feature_panel is not None:
            self.feature_panel.bind_controller(None)
        self.video.clear()
        self.latest_frame = None
        self.latest_image = QImage()
        self.save_button.setEnabled(False)
        self._set_connected_state(False)
        self.statusBar().showMessage("已断开")

    def _toggle_stream(self) -> None:
        if self.capture_thread and self.capture_thread.isRunning():
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self) -> None:
        if self.controller is None or not self.controller.device_open:
            return
        if self.capture_thread and self.capture_thread.isRunning():
            return
        try:
            self.controller.start_grabbing()
            self.capture_thread = PoseCaptureThread(
                self.controller,
                self.estimator,
                self.camera_matrix,
                self.distortion,
                self.source_size,
                self.pose_config,
                self.display_buffer,
            )
            self.capture_thread.statistics.connect(self._display_statistics)
            self.capture_thread.failed.connect(self._capture_failed)
            self.capture_thread.start()
            self.display_timer.start()
            self.stream_button.setText("停止取流")
            self.statusBar().showMessage(
                f"正在取流并识别 AprilTag（{self.estimator.backend_name}）…"
            )
            if self.feature_panel is not None:
                QTimer.singleShot(0, self.feature_panel.refresh_features)
        except Exception as exc:
            if self.controller is not None and self.controller.grabbing:
                try:
                    self.controller.stop_grabbing()
                except Exception:
                    pass
            self._show_error("开始取流失败", exc)

    def stop_stream(self) -> None:
        self.display_timer.stop()
        thread = self.capture_thread
        if thread and thread.isRunning():
            thread.request_stop()
            if not thread.wait(2000):
                thread.wait(2000)
        self.capture_thread = None
        self.display_buffer.clear()
        if self.controller is not None and self.controller.grabbing:
            try:
                self.controller.stop_grabbing()
            except Exception as exc:
                self._show_error("停止取流失败", exc)
        self.stream_button.setText("开始取流")
        if self.feature_panel is not None and self.controller is not None:
            QTimer.singleShot(0, self.feature_panel.refresh_features)

    def _drain_display_frame(self) -> None:
        packet = self.display_buffer.take()
        if packet is not None:
            self._display_frame(packet)

    def _display_frame(self, packet: PoseFrame) -> None:
        self.latest_frame = packet
        rgb = np.ascontiguousarray(packet.rgb)
        height, width = rgb.shape[:2]
        self.latest_image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        ).copy()
        self.video.set_image(self.latest_image)
        self.save_button.setEnabled(True)
        # Text layout is substantially more expensive than replacing the
        # image.  Keep the overlay at the display rate, but refresh the
        # detailed pose panel at 10 Hz so it cannot block the Qt event loop.
        now = time.monotonic()
        if now - self._last_info_update < 0.1:
            return
        text = self._pose_text(packet) if packet.poses else "未检测到 AprilTag"
        if text != self._last_info_text:
            self.info_text.setPlainText(text)
            self._last_info_text = text
        self._last_info_update = now

    def _pose_text(self, packet: PoseFrame) -> str:
        lines = [
            f"frame={packet.frame_number}  size={packet.image_size[0]}×{packet.image_size[1]}  "
            f"detect={packet.elapsed_ms:.1f} ms",
            f"detected={len(packet.poses)}  family={self.pose_config.family}  "
            f"backend={self.estimator.backend_name}",
        ]
        if not self.pose_config.relative_enabled:
            lines.append("relative pose=disabled（配置 relative.enabled=false）")
        elif self.pose_config.reference_tag_id is not None:
            reference_present = any(
                pose.tag_id == self.pose_config.reference_tag_id
                for pose in packet.poses
            )
            if packet.relatives:
                lines.append(
                    f"reference=ID {self.pose_config.reference_tag_id}  "
                    "T_reference_tag = inv(T_camera_reference) · T_camera_tag"
                )
            elif reference_present:
                lines.append(
                    f"reference=ID {self.pose_config.reference_tag_id}（已检测，当前无其他目标）"
                )
            else:
                lines.append(
                    f"reference=ID {self.pose_config.reference_tag_id}（当前帧未找到）"
                )
        elif packet.relatives:
            auto_reference = next(iter(packet.relatives.values())).reference_id
            lines.append(
                f"reference=自动选择 ID {auto_reference}（当前帧最小 ID）"
            )
        for pose in packet.poses:
            lines.append("")
            lines.append(f"Tag {pose.tag_id}  reproj={pose.reprojection_error_px:.3f} px")
            lines.append("  t_camera: " + _format_xyz(pose.translation_mm))
            lines.append(
                "  rvec_camera="
                + _format_vector(pose.rvec.reshape(3), " rad")
            )
            lines.append("  rpy_camera=" + _format_vector(pose.euler_xyz_deg, " deg"))
            relative = packet.relatives.get(pose.tag_id)
            if relative is not None:
                lines.append(
                    "  t_relative: " + _format_xyz(relative.translation_mm)
                )
                lines.append(
                    "  rvec_relative=" + _format_vector(relative.rvec, " rad")
                )
                lines.append(
                    "  rpy_relative=" + _format_vector(relative.euler_xyz_deg, " deg")
                )
        return "\n".join(lines)

    def _display_statistics(
        self, fps: float, width: int, height: int, count: int
    ) -> None:
        self.statusBar().showMessage(
            f"取流正常 | {width}×{height} | {fps:.1f} fps | 检测到 {count} 个 Tag"
        )

    def _capture_failed(self, message: str) -> None:
        self.stop_stream()
        self._show_error("视频流/识别错误", RuntimeError(message))

    def _reference_toggled(self, enabled: bool) -> None:
        self.reference_spin.setEnabled(enabled and self.relative_check.isChecked())
        self._settings_changed()

    def _settings_changed(self, *_args: Any) -> None:
        if self.loading_ui:
            return
        old_backend = self.pose_config.backend
        old_family = self.pose_config.family
        old_tag_size_mm = self.pose_config.tag_size_mm
        self.pose_config.backend = str(self.backend_combo.currentData() or "auto")
        self.pose_config.family = self.family_combo.currentText().strip()
        self.pose_config.tag_size_mm = float(self.tag_size_spin.value())
        self.pose_config.axis_length_mm = float(self.axis_spin.value())
        self.pose_config.reference_tag_id = (
            int(self.reference_spin.value()) if self.reference_check.isChecked() else None
        )
        self.pose_config.relative_enabled = self.relative_check.isChecked()
        self.reference_spin.setEnabled(
            self.reference_check.isChecked() and self.pose_config.relative_enabled
        )
        self.pose_config.draw_axis = self.draw_axis_check.isChecked()
        self.pose_config.draw_corners = self.draw_corners_check.isChecked()
        self.pose_config.show_text = self.show_text_check.isChecked()
        self.pose_config.show_reprojection_error = self.show_error_check.isChecked()
        try:
            # Display, reference and relative-pose edits do not affect the
            # detector.  Keeping the existing Isaac ROS bridge avoids
            # restarting a CUDA component for every checkbox click.
            if (
                self.pose_config.backend == old_backend
                and self.pose_config.family == old_family
                and self.pose_config.tag_size_mm == old_tag_size_mm
            ):
                self.estimator.update_config(self.pose_config)
                if self.capture_thread is not None:
                    self.capture_thread.update_settings(
                        self.estimator,
                        self.camera_matrix,
                        self.distortion,
                        self.source_size,
                        self.pose_config,
                    )
                self.statusBar().showMessage("显示/相对位姿参数已应用（未写回配置文件）")
                return
            new_estimator = PoseEstimator(
                self.camera_matrix,
                self.distortion,
                self.pose_config,
                self.source_size,
            )
            old_estimator = self.estimator
            self.estimator = new_estimator
            old_estimator.close()
            if self.capture_thread is not None:
                self.capture_thread.update_settings(
                    self.estimator,
                    self.camera_matrix,
                    self.distortion,
                    self.source_size,
                    self.pose_config,
                )
            self.statusBar().showMessage("显示/检测参数已应用（未写回配置文件）")
        except Exception as exc:
            self._show_error("应用 AprilTag 参数失败", exc)

    def browse_config(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 AprilTag 配置",
            str(self.config_path.parent),
            "YAML/JSON (*.yaml *.yml *.json)",
        )
        if not filename:
            return
        try:
            self.config_path = Path(filename).resolve()
            self.pose_config = load_pose_config(self.config_path)
            if self.serial_override is None:
                self.serial_hint = self.pose_config.camera_serial or None
            self.config_edit.setText(str(self.config_path))
            self.camera_matrix, self.distortion, self.source_size = load_intrinsics(
                self.pose_config.intrinsics_file
            )
            new_estimator = PoseEstimator(
                self.camera_matrix,
                self.distortion,
                self.pose_config,
                self.source_size,
            )
            old_estimator = self.estimator
            self.estimator = new_estimator
            old_estimator.close()
            if self.capture_thread is not None:
                self.capture_thread.update_settings(
                    self.estimator,
                    self.camera_matrix,
                    self.distortion,
                    self.source_size,
                    self.pose_config,
                )
            self._load_config_into_widgets()
            self.statusBar().showMessage(f"已加载配置：{self.config_path}")
        except Exception as exc:
            self._show_error("加载配置失败", exc)

    def reload_config(self) -> None:
        try:
            loaded = load_pose_config(self.config_path)
            matrix, distortion, source_size = load_intrinsics(loaded.intrinsics_file)
            self.pose_config = loaded
            if self.serial_override is None:
                self.serial_hint = loaded.camera_serial or None
            self.camera_matrix = matrix
            self.distortion = distortion
            self.source_size = source_size
            new_estimator = PoseEstimator(
                matrix, distortion, loaded, source_size
            )
            old_estimator = self.estimator
            self.estimator = new_estimator
            old_estimator.close()
            if self.capture_thread is not None:
                self.capture_thread.update_settings(
                    self.estimator,
                    matrix,
                    distortion,
                    source_size,
                    loaded,
                )
            self._load_config_into_widgets()
            self.statusBar().showMessage("先验参数与内参已重新加载")
        except Exception as exc:
            self._show_error("重新加载配置失败", exc)

    def save_current_frame(self) -> None:
        if self.latest_image.isNull():
            return
        capture_dir = self.config_path.parent / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        default_name = capture_dir / (
            f"apriltag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存 AprilTag 叠加帧", str(default_name), "PNG image (*.png)"
        )
        if filename and not self.latest_image.save(filename, "PNG"):
            self._show_error("保存失败", RuntimeError(f"无法写入 {filename}"))
        elif filename:
            self.statusBar().showMessage(f"已保存：{filename}")

    def _show_error(self, title: str, error: Exception) -> None:
        self.statusBar().showMessage(str(error))
        QMessageBox.critical(self, title, str(error))

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.disconnect_camera()
        self.estimator.close()
        event.accept()


def build_argument_parser(
    *,
    description: str | None = None,
    default_config_path: str | Path | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument(
        "--config",
        default=None,
        help=(f"YAML 配置（默认 {default_config_path}）" if default_config_path
              else "AprilTag YAML 配置（默认优先 config/apriltag_pose.yaml）"),
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--serial", default=None, help="按相机序列号选择设备")
    parser.add_argument(
        "--no-autoconnect", action="store_true", help="启动后不自动连接相机"
    )
    return parser


def _default_config_path() -> Path:
    root = Path(__file__).resolve().parent
    for candidate in (root / "config" / "apriltag_pose.yaml", root / "apriltag_pose_config.yaml"):
        if candidate.exists():
            return candidate
    return root / "config" / "apriltag_pose.yaml"


def main(
    argv: Sequence[str] | None = None,
    *,
    window_class: type[MainWindow] = MainWindow,
    default_config_path: str | Path | None = None,
    description: str | None = None,
) -> int:
    args = build_argument_parser(
        description=description, default_config_path=default_config_path
    ).parse_args(argv)
    config_path = (
        Path(args.config).expanduser() if args.config
        else Path(default_config_path) if default_config_path is not None
        else _default_config_path()
    )
    initialized = False
    window: MainWindow | None = None
    app: QApplication | None = None
    old_signal_handlers: dict[int, Any] = {}
    try:
        # Loading PyQt/config does not require the SDK; initialize it only for
        # the camera connection and always finalize it exactly once.
        import hikrobot_camera as sdk

        sdk.require_ok("MV_CC_Initialize", sdk.MvCamera.MV_CC_Initialize())
        initialized = True
        app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
        app.setApplicationName("HIKROBOT AprilTag Pose GUI")
        # A Python ``KeyboardInterrupt`` can arrive in the middle of Qt's
        # paint/event callback and tear down the MVS worker unsafely.  Convert
        # Ctrl-C/TERM into a normal Qt quit so closeEvent stops the worker.
        def _quit_from_signal(_signum: int, _frame: Any) -> None:
            if app is not None:
                app.quit()

        for _signal_number in (signal.SIGINT, signal.SIGTERM):
            old_signal_handlers[_signal_number] = signal.getsignal(_signal_number)
            signal.signal(_signal_number, _quit_from_signal)
        window = window_class(
            config_path,
            device_index=args.device_index,
            serial=args.serial,
            auto_connect=not args.no_autoconnect,
        )
        window.show()
        return app.exec_()
    except Exception as exc:
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "启动失败", str(exc))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        for _signal_number, _handler in old_signal_handlers.items():
            try:
                signal.signal(_signal_number, _handler)
            except (ValueError, OSError):
                pass
        # ``KeyboardInterrupt``/window-manager termination can leave the Qt
        # event loop without going through ``closeEvent``.  Stop the capture
        # worker before closing the SDK handle; closing the handle first races
        # ``read_rgb_frame`` and can crash the vendor wrapper.
        if window is not None:
            try:
                window.disconnect_camera()
            except Exception:
                if window.controller is not None:
                    try:
                        window.controller.close()
                    except Exception:
                        pass
            try:
                # ``QApplication.quit()`` (used by SIGINT/SIGTERM handlers)
                # does not guarantee that closeEvent runs, so explicitly
                # close the Isaac ROS process group during final teardown.
                window.estimator.close()
            except Exception:
                pass
        if initialized:
            try:
                import hikrobot_camera as sdk

                sdk.MvCamera.MV_CC_Finalize()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
