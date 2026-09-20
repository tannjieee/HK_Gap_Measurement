"""Chessboard detection, camera calibration, and result serialization."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class BoardSpec:
    """Physical chessboard description expressed as square counts."""

    squares_x: int = 12
    squares_y: int = 9
    square_size_mm: float = 15.0

    def __post_init__(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("A chessboard must contain at least 3x3 squares")
        if not math.isfinite(self.square_size_mm) or self.square_size_mm <= 0:
            raise ValueError("square_size_mm must be a positive finite number")

    @property
    def inner_corners_x(self) -> int:
        return self.squares_x - 1

    @property
    def inner_corners_y(self) -> int:
        return self.squares_y - 1

    @property
    def pattern_size(self) -> tuple[int, int]:
        """OpenCV pattern size in (columns, rows) order."""

        return self.inner_corners_x, self.inner_corners_y

    @property
    def point_count(self) -> int:
        return self.inner_corners_x * self.inner_corners_y

    @property
    def square_size_m(self) -> float:
        return self.square_size_mm / 1000.0

    def object_points(self) -> np.ndarray:
        """Return row-major planar object points in metres."""

        columns = np.arange(self.inner_corners_x, dtype=np.float32)
        rows = np.arange(self.inner_corners_y, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(columns, rows)
        points = np.zeros((self.point_count, 3), dtype=np.float32)
        points[:, :2] = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        points[:, :2] *= self.square_size_m
        return points


@dataclass
class ChessboardDetection:
    found: bool
    corners: np.ndarray | None = None
    sharpness: float = 0.0
    coverage: float = 0.0
    minimum_spacing: float = 0.0
    descriptor: np.ndarray | None = None


@dataclass
class CalibrationSample:
    corners: np.ndarray
    image_size: tuple[int, int]
    descriptor: np.ndarray
    sharpness: float
    coverage: float
    gray_image: np.ndarray | None = None
    frame_number: int | None = None
    source_name: str | None = None


@dataclass
class CalibrationResult:
    rms_error: float
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: list[np.ndarray]
    translation_vectors: list[np.ndarray]
    intrinsic_std_deviations: np.ndarray
    per_view_errors: np.ndarray
    image_size: tuple[int, int]
    used_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    rejected_view_errors: dict[int, float]

    @property
    def mean_view_error(self) -> float:
        return float(np.mean(self.per_view_errors))

    @property
    def median_view_error(self) -> float:
        return float(np.median(self.per_view_errors))

    @property
    def maximum_view_error(self) -> float:
        return float(np.max(self.per_view_errors))


def _normalize_grayscale(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("Chessboard detection expects one grayscale image")
    if gray.dtype == np.uint8:
        return np.ascontiguousarray(gray)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return np.ascontiguousarray(normalized, dtype=np.uint8)


def _corner_metrics(
    gray: np.ndarray, corners: np.ndarray, board: BoardSpec
) -> tuple[float, float, float, np.ndarray]:
    height, width = gray.shape
    points = corners.reshape(-1, 2)
    hull = cv2.convexHull(points.astype(np.float32))
    coverage = float(cv2.contourArea(hull) / (width * height))

    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    padding = max(4, round(min(width, height) * 0.01))
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width, x1 + padding + 1)
    y1 = min(height, y1 + padding + 1)
    region = gray[y0:y1, x0:x1]
    sharpness = (
        float(cv2.Laplacian(region, cv2.CV_64F).var()) if region.size else 0.0
    )

    grid = points.reshape(board.inner_corners_y, board.inner_corners_x, 2)
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    minimum_spacing = float(min(horizontal.min(), vertical.min()))

    center = points.mean(axis=0) / np.array([width, height], dtype=np.float32)
    direction = grid[0, -1] - grid[0, 0]
    angle = math.atan2(float(direction[1]), float(direction[0]))
    top = float(np.linalg.norm(grid[0, -1] - grid[0, 0]))
    bottom = float(np.linalg.norm(grid[-1, -1] - grid[-1, 0]))
    left = float(np.linalg.norm(grid[-1, 0] - grid[0, 0]))
    right = float(np.linalg.norm(grid[-1, -1] - grid[0, -1]))
    epsilon = 1e-6
    descriptor = np.array(
        [
            center[0],
            center[1],
            math.sqrt(max(coverage, 0.0)),
            math.sin(angle),
            math.cos(angle),
            math.log((top + epsilon) / (bottom + epsilon)),
            math.log((left + epsilon) / (right + epsilon)),
        ],
        dtype=np.float32,
    )
    return sharpness, coverage, minimum_spacing, descriptor


def detect_chessboard(
    gray: np.ndarray,
    board: BoardSpec,
    *,
    maximum_detection_dimension: int = 1280,
    exhaustive: bool = True,
) -> ChessboardDetection:
    """Detect and characterize every inner corner of a chessboard."""

    gray = _normalize_grayscale(gray)
    height, width = gray.shape
    scale = min(1.0, maximum_detection_dimension / max(width, height))
    detection_image = gray
    if scale < 1.0:
        detection_image = cv2.resize(
            gray,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    classic_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(
        detection_image, board.pattern_size, flags=classic_flags
    )
    used_sb = False
    if not found and exhaustive and hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        found, corners = cv2.findChessboardCornersSB(
            detection_image, board.pattern_size, flags=sb_flags
        )
        used_sb = bool(found)

    if not found or corners is None or len(corners) != board.point_count:
        return ChessboardDetection(found=False)

    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    if scale < 1.0:
        corners /= scale
    if not used_sb:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not np.isfinite(corners).all():
        return ChessboardDetection(found=False)
    sharpness, coverage, spacing, descriptor = _corner_metrics(
        gray, corners, board
    )
    return ChessboardDetection(
        found=True,
        corners=corners,
        sharpness=sharpness,
        coverage=coverage,
        minimum_spacing=spacing,
        descriptor=descriptor,
    )


def descriptor_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return a normalized pose-distance score for automatic sample de-duplication."""

    center_distance = float(np.linalg.norm(first[:2] - second[:2])) / 0.08
    scale_distance = abs(float(first[2] - second[2])) / 0.04
    cosine = float(np.clip(first[3] * second[3] + first[4] * second[4], -1, 1))
    angle_distance = math.acos(abs(cosine)) / math.radians(10.0)
    perspective_distance = float(np.linalg.norm(first[5:] - second[5:])) / 0.10
    return math.sqrt(
        center_distance**2
        + scale_distance**2
        + angle_distance**2
        + perspective_distance**2
    )


def novelty_score(
    descriptor: np.ndarray, existing_descriptors: Sequence[np.ndarray]
) -> float:
    if not existing_descriptors:
        return math.inf
    return min(descriptor_distance(descriptor, item) for item in existing_descriptors)


def make_sample(
    detection: ChessboardDetection,
    image_size: tuple[int, int],
    *,
    gray_image: np.ndarray | None = None,
    frame_number: int | None = None,
    source_name: str | None = None,
) -> CalibrationSample:
    if (
        not detection.found
        or detection.corners is None
        or detection.descriptor is None
    ):
        raise ValueError("Cannot create a calibration sample without detected corners")
    return CalibrationSample(
        corners=detection.corners.copy(),
        image_size=image_size,
        descriptor=detection.descriptor.copy(),
        sharpness=detection.sharpness,
        coverage=detection.coverage,
        gray_image=gray_image.copy() if gray_image is not None else None,
        frame_number=frame_number,
        source_name=source_name,
    )


def _run_calibration(
    samples: Sequence[CalibrationSample],
    board: BoardSpec,
    indices: Sequence[int],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    object_template = board.object_points()
    object_points = [object_template.copy() for _ in indices]
    image_points = [samples[index].corners.astype(np.float32) for index in indices]
    image_size = samples[indices[0]].image_size
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-9,
    )
    (
        rms,
        camera_matrix,
        distortion,
        rotation_vectors,
        translation_vectors,
        intrinsic_std,
        _extrinsic_std,
        _opencv_view_errors,
    ) = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=0,
        criteria=criteria,
    )

    per_view_errors = []
    for object_view, image_view, rotation, translation in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(
            object_view, rotation, translation, camera_matrix, distortion
        )
        residual = image_view.reshape(-1, 2) - projected.reshape(-1, 2)
        per_view_errors.append(
            math.sqrt(float(np.mean(np.sum(residual * residual, axis=1))))
        )

    outputs = (camera_matrix, distortion, intrinsic_std, per_view_errors)
    if not all(np.isfinite(np.asarray(item)).all() for item in outputs):
        raise RuntimeError("OpenCV returned non-finite calibration parameters")
    return (
        float(rms),
        camera_matrix,
        distortion,
        list(rotation_vectors),
        list(translation_vectors),
        np.asarray(intrinsic_std).reshape(-1),
        np.asarray(per_view_errors, dtype=np.float64),
    )


def calibrate_samples(
    samples: Sequence[CalibrationSample],
    board: BoardSpec,
    *,
    minimum_samples: int = 12,
    reject_outliers: bool = True,
) -> CalibrationResult:
    """Calibrate a pinhole/plumb-bob model and optionally remove gross bad views."""

    if minimum_samples < 3:
        raise ValueError("minimum_samples must be at least 3")
    if len(samples) < minimum_samples:
        raise ValueError(
            f"Need at least {minimum_samples} samples; received {len(samples)}"
        )
    image_size = samples[0].image_size
    for index, sample in enumerate(samples):
        if sample.image_size != image_size:
            raise ValueError(
                f"Sample {index + 1} has size {sample.image_size}, "
                f"expected {image_size}"
            )
        if sample.corners.reshape(-1, 2).shape[0] != board.point_count:
            raise ValueError(
                f"Sample {index + 1} does not contain {board.point_count} corners"
            )
        if not np.isfinite(sample.corners).all():
            raise ValueError(f"Sample {index + 1} contains non-finite corners")

    active = list(range(len(samples)))
    rejected: list[int] = []
    rejected_errors: dict[int, float] = {}
    maximum_rejections = math.floor(len(samples) * 0.20)

    while True:
        run = _run_calibration(samples, board, active)
        errors = run[-1]
        if not reject_outliers or len(rejected) >= maximum_rejections:
            break
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        threshold = max(1.0, median + 3.0 * 1.4826 * mad)
        worst_position = int(np.argmax(errors))
        worst_error = float(errors[worst_position])
        if worst_error <= threshold or len(active) - 1 < minimum_samples:
            break
        original_index = active.pop(worst_position)
        rejected.append(original_index)
        rejected_errors[original_index] = worst_error

    (
        rms,
        camera_matrix,
        distortion,
        rotations,
        translations,
        intrinsic_std,
        view_errors,
    ) = run
    return CalibrationResult(
        rms_error=rms,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion.reshape(-1),
        rotation_vectors=rotations,
        translation_vectors=translations,
        intrinsic_std_deviations=intrinsic_std,
        per_view_errors=view_errors,
        image_size=image_size,
        used_indices=tuple(active),
        rejected_indices=tuple(rejected),
        rejected_view_errors=rejected_errors,
    )


def calibration_quality(result: CalibrationResult) -> tuple[str, str]:
    if result.rms_error <= 0.5:
        return "good", "重投影误差良好"
    if result.rms_error <= 1.0:
        return "review", "误差可用，但建议检查高误差视图和边缘覆盖"
    return "poor", "误差偏高，建议重新采集更多清晰且姿态多样的图像"


def _matrix_block(matrix: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(matrix)
    return {
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "data": [float(value) for value in matrix.reshape(-1)],
    }


def result_document(
    result: CalibrationResult,
    board: BoardSpec,
    samples: Sequence[CalibrationSample],
    *,
    camera_name: str = "hikrobot_camera",
) -> dict[str, object]:
    width, height = result.image_size
    matrix = result.camera_matrix
    distortion = result.distortion_coefficients
    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:, :3] = matrix
    quality, quality_message = calibration_quality(result)
    fov_x = math.degrees(2.0 * math.atan(width / (2.0 * matrix[0, 0])))
    fov_y = math.degrees(2.0 * math.atan(height / (2.0 * matrix[1, 1])))

    distortion_names = ["k1", "k2", "p1", "p2", "k3"]
    distortion_map = {
        name: float(value)
        for name, value in zip(distortion_names, distortion, strict=False)
    }
    views: list[dict[str, object]] = []
    for output_index, sample_index in enumerate(result.used_indices):
        sample = samples[sample_index]
        views.append(
            {
                "sample_index": sample_index + 1,
                "source": sample.source_name,
                "frame_number": sample.frame_number,
                "sharpness_laplacian_variance": float(sample.sharpness),
                "board_coverage_ratio": float(sample.coverage),
                "reprojection_error_px": float(
                    result.per_view_errors[output_index]
                ),
                "rotation_vector": [
                    float(value)
                    for value in result.rotation_vectors[output_index].reshape(-1)
                ],
                "translation_vector_m": [
                    float(value)
                    for value in result.translation_vectors[output_index].reshape(-1)
                ],
            }
        )

    rejected_views = [
        {
            "sample_index": index + 1,
            "source": samples[index].source_name,
            "reprojection_error_when_rejected_px": float(
                result.rejected_view_errors[index]
            ),
        }
        for index in result.rejected_indices
    ]
    return {
        "schema_version": 1,
        "calibration_time_utc": datetime.now(timezone.utc).isoformat(),
        "camera_name": camera_name,
        "image_width": width,
        "image_height": height,
        "distortion_model": "plumb_bob",
        "camera_matrix": _matrix_block(matrix),
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(distortion.size),
            "data": [float(value) for value in distortion],
        },
        "rectification_matrix": _matrix_block(np.eye(3, dtype=np.float64)),
        "projection_matrix": _matrix_block(projection),
        "intrinsics": {
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
            "skew": float(matrix[0, 1]),
        },
        "distortion": distortion_map,
        "field_of_view_degrees": {"horizontal": fov_x, "vertical": fov_y},
        "board": {
            "squares_x": board.squares_x,
            "squares_y": board.squares_y,
            "inner_corners_x": board.inner_corners_x,
            "inner_corners_y": board.inner_corners_y,
            "point_count": board.point_count,
            "square_size_mm": board.square_size_mm,
            "square_size_m": board.square_size_m,
        },
        "calibration": {
            "method": "cv2.calibrateCameraExtended",
            "opencv_version": cv2.__version__,
            "input_sample_count": len(samples),
            "used_sample_count": len(result.used_indices),
            "rejected_sample_count": len(result.rejected_indices),
            "rms_reprojection_error_px": result.rms_error,
            "mean_view_error_px": result.mean_view_error,
            "median_view_error_px": result.median_view_error,
            "maximum_view_error_px": result.maximum_view_error,
            "quality": quality,
            "quality_message": quality_message,
            "intrinsic_std_deviations": [
                float(value) for value in result.intrinsic_std_deviations
            ],
        },
        "views": views,
        "rejected_views": rejected_views,
    }


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def save_calibration(
    output_path: Path,
    result: CalibrationResult,
    board: BoardSpec,
    samples: Sequence[CalibrationSample],
    *,
    camera_name: str = "hikrobot_camera",
    save_sample_images: bool = True,
) -> tuple[Path, Path, Path | None]:
    """Write ROS-compatible YAML, a matching JSON report, and accepted images."""

    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() not in {".yaml", ".yml"}:
        output_path = output_path.with_suffix(".yaml")
    json_path = output_path.with_suffix(".json")
    document = result_document(
        result, board, samples, camera_name=camera_name
    )
    yaml_text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    json_text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_text(output_path, yaml_text)
    _atomic_write_text(json_path, json_text)

    sample_directory: Path | None = None
    if save_sample_images and any(sample.gray_image is not None for sample in samples):
        sample_directory = output_path.parent / f"{output_path.stem}_samples"
        sample_directory.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples, start=1):
            if sample.gray_image is None:
                continue
            image_path = sample_directory / f"view_{index:03d}.png"
            if not cv2.imwrite(str(image_path), sample.gray_image):
                raise OSError(f"Unable to save calibration sample {image_path}")
    return output_path, json_path, sample_directory
