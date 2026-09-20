"""AprilTag detection and metric pose estimation.

This module deliberately keeps the detector and the GUI separate.  It accepts
the corner output of any AprilTag detector and computes the pose with OpenCV,
so the optional ``pupil_apriltags`` and ``apriltag`` packages are not required
when OpenCV's built-in AprilTag dictionaries are available (``opencv-contrib``).

The camera pose convention is the usual OpenCV convention::

    X_camera = R_camera_tag @ X_tag + t_camera_tag

Tag coordinates are centred on the tag. Relative to the original solver frame,
new +X = old +Z, new +Y = old -X, and new +Z = old -Y.
Translation is in metres in the unchanged camera frame;
``translation_mm`` is provided for display.  ``relative_pose(a, b)`` returns
the pose of tag *b* expressed in tag *a*'s coordinate frame.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from apriltag_coordinates import remap_tag_rotation

try:  # PyYAML is already used by the calibration tool, but keep this module optional.
    import yaml
except Exception:  # pragma: no cover - exercised only in minimal installations
    yaml = None


DEFAULT_INTRINSICS_BASENAME = "hikrobot_intrinsics_20260905_124817"
DEFAULT_TAG_SIZE_MM = 48.35
DEFAULT_TAG_FAMILY = "tag36h11"


class AprilTagError(RuntimeError):
    """Base class for AprilTag configuration/detection errors."""


class AprilTagBackendUnavailable(AprilTagError):
    """Raised when no detector backend can be imported or used."""


class AprilTagConfigurationError(AprilTagError, ValueError):
    """Raised when camera/tag configuration is malformed."""


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - numpy error wording varies
        raise AprilTagConfigurationError(f"{name} must be numeric") from exc
    if not np.isfinite(array).all():
        raise AprilTagConfigurationError(f"{name} contains non-finite values")
    return array


def _matrix_data(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Read a plain array or OpenCV's ``{rows, cols, data}`` representation."""

    if isinstance(value, Mapping) and "data" in value:
        value = value["data"]
    array = _as_float_array(value, name=name)
    if array.size != int(np.prod(shape)):
        raise AprilTagConfigurationError(
            f"{name} must contain {int(np.prod(shape))} values, got {array.size}"
        )
    return np.ascontiguousarray(array.reshape(shape), dtype=np.float64)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and not _is_missing(mapping[key]):
            return mapping[key]
    return default


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera matrix and distortion coefficients loaded from calibration output."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 5), dtype=np.float64)
    )
    image_size: tuple[int, int] | None = None
    distortion_model: str = "plumb_bob"
    source_path: str | None = None

    def __post_init__(self) -> None:
        matrix = _matrix_data(
            self.camera_matrix, name="camera_matrix", shape=(3, 3)
        )
        distortion = _as_float_array(
            self.distortion_coefficients, name="distortion_coefficients"
        ).reshape(-1)
        if distortion.size == 0:
            distortion = np.zeros(5, dtype=np.float64)
        if distortion.size not in (4, 5, 8, 12, 14):
            raise AprilTagConfigurationError(
                "distortion_coefficients must contain 4, 5, 8, 12 or 14 values"
            )
        if matrix[2, 2] == 0:
            raise AprilTagConfigurationError("camera_matrix[2,2] cannot be zero")
        size = self.image_size
        if size is not None:
            if len(size) != 2 or int(size[0]) <= 0 or int(size[1]) <= 0:
                raise AprilTagConfigurationError("image_size must be (width, height)")
            size = (int(size[0]), int(size[1]))
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion.reshape(1, -1))
        object.__setattr__(self, "image_size", size)

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], *, source_path: str | None = None
    ) -> "CameraIntrinsics":
        """Construct from a calibration JSON/YAML mapping.

        Both the calibration tool's keys (``camera_matrix`` and
        ``distortion_coefficients``) and compact ``intrinsics`` dictionaries
        (``fx/fy/cx/cy``) are accepted.
        """

        camera_section = mapping.get("camera", {})
        if not isinstance(camera_section, Mapping):
            camera_section = {}
        matrix_value = _first(
            mapping,
            "camera_matrix",
            "intrinsic_matrix",
            "K",
            default=None,
        )
        if matrix_value is None:
            matrix_value = _first(camera_section, "camera_matrix", "K", default=None)
        if matrix_value is None:
            intrinsics = _first(mapping, "intrinsics", default=None)
            if intrinsics is None:
                intrinsics = _first(camera_section, "intrinsics", default=None)
            if isinstance(intrinsics, Mapping):
                fx = _first(intrinsics, "fx")
                fy = _first(intrinsics, "fy")
                cx = _first(intrinsics, "cx")
                cy = _first(intrinsics, "cy")
                if None not in (fx, fy, cx, cy):
                    matrix_value = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        if matrix_value is None:
            raise AprilTagConfigurationError("camera_matrix/intrinsics is missing")

        distortion = _first(
            mapping,
            "distortion_coefficients",
            "distortion",
            "dist_coeffs",
            "D",
            default=None,
        )
        if distortion is None:
            distortion = _first(
                camera_section,
                "distortion_coefficients",
                "distortion",
                "D",
                default=[0.0] * 5,
            )
        if isinstance(distortion, Mapping):
            distortion = [
                distortion.get(key, 0.0)
                for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
            ]

        width = _first(
            mapping,
            "image_width",
            "width",
            default=_first(camera_section, "image_width", "width", default=None),
        )
        height = _first(
            mapping,
            "image_height",
            "height",
            default=_first(camera_section, "image_height", "height", default=None),
        )
        if width is None or height is None:
            size = _first(mapping, "image_size", default=None)
            if size is None:
                size = _first(camera_section, "image_size", default=None)
            if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) >= 2:
                width, height = size[:2]
        image_size = None if width is None or height is None else (int(width), int(height))
        model = str(
            _first(mapping, "distortion_model", default="plumb_bob")
        )
        return cls(
            camera_matrix=_matrix_data(matrix_value, name="camera_matrix", shape=(3, 3)),
            distortion_coefficients=distortion,
            image_size=image_size,
            distortion_model=model,
            source_path=source_path,
        )

    def for_image_size(self, image_size: tuple[int, int] | None) -> "CameraIntrinsics":
        """Scale ``K`` to a resized image while retaining the calibrated centre.

        If no calibration image size is known, or the requested size matches
        it, the same values are returned.  This is useful when the live preview
        is intentionally downscaled before detection.
        """

        if image_size is None or self.image_size is None:
            return self
        width, height = int(image_size[0]), int(image_size[1])
        if (width, height) == self.image_size:
            return self
        sx = width / self.image_size[0]
        sy = height / self.image_size[1]
        matrix = self.camera_matrix.copy()
        matrix[0, 0] *= sx
        matrix[0, 2] *= sx
        matrix[1, 1] *= sy
        matrix[1, 2] *= sy
        return CameraIntrinsics(
            matrix,
            self.distortion_coefficients.copy(),
            (width, height),
            self.distortion_model,
            self.source_path,
        )


def _load_mapping(path: os.PathLike[str] | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    elif yaml is not None:
        value = yaml.safe_load(text)
    else:
        raise AprilTagConfigurationError(
            "PyYAML is required to read YAML configuration files"
        )
    if not isinstance(value, Mapping):
        raise AprilTagConfigurationError(f"configuration root must be a mapping: {path}")
    return dict(value)


def _find_default_intrinsics(base_dir: Path | None = None) -> Path | None:
    roots = []
    if base_dir is not None:
        roots.append(base_dir)
    roots.extend(
        [
            Path.cwd() / "calibration_results",
            Path(__file__).resolve().parent / "calibration_results",
        ]
    )
    for root in roots:
        for suffix in (".yaml", ".yml", ".json"):
            candidate = root / f"{DEFAULT_INTRINSICS_BASENAME}{suffix}"
            if candidate.is_file():
                return candidate
    return None


@dataclass(frozen=True)
class AprilTagPoseConfig:
    """All parameters needed by :class:`AprilTagPoseEstimator`."""

    camera: CameraIntrinsics
    tag_size_m: float = DEFAULT_TAG_SIZE_MM / 1000.0
    family: str = DEFAULT_TAG_FAMILY
    backend: str = "auto"
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    nthreads: int = 1
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    min_decision_margin: float | None = None
    max_hamming: int = 0
    allowed_ids: tuple[int, ...] | None = None
    max_reprojection_error_px: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.tag_size_m) or self.tag_size_m <= 0:
            raise AprilTagConfigurationError("tag_size_m must be positive")
        if not math.isfinite(self.quad_decimate) or self.quad_decimate <= 0:
            raise AprilTagConfigurationError("quad_decimate must be positive")
        if self.max_hamming < 0:
            raise AprilTagConfigurationError("max_hamming cannot be negative")
        if self.max_reprojection_error_px is not None and (
            not math.isfinite(self.max_reprojection_error_px)
            or self.max_reprojection_error_px <= 0
        ):
            raise AprilTagConfigurationError(
                "max_reprojection_error_px must be positive when specified"
            )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
        intrinsics_path: os.PathLike[str] | str | None = None,
    ) -> "AprilTagPoseConfig":
        section = mapping.get("apriltag", mapping.get("april_tag", mapping))
        if not isinstance(section, Mapping):
            section = mapping
        tag_section = section.get("tag", mapping.get("tag", {}))
        if not isinstance(tag_section, Mapping):
            tag_section = {}
        size_mm = _first(
            section,
            "tag_size_mm",
            default=_first(tag_section, "size_mm", "tag_size_mm", default=None),
        )
        size_m = _first(
            section,
            "tag_size_m",
            default=_first(tag_section, "size_m", "tag_size_m", default=None),
        )
        if size_m is None:
            size_m = (DEFAULT_TAG_SIZE_MM if size_mm is None else float(size_mm)) / 1000.0
        else:
            size_m = float(size_m)

        family = str(
            _first(
                section,
                "family",
                "tag_family",
                default=_first(
                    tag_section, "family", "tag_family", default=DEFAULT_TAG_FAMILY
                ),
            )
        )
        backend = str(_first(section, "backend", default="auto")).lower()
        detector = section.get("detector", {})
        if not isinstance(detector, Mapping):
            detector = {}
        get_detector = lambda key, default: _first(detector, key, default=_first(section, key, default=default))
        path_value: Any = intrinsics_path
        if path_value is None:
            path_value = _first(
                section,
                "intrinsics_file",
                "camera_intrinsics",
                "calibration_file",
                default=None,
            )
        camera_section = mapping.get("camera", {})
        if path_value is None and isinstance(camera_section, Mapping):
            path_value = _first(
                camera_section,
                "intrinsics_file",
                "calibration_file",
                default=None,
            )
        if path_value is not None:
            path = Path(path_value).expanduser()
            if not path.is_absolute():
                # Configuration files in ``config/`` commonly contain a path
                # rooted at the project (``calibration_results/...``).  Try
                # the config directory first, then cwd/project root so both
                # conventions remain portable.
                candidates = []
                if base_dir is not None:
                    candidates.append(base_dir / path)
                candidates.extend([Path.cwd() / path, Path(__file__).resolve().parent / path])
                path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0] if candidates else path)
            camera_mapping = _load_mapping(path)
            camera = CameraIntrinsics.from_mapping(camera_mapping, source_path=str(path))
        else:
            # A compact config may contain camera_matrix directly.
            try:
                camera = CameraIntrinsics.from_mapping(mapping)
            except AprilTagConfigurationError:
                default_path = _find_default_intrinsics(base_dir)
                if default_path is None:
                    raise
                camera_mapping = _load_mapping(default_path)
                camera = CameraIntrinsics.from_mapping(
                    camera_mapping, source_path=str(default_path)
                )
        margin = get_detector("min_decision_margin", None)
        ids_value = _first(
            section,
            "ids",
            "tag_ids",
            default=_first(tag_section, "ids", "tag_ids", default=None),
        )
        if _is_missing(ids_value):
            allowed_ids = None
        else:
            if isinstance(ids_value, (str, int, np.integer)):
                ids_value = [ids_value]
            allowed_ids = tuple(int(value) for value in ids_value)
            if not allowed_ids:
                allowed_ids = None
        max_error = get_detector("max_reprojection_error_px", None)
        return cls(
            camera=camera,
            tag_size_m=size_m,
            family=family,
            backend=backend,
            quad_decimate=float(get_detector("quad_decimate", 1.0)),
            quad_sigma=float(get_detector("quad_sigma", 0.0)),
            nthreads=int(get_detector("nthreads", 1)),
            refine_edges=bool(get_detector("refine_edges", True)),
            decode_sharpening=float(get_detector("decode_sharpening", 0.25)),
            min_decision_margin=None if margin is None else float(margin),
            max_hamming=int(get_detector("max_hamming", 0)),
            allowed_ids=allowed_ids,
            max_reprojection_error_px=None if max_error is None else float(max_error),
        )

    @classmethod
    def from_file(
        cls,
        path: os.PathLike[str] | str,
        *,
        intrinsics_path: os.PathLike[str] | str | None = None,
    ) -> "AprilTagPoseConfig":
        config_path = Path(path).expanduser()
        return cls.from_mapping(
            _load_mapping(config_path),
            base_dir=config_path.parent,
            intrinsics_path=intrinsics_path,
        )


def load_pose_config(
    path: os.PathLike[str] | str | None = None,
    *,
    intrinsics_path: os.PathLike[str] | str | None = None,
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM,
    family: str = DEFAULT_TAG_FAMILY,
    backend: str = "auto",
) -> AprilTagPoseConfig:
    """Load an AprilTag config, with the project's 20260905 calibration default.

    If *path* is omitted, a compact default config is generated from the
    ``hikrobot_intrinsics_20260905_124817`` calibration file.  Explicit
    ``tag_size_mm``/``family``/``backend`` values override file values only in
    this no-config convenience path.
    """

    if path is not None:
        return AprilTagPoseConfig.from_file(path, intrinsics_path=intrinsics_path)
    chosen = intrinsics_path
    if chosen is None:
        chosen = _find_default_intrinsics()
    if chosen is None:
        raise FileNotFoundError(
            "No AprilTag config or default intrinsics found; pass --config or --intrinsics"
        )
    camera = CameraIntrinsics.from_mapping(
        _load_mapping(chosen), source_path=str(chosen)
    )
    return AprilTagPoseConfig(
        camera=camera,
        tag_size_m=float(tag_size_mm) / 1000.0,
        family=family,
        backend=backend,
    )


def _gray_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 1:
        gray = array[..., 0]
    elif array.ndim == 3 and array.shape[2] >= 3:
        gray = cv2.cvtColor(array[..., :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image must be a grayscale or BGR array")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(gray)


def _canonical_corners(corners: Any) -> np.ndarray:
    """Return corners in screen top-left, top-right, bottom-right, bottom-left order.

    This helper is intentionally *not* used for detector output.  A decoded
    AprilTag detector's first corner identifies the tag's canonical (encoded)
    corner, which is not necessarily the screen-space top-left corner after a
    large in-plane rotation.  Re-sorting by image coordinates would silently
    rotate the tag frame and corrupt 6D orientation.  It remains useful for
    callers that explicitly provide unordered points.
    """

    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("AprilTag corners must be four finite (x,y) points")
    # This diagonal assignment remains stable under normal perspective skew and
    # avoids relying on the corner-order convention of optional detector libs.
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    tl = int(np.argmin(sums))
    br = int(np.argmax(sums))
    tr = int(np.argmax(diffs))
    bl = int(np.argmin(diffs))
    selected = [tl, tr, br, bl]
    if len(set(selected)) != 4:
        # Degenerate/near-axis-aligned ties: angular sort then rotate to TL.
        centre = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
        order = list(np.argsort(angles))
        # np.argsort starts at the left side; rotate by the smallest x+y point.
        start = min(range(4), key=lambda i: sums[order[i]])
        order = order[start:] + order[:start]
        selected = order
        # Ensure clockwise order in image coordinates.
        area2 = sum(
            points[selected[i], 0] * points[selected[(i + 1) % 4], 1]
            - points[selected[(i + 1) % 4], 0] * points[selected[i], 1]
            for i in range(4)
        )
        if area2 < 0:
            selected = [selected[0], selected[3], selected[2], selected[1]]
    return np.ascontiguousarray(points[selected], dtype=np.float64)


def _detector_corners(corners: Any, *, source: str = "detector") -> np.ndarray:
    """Normalize detector corner order without losing the tag's orientation.

    ``pupil_apriltags`` and OpenCV ArUco return the canonical clockwise order
    ``top-left, top-right, bottom-right, bottom-left``.  The C ``apriltag``
    wrapper exposes ``lb-rb-rt-lt`` instead; rotate that sequence to the same
    object-point order.  In either case the input start corner is preserved --
    no screen-coordinate sorting is performed.
    """

    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("AprilTag corners must be four finite (x,y) points")
    if source == "apriltag_lb_rb_rt_lt":
        points = points[[2, 3, 0, 1]]
    return np.ascontiguousarray(points, dtype=np.float64)


def _rotation_to_euler_xyz(rotation: np.ndarray) -> np.ndarray:
    """Return fixed-axis roll, pitch, yaw (degrees) from a rotation matrix."""

    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.hypot(float(r[0, 0]), float(r[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        pitch = math.atan2(float(-r[2, 0]), sy)
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = math.atan2(float(-r[1, 2]), float(r[1, 1]))
        pitch = math.atan2(float(-r[2, 0]), sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw]).astype(np.float64)


def _transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


@dataclass
class TagPose:
    """A detected AprilTag and its metric camera-relative pose."""

    tag_id: int
    family: str
    corners: np.ndarray
    center: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray
    euler_xyz_deg: np.ndarray
    reprojection_error_px: float
    decision_margin: float | None = None
    hamming: int | None = None

    @property
    def translation_m(self) -> np.ndarray:
        return np.asarray(self.tvec, dtype=np.float64).reshape(3)

    @property
    def translation_mm(self) -> np.ndarray:
        return self.translation_m * 1000.0

    @property
    def rpy_deg(self) -> np.ndarray:
        return self.euler_xyz_deg

    @property
    def transform_camera_tag(self) -> np.ndarray:
        return _transform_matrix(self.rotation_matrix, self.translation_m)

    @property
    def pose_valid(self) -> bool:
        return bool(np.isfinite(self.translation_m).all() and self.translation_m[2] > 0)


@dataclass
class RelativePose:
    """Pose of ``target_id`` expressed in ``reference_id`` coordinates."""

    reference_id: int | str
    target_id: int | str
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray
    euler_xyz_deg: np.ndarray

    @property
    def translation_m(self) -> np.ndarray:
        return np.asarray(self.tvec, dtype=np.float64).reshape(3)

    @property
    def translation_mm(self) -> np.ndarray:
        return self.translation_m * 1000.0

    @property
    def rpy_deg(self) -> np.ndarray:
        return self.euler_xyz_deg

    @property
    def transform_reference_target(self) -> np.ndarray:
        return _transform_matrix(self.rotation_matrix, self.translation_m)


def estimate_tag_pose(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray | None,
    tag_size_m: float,
    *,
    use_ippe: bool = True,
    canonicalize_corners: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimate one square-tag pose.

    ``corners`` should be in the detector's canonical clockwise order
    ``TL, TR, BR, BL``.  Set ``canonicalize_corners=True`` only when the four
    points are genuinely unordered screen coordinates.  Returns
    ``(rvec, tvec, rotation_matrix, reprojection_error_px)`` in the remapped
    application tag frame (new X = old Z, new Y = -old X, new Z = -old Y).
    IPPE's
    two candidate solutions are evaluated and the positive-depth solution with
    the smallest reprojection error is selected; ITERATIVE is used as a
    compatibility fallback for older OpenCV builds.
    """

    # Detector libraries already return the tag's canonical corner order.  A
    # screen-space re-sort is opt-in only because it changes the tag frame when
    # the marker is rotated far from upright.
    image_points = (
        _canonical_corners(corners) if canonicalize_corners else _detector_corners(corners)
    ).astype(np.float64)
    size = float(tag_size_m)
    if not math.isfinite(size) or size <= 0:
        raise ValueError("tag_size_m must be positive")
    half = size / 2.0
    # This is the point order required by SOLVEPNP_IPPE_SQUARE.
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    matrix = _matrix_data(camera_matrix, name="camera_matrix", shape=(3, 3))
    if distortion_coefficients is None:
        distortion = np.zeros((1, 5), dtype=np.float64)
    else:
        distortion = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1)
        if distortion.size == 0:
            distortion = np.zeros(5, dtype=np.float64)
        if distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError(
                "distortion_coefficients must contain 4, 5, 8, 12 or 14 values"
            )
        distortion = distortion.reshape(1, -1)

    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    if use_ippe and hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
        try:
            result = cv2.solvePnPGeneric(
                object_points,
                image_points,
                matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            # OpenCV versions return (ok, rvecs, tvecs[, reprojectionErrors]).
            if len(result) >= 3 and bool(result[0]):
                rvecs, tvecs = result[1], result[2]
                for index, (rv, tv) in enumerate(zip(rvecs, tvecs)):
                    rv = np.asarray(rv, dtype=np.float64).reshape(3, 1)
                    tv = np.asarray(tv, dtype=np.float64).reshape(3, 1)
                    if float(tv[2, 0]) <= 0:
                        continue
                    projected, _ = cv2.projectPoints(object_points, rv, tv, matrix, distortion)
                    error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
                    candidates.append((rv, tv, error))
        except cv2.error:
            pass
    # OpenCV's IPPE implementation has a known exact-symmetry corner case: for
    # an integer-aligned, fronto-parallel square it can return two zero-rotation
    # candidates with a large reprojection error instead of the valid pi-flip
    # solution.  Fall back to ITERATIVE whenever IPPE's best fit is visibly
    # inconsistent with the measured corners.  Normal sub-pixel noise remains
    # below this guard and retains IPPE's better planar disambiguation.
    best_ippe_error = min((item[2] for item in candidates), default=float("inf"))
    if not candidates or best_ippe_error > 1.0:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            ok = False
        if ok:
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, matrix, distortion
            )
            error = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (projected.reshape(-1, 2) - image_points) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            iterative_candidate = (np.asarray(rvec), np.asarray(tvec), error)
            if not candidates or error < best_ippe_error:
                candidates = [iterative_candidate]
        elif not candidates:
            raise AprilTagError("OpenCV solvePnP failed for AprilTag corners")
    rvec, tvec, reprojection_error = min(candidates, key=lambda item: item[2])
    rotation, _ = cv2.Rodrigues(rvec)
    rvec, rotation = remap_tag_rotation(rotation)
    return rvec, tvec, rotation, float(reprojection_error)


def relative_pose(
    reference: TagPose | np.ndarray,
    target: TagPose | np.ndarray,
    *,
    reference_id: int | str | None = None,
    target_id: int | str | None = None,
) -> RelativePose:
    """Compute target pose in the reference tag frame.

    Inputs may be ``TagPose`` objects or homogeneous 4x4 camera-to-tag
    transforms.  For two ``TagPose`` objects, IDs are copied automatically.
    """

    def get_transform(value: TagPose | np.ndarray) -> np.ndarray:
        if isinstance(value, TagPose):
            return value.transform_camera_tag
        transform = np.asarray(value, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("transform must be a 4x4 matrix")
        return transform

    ref_transform = get_transform(reference)
    target_transform = get_transform(target)
    relative = np.linalg.inv(ref_transform) @ target_transform
    rotation = relative[:3, :3]
    translation = relative[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rotation)
    if reference_id is None:
        reference_id = reference.tag_id if isinstance(reference, TagPose) else "reference"
    if target_id is None:
        target_id = target.tag_id if isinstance(target, TagPose) else "target"
    return RelativePose(
        reference_id=reference_id,
        target_id=target_id,
        rvec=rvec,
        tvec=translation,
        rotation_matrix=rotation,
        euler_xyz_deg=_rotation_to_euler_xyz(rotation),
    )


def relative_poses(
    poses: Sequence[TagPose], reference_id: int | None = None
) -> list[RelativePose]:
    """Return each detected tag's pose relative to one reference tag."""

    valid = [pose for pose in poses if pose.pose_valid]
    if len(valid) < 2:
        return []
    if reference_id is None:
        reference = min(valid, key=lambda pose: pose.tag_id)
    else:
        matches = [pose for pose in valid if pose.tag_id == reference_id]
        if not matches:
            raise KeyError(f"reference tag {reference_id} was not detected")
        reference = matches[0]
    return [relative_pose(reference, pose) for pose in valid if pose is not reference]


def _import_backend(name: str) -> Any:
    return importlib.import_module(name)


_ARUCO_FAMILY_TO_DICT = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}


class AprilTagPoseEstimator:
    """Detect AprilTags and estimate their camera-relative metric poses."""

    def __init__(
        self,
        config: AprilTagPoseConfig | Mapping[str, Any] | str | os.PathLike[str] | np.ndarray | None = None,
        distortion: np.ndarray | None = None,
        *,
        camera_matrix: np.ndarray | None = None,
        distortion_coefficients: np.ndarray | None = None,
        tag_size_m: float | None = None,
        family: str | None = None,
        tag_ids: Sequence[int] | None = None,
        detector_config: Mapping[str, Any] | None = None,
        backend: str = "auto",
        image_size: tuple[int, int] | None = None,
    ):
        """Create an estimator from a config or direct camera arrays.

        The preferred form is ``AprilTagPoseEstimator(AprilTagPoseConfig)``.
        For small integrations and older GUI code a compatibility form is also
        accepted::

            AprilTagPoseEstimator(K, D, tag_size_m=0.04835,
                                  family="tag36h11")

        ``camera_matrix=``/``distortion_coefficients=`` are aliases for the
        direct-array form.  Optional detector keys (``quad_decimate``,
        ``max_hamming``, ``max_reprojection_error_px``...) may be passed in
        ``detector_config``.
        """

        if camera_matrix is not None:
            if config is not None and not isinstance(config, (str, os.PathLike)):
                raise TypeError("pass either config or camera_matrix, not both")
            config = camera_matrix
        if distortion_coefficients is not None:
            if distortion is not None:
                raise TypeError("pass either distortion or distortion_coefficients, not both")
            distortion = distortion_coefficients

        direct_arrays = isinstance(config, np.ndarray) or (
            config is not None
            and distortion is not None
            and not isinstance(config, (AprilTagPoseConfig, Mapping, str, os.PathLike))
        )
        if direct_arrays:
            if config is None:
                raise TypeError("camera_matrix is required for direct-array construction")
            detector_values = dict(detector_config or {})
            allowed_ids = None if tag_ids is None else tuple(int(value) for value in tag_ids)
            if allowed_ids == ():
                allowed_ids = None
            camera = CameraIntrinsics(
                np.asarray(config, dtype=np.float64),
                np.zeros(5, dtype=np.float64) if distortion is None else distortion,
                image_size=image_size,
            )
            self.config = AprilTagPoseConfig(
                camera=camera,
                tag_size_m=DEFAULT_TAG_SIZE_MM / 1000.0 if tag_size_m is None else float(tag_size_m),
                family=DEFAULT_TAG_FAMILY if family is None else str(family),
                backend=backend,
                quad_decimate=float(detector_values.get("quad_decimate", 1.0)),
                quad_sigma=float(detector_values.get("quad_sigma", 0.0)),
                nthreads=int(detector_values.get("nthreads", 1)),
                refine_edges=bool(detector_values.get("refine_edges", True)),
                decode_sharpening=float(detector_values.get("decode_sharpening", 0.25)),
                min_decision_margin=(
                    None
                    if detector_values.get("min_decision_margin") is None
                    else float(detector_values["min_decision_margin"])
                ),
                max_hamming=int(detector_values.get("max_hamming", 0)),
                allowed_ids=allowed_ids,
                max_reprojection_error_px=(
                    None
                    if detector_values.get("max_reprojection_error_px") is None
                    else float(detector_values["max_reprojection_error_px"])
                ),
            )
        elif isinstance(config, AprilTagPoseConfig):
            self.config = config
        elif isinstance(config, (str, os.PathLike)):
            self.config = AprilTagPoseConfig.from_file(config)
        elif config is None:
            self.config = load_pose_config(
                tag_size_mm=DEFAULT_TAG_SIZE_MM if tag_size_m is None else float(tag_size_m),
                family=DEFAULT_TAG_FAMILY if family is None else str(family),
                backend=backend,
            )
        else:
            self.config = AprilTagPoseConfig.from_mapping(config)
        self.backend = ""
        self._detector: Any = None
        self._aruco_dictionaries: list[tuple[str, Any]] = []
        self._aruco_detectors: list[tuple[str, Any]] = []
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        requested = self.config.backend.lower()
        errors: list[str] = []
        if requested in ("auto", "pupil_apriltags", "pupil"):
            try:
                module = _import_backend("pupil_apriltags")
                self._detector = module.Detector(
                    families=self.config.family,
                    nthreads=self.config.nthreads,
                    quad_decimate=self.config.quad_decimate,
                    quad_sigma=self.config.quad_sigma,
                    refine_edges=self.config.refine_edges,
                    decode_sharpening=self.config.decode_sharpening,
                )
                self.backend = "pupil_apriltags"
                return
            except Exception as exc:
                errors.append(f"pupil_apriltags: {exc}")
                if requested not in ("auto", "pupil_apriltags", "pupil"):
                    raise
        if requested in ("auto", "apriltag"):
            try:
                module = _import_backend("apriltag")
                options = {
                    "families": self.config.family,
                    "quad_decimate": self.config.quad_decimate,
                    "quad_sigma": self.config.quad_sigma,
                    "refine_edges": self.config.refine_edges,
                    "decode_sharpening": self.config.decode_sharpening,
                    "nthreads": self.config.nthreads,
                }
                self._detector = module.Detector(options)
                self.backend = "apriltag"
                return
            except Exception as exc:
                errors.append(f"apriltag: {exc}")
        if requested in ("auto", "opencv", "aruco", "opencv_aruco"):
            aruco = getattr(cv2, "aruco", None)
            if aruco is not None:
                families = [self.config.family.lower()]
                if families[0] in ("all", "apriltag"):
                    families = list(_ARUCO_FAMILY_TO_DICT)
                for family in families:
                    constant_name = _ARUCO_FAMILY_TO_DICT.get(family)
                    constant = getattr(aruco, constant_name, None) if constant_name else None
                    if constant is None:
                        errors.append(f"OpenCV: unsupported family {family}")
                        continue
                    dictionary = aruco.getPredefinedDictionary(constant)
                    self._aruco_dictionaries.append((family, dictionary))
                    parameters = aruco.DetectorParameters()
                    # OpenCV exposes these AprilTag-specific knobs only in
                    # relatively recent releases.  Set them when present and
                    # retain defaults on vendor/older builds.
                    for parameter_name, parameter_value in (
                        ("aprilTagQuadDecimate", self.config.quad_decimate),
                        ("aprilTagQuadSigma", self.config.quad_sigma),
                        ("aprilTagDecodeSharpening", self.config.decode_sharpening),
                    ):
                        if hasattr(parameters, parameter_name):
                            try:
                                setattr(parameters, parameter_name, float(parameter_value))
                            except (TypeError, ValueError):
                                pass
                    if self.config.refine_edges and hasattr(
                        parameters, "cornerRefinementMethod"
                    ):
                        refine = getattr(aruco, "CORNER_REFINE_APRILTAG", None)
                        if refine is not None:
                            parameters.cornerRefinementMethod = refine
                    if hasattr(aruco, "ArucoDetector"):
                        self._aruco_detectors.append(
                            (family, aruco.ArucoDetector(dictionary, parameters))
                        )
                    else:
                        # Keep a ``None`` marker; the legacy functional API
                        # is used in _raw_detections below.
                        self._aruco_detectors.append((family, None))
                if self._aruco_dictionaries:
                    self.backend = "opencv_aruco"
                    return
            errors.append("OpenCV aruco AprilTag dictionaries are unavailable")
        detail = "; ".join(errors) if errors else f"unknown backend {requested!r}"
        raise AprilTagBackendUnavailable(
            "No usable AprilTag detector backend. Install pupil_apriltags or "
            f"opencv-contrib-python. Details: {detail}"
        )

    @property
    def camera_matrix(self) -> np.ndarray:
        return self.config.camera.camera_matrix

    @property
    def distortion_coefficients(self) -> np.ndarray:
        return self.config.camera.distortion_coefficients

    def _raw_detections(self, gray: np.ndarray) -> list[dict[str, Any]]:
        if self.backend == "pupil_apriltags":
            detections = self._detector.detect(gray, estimate_tag_pose=False)
            result = []
            for item in detections:
                result.append(
                    {
                        "tag_id": int(item.tag_id),
                        "corners": item.corners,
                        "decision_margin": getattr(item, "decision_margin", None),
                        "hamming": getattr(item, "hamming", None),
                    }
                )
            return result
        if self.backend == "apriltag":
            result = []
            for item in self._detector.detect(gray):
                if "lb-rb-rt-lt" in item:
                    corners = _detector_corners(
                        item["lb-rb-rt-lt"], source="apriltag_lb_rb_rt_lt"
                    )
                else:
                    corners = item.get("corners")
                result.append(
                    {
                        "tag_id": int(item.get("id", item.get("tag_id", -1))),
                        "corners": corners,
                        "decision_margin": item.get("decision_margin"),
                        "hamming": item.get("hamming"),
                    }
                )
            return result
        result = []
        aruco = cv2.aruco
        parameters = None
        # AprilTag-specific parameters exist only in newer OpenCV versions;
        # leave defaults untouched when absent.  Detector instances are cached
        # because constructing one for every video frame is needlessly costly.
        for (family, dictionary), (_, detector) in zip(
            self._aruco_dictionaries, self._aruco_detectors
        ):
            if detector is not None:
                corners, ids, _ = detector.detectMarkers(gray)
            else:  # OpenCV 4.6 compatibility
                if parameters is None:
                    parameters = aruco.DetectorParameters()
                corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
            if ids is None:
                continue
            for corner, tag_id in zip(corners, ids.reshape(-1)):
                result.append(
                    {
                        "tag_id": int(tag_id),
                        "corners": np.asarray(corner).reshape(4, 2),
                        "decision_margin": None,
                        "hamming": None,
                        "family": family,
                    }
                )
        return result

    def detect(
        self,
        image: np.ndarray,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> list[TagPose]:
        """Detect tags in a grayscale/BGR frame and return sorted poses."""

        gray = _gray_image(image)
        actual_size = image_size or (gray.shape[1], gray.shape[0])
        camera = self.config.camera.for_image_size(actual_size)
        detections = self._raw_detections(gray)
        poses: list[TagPose] = []
        for raw in detections:
            raw_tag_id = int(raw["tag_id"])
            if self.config.allowed_ids is not None and raw_tag_id not in self.config.allowed_ids:
                continue
            hamming = raw.get("hamming")
            if hamming is not None and int(hamming) > self.config.max_hamming:
                continue
            margin = raw.get("decision_margin")
            if (
                margin is not None
                and self.config.min_decision_margin is not None
                and float(margin) < self.config.min_decision_margin
            ):
                continue
            try:
                corners = _detector_corners(raw["corners"])
                rvec, tvec, rotation, error = estimate_tag_pose(
                    corners,
                    camera.camera_matrix,
                    camera.distortion_coefficients,
                    self.config.tag_size_m,
                )
            except (ValueError, cv2.error, AprilTagError):
                continue
            if (
                self.config.max_reprojection_error_px is not None
                and error > self.config.max_reprojection_error_px
            ):
                continue
            poses.append(
                TagPose(
                    tag_id=raw_tag_id,
                    family=str(raw.get("family", self.config.family)),
                    corners=corners,
                    center=corners.mean(axis=0),
                    rvec=rvec,
                    tvec=tvec,
                    rotation_matrix=rotation,
                    euler_xyz_deg=_rotation_to_euler_xyz(rotation),
                    reprojection_error_px=error,
                    # NaN is preferable to None for compatibility with GUI
                    # adapters that expose this value as a numeric label.
                    decision_margin=float("nan") if margin is None else float(margin),
                    hamming=None if hamming is None else int(hamming),
                )
            )
        # When multiple dictionaries are scanned, duplicate a tag only once.
        unique: dict[tuple[int, tuple[int, ...]], TagPose] = {}
        for pose in poses:
            key = (pose.tag_id, tuple(np.round(pose.center).astype(int)))
            old = unique.get(key)
            if old is None or pose.reprojection_error_px < old.reprojection_error_px:
                unique[key] = pose
        return sorted(unique.values(), key=lambda pose: pose.tag_id)

    def relative_poses(
        self, poses: Sequence[TagPose], reference_id: int | None = None
    ) -> list[RelativePose]:
        return relative_poses(poses, reference_id)


def draw_detections(
    image: np.ndarray,
    poses: Iterable[TagPose],
    camera: CameraIntrinsics | np.ndarray,
    distortion_coefficients: np.ndarray | None = None,
    *,
    axis_length_m: float | None = None,
    draw_axes: bool = True,
    draw_text: bool = True,
) -> np.ndarray:
    """Draw tag outlines, IDs, 6D pose text, and optional XYZ axes in-place."""

    if isinstance(camera, CameraIntrinsics):
        # Match the estimator's resolution-aware intrinsics behaviour when a
        # caller draws onto a resized preview directly.
        height, width = image.shape[:2]
        scaled_camera = camera.for_image_size((width, height))
        matrix = scaled_camera.camera_matrix
        distortion = scaled_camera.distortion_coefficients
    else:
        matrix = np.asarray(camera, dtype=np.float64)
        distortion = (
            np.zeros((1, 5), dtype=np.float64)
            if distortion_coefficients is None
            else np.asarray(distortion_coefficients, dtype=np.float64)
        )
    canvas = image
    for pose in poses:
        points = np.round(pose.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (0, 220, 0), 2, cv2.LINE_AA)
        for point in points.reshape(-1, 2):
            cv2.circle(canvas, tuple(int(v) for v in point), 3, (0, 180, 255), -1)
        if draw_axes:
            length = axis_length_m if axis_length_m is not None else 0.5 * np.linalg.norm(pose.corners[1] - pose.corners[0]) * pose.translation_m[2] / max(matrix[0, 0], 1e-9)
            length = max(float(length), 0.005)
            try:
                cv2.drawFrameAxes(canvas, matrix, distortion, pose.rvec, pose.tvec, length, 2)
            except cv2.error:
                pass
        if draw_text:
            x, y = np.round(pose.corners[0]).astype(int)
            translation = pose.translation_mm
            angles = pose.euler_xyz_deg
            lines = [
                f"ID {pose.tag_id}  Z {translation[2]:.1f} mm",
                f"X {translation[0]:+.1f} Y {translation[1]:+.1f} mm",
                f"R {angles[0]:+.1f} P {angles[1]:+.1f} Y {angles[2]:+.1f} deg",
            ]
            for index, line in enumerate(lines):
                origin = (int(x), int(y) - 8 - index * 18)
                cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 255, 30), 1, cv2.LINE_AA)
    return canvas


__all__ = [
    "AprilTagBackendUnavailable",
    "AprilTagConfigurationError",
    "AprilTagError",
    "AprilTagPoseConfig",
    "AprilTagPoseEstimator",
    "CameraIntrinsics",
    "RelativePose",
    "TagPose",
    "DEFAULT_INTRINSICS_BASENAME",
    "DEFAULT_TAG_FAMILY",
    "DEFAULT_TAG_SIZE_MM",
    "draw_detections",
    "estimate_tag_pose",
    "load_pose_config",
    "relative_pose",
    "relative_poses",
]
