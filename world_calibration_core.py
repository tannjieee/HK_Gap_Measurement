"""Calibrate a fixed camera from a tag with a known world pose.

All transforms map column vectors from the suffix frame to the prefix frame;
translations are in metres. Input tag poses already use the application's
remapped axes. Do not apply the AprilTag axis mapping a second time here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from apriltag_coordinates import TAG_BASIS_OLD_FROM_NEW


def rigid_transform(rotation: np.ndarray, translation_m: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_m, dtype=np.float64).reshape(3)
    return validate_transform(transform)


def validate_transform(value: Any) -> np.ndarray:
    transform = np.array(value, dtype=np.float64, copy=True)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("位姿必须为有限的 4×4 矩阵")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("位姿不是合法的右手刚体变换")
    return transform


def inverse_transform(value: np.ndarray) -> np.ndarray:
    transform = validate_transform(value)
    rotation = transform[:3, :3].T
    return rigid_transform(rotation, -rotation @ transform[:3, 3])


def reference_world_pose(
    xyz_mm: Sequence[float] = (126.1, -12.5, 142.0),
    rpy_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Known remapped tag pose; R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    angles = np.asarray(rpy_deg, dtype=np.float64)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("世界系旋转角必须为三个有限数值（度）")
    roll, pitch, yaw = np.radians(angles)
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rigid_transform(rz @ ry @ rx, np.asarray(xyz_mm) / 1000.0)


@dataclass
class WorldCalibration:
    world_from_camera: np.ndarray
    world_from_reference: np.ndarray
    camera_from_reference: np.ndarray
    sample_count: int
    translation_rms_mm: float
    rotation_rms_deg: float
    metadata: dict[str, Any]

    def world_pose(self, camera_from_tag: np.ndarray) -> np.ndarray:
        return self.world_from_camera @ validate_transform(camera_from_tag)

    def world_points(self, camera_points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(camera_points_m, dtype=np.float64)
        if points.ndim not in (1, 2) or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("相机坐标点须为 (3,) 或 (N, 3)，单位米")
        return points @ self.world_from_camera[:3, :3].T + self.world_from_camera[:3, 3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "translation_unit": "m",
            "transform_convention": "p_world = T_world_camera @ p_camera",
            "tag_basis_old_from_new": TAG_BASIS_OLD_FROM_NEW.tolist(),
            "T_world_camera": self.world_from_camera.tolist(),
            "T_camera_world": inverse_transform(self.world_from_camera).tolist(),
            "T_world_reference": self.world_from_reference.tolist(),
            "T_camera_reference": self.camera_from_reference.tolist(),
            "camera_origin_world_mm": (self.world_from_camera[:3, 3] * 1000).tolist(),
            "sample_count": self.sample_count,
            "translation_rms_mm": self.translation_rms_mm,
            "rotation_rms_deg": self.rotation_rms_deg,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        # Serialize before opening the destination so invalid metadata cannot
        # truncate an existing result.
        content = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> WorldCalibration:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("translation_unit") != "m":
            raise ValueError("不支持的世界标定格式或单位")
        if not np.array_equal(data.get("tag_basis_old_from_new"), TAG_BASIS_OLD_FROM_NEW):
            raise ValueError("标定文件的标签坐标轴定义不匹配")
        world_camera = validate_transform(data["T_world_camera"])
        world_reference = validate_transform(data["T_world_reference"])
        camera_reference = validate_transform(data["T_camera_reference"])
        camera_world = validate_transform(data["T_camera_world"])
        if (
            not np.allclose(world_camera @ camera_reference, world_reference, atol=1e-7, rtol=0)
            or not np.allclose(world_camera @ camera_world, np.eye(4), atol=1e-7, rtol=0)
        ):
            raise ValueError("标定矩阵之间不一致")
        count = data["sample_count"]
        translation_rms = float(data["translation_rms_mm"])
        rotation_rms = float(data["rotation_rms_deg"])
        if (
            not isinstance(count, int) or count < 1
            or not np.isfinite([translation_rms, rotation_rms]).all()
            or min(translation_rms, rotation_rms) < 0
            or not isinstance(data["metadata"], dict)
        ):
            raise ValueError("标定统计信息无效")
        return cls(world_camera, world_reference, camera_reference, count,
                   translation_rms, rotation_rms, data["metadata"])


def calibrate_world(
    camera_from_reference_samples: Sequence[np.ndarray],
    world_from_reference: np.ndarray,
    *,
    max_translation_rms_mm: float = 2.0,
    max_rotation_rms_deg: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> WorldCalibration:
    """Average static tag observations, check scatter, and solve T_world_camera.

    Scatter measures repeatability, not absolute calibration accuracy. Samples
    must be captured with both the camera and reference tag held fixed.
    """

    if len(camera_from_reference_samples) < 3:
        raise ValueError("至少需要 3 帧不同的有效观测")
    limits = [max_translation_rms_mm, max_rotation_rms_deg]
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError("稳定性阈值必须为有限正数")
    samples = np.stack([validate_transform(sample) for sample in camera_from_reference_samples])
    if np.any(samples[:, 2, 3] <= 0):
        raise ValueError("参考标签必须位于相机前方")
    known_pose = validate_transform(world_from_reference)
    # Project the average rotation onto SO(3), avoiding Euler-angle wraparound.
    u, _, vt = np.linalg.svd(samples[:, :3, :3].mean(axis=0))
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    translation = samples[:, :3, 3].mean(axis=0)
    mean_pose = rigid_transform(rotation, translation)
    translation_rms = float(np.sqrt(np.mean(np.sum((samples[:, :3, 3] - translation) ** 2, axis=1))) * 1000)
    angle_errors = [
        np.linalg.norm(cv2.Rodrigues(rotation.T @ sample[:3, :3])[0])
        for sample in samples
    ]
    rotation_rms = float(np.degrees(np.sqrt(np.mean(np.square(angle_errors)))))
    if translation_rms > max_translation_rms_mm or rotation_rms > max_rotation_rms_deg:
        raise ValueError(
            f"采样不稳定：位置 RMS={translation_rms:.3f} mm，角度 RMS={rotation_rms:.3f}°；"
            "请固定相机和 ID 0，改善成像后重新采样"
        )
    world_camera = known_pose @ inverse_transform(mean_pose)
    return WorldCalibration(world_camera, known_pose, mean_pose, len(samples),
                            translation_rms, rotation_rms, dict(metadata or {}))
