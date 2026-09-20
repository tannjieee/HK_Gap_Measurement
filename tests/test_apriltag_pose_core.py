from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from apriltag_pose_core import (
    AprilTagPoseConfig,
    AprilTagPoseEstimator,
    CameraIntrinsics,
    estimate_tag_pose,
    load_pose_config,
    relative_pose,
)


class PoseMathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = np.array(
            [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.distortion = np.zeros((1, 5), dtype=np.float64)
        self.size = 0.04835

    def test_solve_pnp_recovers_synthetic_square(self) -> None:
        half = self.size / 2.0
        object_points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        expected_rvec = np.array([[0.18], [-0.11], [0.07]], dtype=np.float64)
        expected_tvec = np.array([[0.025], [-0.018], [0.42]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            object_points,
            expected_rvec,
            expected_tvec,
            self.camera,
            self.distortion,
        )
        rvec, tvec, rotation, error = estimate_tag_pose(
            image_points.reshape(4, 2),
            self.camera,
            self.distortion,
            self.size,
        )
        self.assertLess(error, 1e-6)
        np.testing.assert_allclose(tvec, expected_tvec, atol=2e-4)
        self.assertEqual(rotation.shape, (3, 3))
        self.assertTrue(np.isfinite(rvec).all())
        original_rotation, _ = cv2.Rodrigues(expected_rvec)
        np.testing.assert_allclose(rotation[:, 0], original_rotation[:, 2], atol=1e-6)
        np.testing.assert_allclose(rotation[:, 1], -original_rotation[:, 0], atol=1e-6)
        np.testing.assert_allclose(rotation[:, 2], -original_rotation[:, 1], atol=1e-6)
        # The same physical corners expressed in the new tag frame must still
        # project to the detected pixels with the returned pose.
        new_points = np.column_stack(
            (object_points[:, 2], -object_points[:, 0], -object_points[:, 1])
        )
        projected, _ = cv2.projectPoints(
            new_points, rvec, tvec, self.camera, self.distortion
        )
        np.testing.assert_allclose(projected, image_points, atol=1e-5)

    def test_detector_order_is_preserved_under_large_tag_rotation(self) -> None:
        """Screen-space corner sorting must not rotate the tag frame."""
        half = self.size / 2.0
        object_points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        # A substantial tilt makes the physical canonical bottom-left corner
        # appear above the screen-space top-left corner.
        expected_rvec = np.array([[0.40], [-0.20], [0.10]], dtype=np.float64)
        expected_tvec = np.array([[0.03], [-0.02], [0.35]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            object_points,
            expected_rvec,
            expected_tvec,
            self.camera,
            np.array([-0.0667, -0.0594, -0.00057, -0.00034, 0.5226]),
        )
        rvec, tvec, _rotation, error = estimate_tag_pose(
            image_points.reshape(4, 2),
            self.camera,
            np.array([-0.0667, -0.0594, -0.00057, -0.00034, 0.5226]),
            self.size,
        )
        self.assertLess(error, 1e-5)
        np.testing.assert_allclose(tvec, expected_tvec, atol=1e-4)
        original_rotation, _ = cv2.Rodrigues(expected_rvec)
        actual_rotation, _ = cv2.Rodrigues(rvec)
        expected_rotation = np.column_stack(
            (original_rotation[:, 2], -original_rotation[:, 0], -original_rotation[:, 1])
        )
        np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-4)

    def test_ippe_exact_axis_aligned_square_falls_back_to_iterative(self) -> None:
        image_points = np.array(
            [[540.0, 380.0], [740.0, 380.0], [740.0, 580.0], [540.0, 580.0]],
            dtype=np.float64,
        )
        _rvec, tvec, _rotation, error = estimate_tag_pose(
            image_points,
            np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            self.size,
        )
        self.assertLess(error, 1e-4)
        self.assertGreater(float(tvec[2, 0]), 0.0)

    def test_relative_transform_has_expected_translation(self) -> None:
        # Camera-to-tag transforms with identity orientation.  The target is
        # 100 mm to the right of the reference in reference coordinates.
        ref_matrix = np.eye(4, dtype=np.float64)
        ref_matrix[:3, 3] = [0.0, 0.0, 0.5]
        target_matrix = np.eye(4, dtype=np.float64)
        target_matrix[:3, 3] = [0.1, 0.0, 0.5]
        rel = relative_pose(ref_matrix, target_matrix)
        np.testing.assert_allclose(rel.translation_mm, [100.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(rel.rotation_matrix, np.eye(3), atol=1e-12)


class ConfigTest(unittest.TestCase):
    def test_project_calibration_config_loads(self) -> None:
        path = Path(__file__).parents[1] / "config" / "apriltag_pose.yaml"
        if not path.is_file():
            self.skipTest("AprilTag example config has not been created yet")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = AprilTagPoseConfig.from_file(path)
        # This is the user's editable camera/tag configuration, not a fixed
        # 48.35 mm fixture. Check that its selected size and IDs are honored.
        self.assertAlmostEqual(config.tag_size_m, raw["tag"]["size_mm"] / 1000.0, places=8)
        self.assertEqual(config.family, "tag36h11")
        self.assertEqual(config.camera.image_size, (1440, 1080))
        self.assertEqual(config.allowed_ids, tuple(raw["tag"].get("ids") or ()) or None)
        self.assertTrue(config.camera.source_path)

    def test_compact_mapping_accepts_intrinsics_and_ids(self) -> None:
        mapping = {
            "tag": {"family": "tag25h9", "size_mm": 48.35, "ids": [3, 7]},
            "camera_matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
            "image_size": [1280, 720],
        }
        config = AprilTagPoseConfig.from_mapping(mapping)
        self.assertEqual(config.allowed_ids, (3, 7))
        self.assertEqual(config.camera.image_size, (1280, 720))
        self.assertAlmostEqual(config.tag_size_m, 0.04835)


class DetectorTest(unittest.TestCase):
    def test_blank_frame_has_no_tags_with_opencv_fallback(self) -> None:
        camera = CameraIntrinsics(
            np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
            np.zeros(5),
            (640, 480),
        )
        config = AprilTagPoseConfig(camera=camera, backend="opencv", family="tag36h11")
        estimator = AprilTagPoseEstimator(config)
        poses = estimator.detect(np.full((480, 640), 127, dtype=np.uint8))
        self.assertEqual(poses, [])

    def test_direct_constructor_scales_intrinsics_for_live_size(self) -> None:
        matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        estimator = AprilTagPoseEstimator(
            matrix,
            np.zeros(5),
            image_size=(640, 480),
            backend="opencv",
        )
        scaled = estimator.config.camera.for_image_size((320, 240))
        np.testing.assert_allclose(
            scaled.camera_matrix,
            [[400, 0, 160], [0, 400, 120], [0, 0, 1]],
        )

    def test_generated_apriltag_is_detected_when_aruco_is_available(self) -> None:
        if not hasattr(cv2, "aruco"):
            self.skipTest("OpenCV aruco module is unavailable")
        aruco = cv2.aruco
        if not hasattr(aruco, "DICT_APRILTAG_36h11"):
            self.skipTest("OpenCV AprilTag dictionaries are unavailable")
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        marker = aruco.generateImageMarker(dictionary, 7, 240)
        frame = np.full((480, 640), 255, dtype=np.uint8)
        frame[120:360, 200:440] = marker
        camera = CameraIntrinsics(
            np.array([[900, 0, 320], [0, 900, 240], [0, 0, 1]], dtype=np.float64),
            np.zeros(5),
            (640, 480),
        )
        estimator = AprilTagPoseEstimator(
            AprilTagPoseConfig(camera=camera, backend="opencv", family="tag36h11")
        )
        poses = estimator.detect(frame)
        self.assertEqual(len(poses), 1)
        self.assertEqual(poses[0].tag_id, 7)
        self.assertGreater(poses[0].translation_m[2], 0)
        self.assertLess(poses[0].reprojection_error_px, 1.0)


if __name__ == "__main__":
    unittest.main()
