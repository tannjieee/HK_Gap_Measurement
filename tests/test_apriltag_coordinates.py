from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from apriltag_coordinates import remap_tag_rotation
from apriltag_pose_core import estimate_tag_pose, relative_pose
from apriltag_pose_gui import (
    OpenCVAprilTagEstimator,
    PoseGuiConfig,
    _coerce_pose_record,
    relative_poses,
)
from nvidia_apriltag_backend import NvidiaAprilTagEstimator, _square_object_points


class TagCoordinatesTest(unittest.TestCase):
    def test_axes_remain_right_handed(self):
        original, _ = cv2.Rodrigues(np.array([0.3, -0.4, 0.2]))
        rvec, rotation = remap_tag_rotation(original)
        np.testing.assert_allclose(rotation[:, 0], original[:, 2])
        np.testing.assert_allclose(rotation[:, 1], -original[:, 0])
        np.testing.assert_allclose(rotation[:, 2], -original[:, 1])
        np.testing.assert_allclose(np.cross(rotation[:, 0], rotation[:, 1]), rotation[:, 2])
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
        np.testing.assert_allclose(cv2.Rodrigues(rvec)[0], rotation, atol=1e-12)

    def test_relative_pose_uses_new_reference_axes(self):
        _, reference_rotation = remap_tag_rotation(np.eye(3))
        original_target_rotation, _ = cv2.Rodrigues(np.array([0.2, -0.1, 0.3]))
        _, target_rotation = remap_tag_rotation(original_target_rotation)
        reference = np.eye(4)
        reference[:3, :3] = reference_rotation
        reference[:3, 3] = [0, 0, 0.5]
        target = np.eye(4)
        target[:3, :3] = target_rotation
        target[:3, 3] = [0.1, 0.2, 0.8]
        result = relative_pose(reference, target)
        np.testing.assert_allclose(result.translation_mm, [300, -100, -200], atol=1e-9)
        reconstructed = reference @ result.transform_reference_target
        np.testing.assert_allclose(reconstructed, target, atol=1e-12)

    def test_gui_fallback_matches_core_and_adapter_does_not_remap_twice(self):
        matrix = np.array([[900., 0, 320], [0, 900., 240], [0, 0, 1.]])
        distortion = np.zeros(5)
        config = PoseGuiConfig(Path('unused'), Path('unused'), backend='opencv')
        estimator = OpenCVAprilTagEstimator(matrix, distortion, config)
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 7, 240
        )
        frame = np.full((480, 640), 255, dtype=np.uint8)
        frame[120:360, 200:440] = marker
        records = estimator.detect(frame)
        self.assertEqual(len(records), 1)
        record = records[0]
        _, tvec, rotation, error = estimate_tag_pose(
            record.corners, matrix, distortion, config.tag_size_m
        )
        np.testing.assert_allclose(record.rotation_matrix, rotation, atol=1e-5)
        np.testing.assert_allclose(record.tvec, tvec, atol=1e-6)
        self.assertLess(error, 1.0)
        adapted = _coerce_pose_record(record, config.family)
        np.testing.assert_allclose(adapted.rotation_matrix, rotation, atol=1e-5)
        np.testing.assert_allclose(adapted.euler_xyz_deg, record.euler_xyz_deg)
        target = _coerce_pose_record(record, config.family)
        target.tag_id = 8
        # Move 100 mm along the original +X (= new -Y).
        target.tvec = record.tvec - record.rotation_matrix[:, 1:2] * 0.1
        relative = relative_poses([record, target], record.tag_id)[8]
        np.testing.assert_allclose(relative.translation_mm, [0, -100, 0], atol=1e-8)

    def test_nvidia_reprojection_precedes_frame_remap(self):
        matrix = np.array([[900., 0, 320], [0, 900., 240], [0, 0, 1.]])
        points = _square_object_points(0.04835)
        tvec = np.array([0.025, -0.018, 0.42])
        corners, _ = cv2.projectPoints(points, np.zeros(3), tvec, matrix, np.zeros(5))
        native_pose = SimpleNamespace(
            position=SimpleNamespace(x=tvec[0], y=tvec[1], z=tvec[2]),
            orientation=SimpleNamespace(x=0., y=0., z=0., w=1.),
        )
        detection = SimpleNamespace(
            id=7,
            corners=[SimpleNamespace(x=x, y=y) for x, y in corners.reshape(4, 2)],
            pose=SimpleNamespace(pose=SimpleNamespace(pose=native_pose)),
        )
        estimator = object.__new__(NvidiaAprilTagEstimator)
        estimator._config = SimpleNamespace(tag_ids=(), tag_size_m=0.04835, max_reprojection_error_px=1.)
        estimator._distortion = np.zeros(5)
        estimator._family = 'tag36h11'
        records = estimator._to_records(
            SimpleNamespace(detections=[detection]), matrix, np.zeros(5), (640, 480)
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertLess(record.reprojection_error_px, 1e-8)
        np.testing.assert_allclose(record.translation_mm, [25, -18, 420])
        np.testing.assert_allclose(record.rotation_matrix[:, 0], [0, 0, 1])
        np.testing.assert_allclose(record.rotation_matrix[:, 1], [-1, 0, 0])
        np.testing.assert_allclose(record.rotation_matrix[:, 2], [0, -1, 0])
        new_points = np.column_stack((points[:, 2], -points[:, 0], -points[:, 1]))
        projected, _ = cv2.projectPoints(new_points, record.rvec, record.tvec, matrix, np.zeros(5))
        np.testing.assert_allclose(projected, corners, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
