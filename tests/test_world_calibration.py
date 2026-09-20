from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from apriltag_coordinates import remap_tag_rotation
from world_calibration_core import (
    WorldCalibration, calibrate_world, inverse_transform,
    reference_world_pose, rigid_transform,
)


class WorldCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.known = reference_world_pose()
        original, _ = cv2.Rodrigues(np.array([0.35, -0.2, 0.1]))
        _, rotation = remap_tag_rotation(original)
        self.measured = rigid_transform(rotation, np.array([0.04, -0.03, 0.6]))

    def test_known_world_origin_and_axis_alignment(self):
        result = calibrate_world([self.measured] * 5, self.known)
        reference_in_world = result.world_pose(self.measured)
        np.testing.assert_allclose(reference_in_world[:3, 3] * 1000, [126.1, -12.5, 142])
        # No second application of the tag basis change is allowed here.
        np.testing.assert_allclose(reference_in_world[:3, :3], np.eye(3), atol=1e-12)
        expected_rotation = self.measured[:3, :3].T
        expected_translation = self.known[:3, 3] - expected_rotation @ self.measured[:3, 3]
        np.testing.assert_allclose(result.world_from_camera[:3, :3], expected_rotation, atol=1e-12)
        np.testing.assert_allclose(result.world_from_camera[:3, 3], expected_translation)
        np.testing.assert_allclose(result.world_points(self.measured[:3, 3]), self.known[:3, 3])

    def test_new_target_and_points_recover_world_coordinates(self):
        known = reference_world_pose([126.1, -12.5, 142], [10, -20, 30])
        result = calibrate_world([self.measured] * 5, known)
        target_world = reference_world_pose([300, 200, 400], [-20, 15, 60])
        target_camera = inverse_transform(result.world_from_camera) @ target_world
        np.testing.assert_allclose(result.world_pose(target_camera), target_world, atol=1e-12)
        camera_points = np.array([[0.02, 0.03, 0.6], [-0.1, -0.2, 0.9]])
        expected = (result.world_from_camera @ np.column_stack([camera_points, np.ones(2)]).T).T[:, :3]
        np.testing.assert_allclose(result.world_points(camera_points), expected)

    def test_average_handles_rotation_wraparound(self):
        samples = [reference_world_pose([0, 0, 600], [0, 0, angle]) for angle in [179.8, -179.8, 180]]
        result = calibrate_world(samples, self.known)
        np.testing.assert_allclose(result.camera_from_reference[:3, :3], np.diag([-1, -1, 1]), atol=1e-10)
        self.assertLess(result.rotation_rms_deg, 0.2)
        self.assertAlmostEqual(result.translation_rms_mm, 0)

    def test_moving_camera_or_reference_is_rejected(self):
        moved = self.measured.copy()
        moved[0, 3] += 0.02
        with self.assertRaisesRegex(ValueError, "采样不稳定"):
            calibrate_world([self.measured, self.measured, moved], self.known)
        rotated = self.measured.copy()
        rotated[:3, :3] = rotated[:3, :3] @ cv2.Rodrigues(np.array([0, 0, 0.15]))[0]
        with self.assertRaisesRegex(ValueError, "采样不稳定"):
            calibrate_world([self.measured, self.measured, rotated], self.known)

    def test_invalid_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "至少"):
            calibrate_world([self.measured], self.known)
        for bad in (
            np.full((4, 4), np.nan),
            np.diag([1., 1., -1., 1.]),
            rigid_transform(np.eye(3), [0, 0, -1]),
        ):
            with self.assertRaises(ValueError):
                calibrate_world([bad] * 3, self.known)

    def test_save_load_roundtrip_and_reject_inconsistent_files(self):
        result = calibrate_world([self.measured] * 5, self.known, metadata={"camera_serial": "test"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.json"
            result.save(path)
            loaded = WorldCalibration.load(path)
            np.testing.assert_allclose(loaded.world_from_camera, result.world_from_camera)
            self.assertEqual(loaded.metadata, result.metadata)
            original = result.to_dict()
            for field, value in (
                ("translation_unit", "mm"),
                ("tag_basis_old_from_new", np.eye(3).tolist()),
                ("T_world_camera", np.eye(4).tolist()),
                ("T_camera_world", np.eye(4).tolist()),
            ):
                modified = dict(original)
                modified[field] = value
                path.write_text(json.dumps(modified), encoding="utf-8")
                with self.assertRaises(ValueError):
                    WorldCalibration.load(path)


if __name__ == "__main__":
    unittest.main()
