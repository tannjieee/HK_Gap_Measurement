from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibration_core import (
    BoardSpec,
    CalibrationSample,
    calibrate_samples,
    detect_chessboard,
    save_calibration,
)


def rotation_vector(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotate_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotate_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotate_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    vector, _ = cv2.Rodrigues(rotate_z @ rotate_y @ rotate_x)
    return vector


def synthetic_samples(
    board: BoardSpec,
) -> tuple[list[CalibrationSample], np.ndarray, np.ndarray]:
    image_size = (1440, 1080)
    camera_matrix = np.array(
        [[1100.0, 0.0, 720.0], [0.0, 1080.0, 540.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.array([-0.08, 0.01, 0.001, -0.0007, -0.001])
    object_points = board.object_points()
    board_center = np.array(
        [
            (board.inner_corners_x - 1) * board.square_size_m / 2,
            (board.inner_corners_y - 1) * board.square_size_m / 2,
            0.0,
        ]
    )
    samples: list[CalibrationSample] = []
    rng = np.random.default_rng(9234)
    for index in range(24):
        rx = rng.uniform(-0.38, 0.38)
        ry = rng.uniform(-0.42, 0.42)
        rz = rng.uniform(-0.35, 0.35)
        rvec = rotation_vector(rx, ry, rz)
        rotation, _ = cv2.Rodrigues(rvec)
        z = rng.uniform(0.48, 0.90)
        target_u = rng.uniform(320.0, 1120.0)
        target_v = rng.uniform(250.0, 830.0)
        camera_center = np.array(
            [
                (target_u - camera_matrix[0, 2]) * z / camera_matrix[0, 0],
                (target_v - camera_matrix[1, 2]) * z / camera_matrix[1, 1],
                z,
            ]
        )
        tvec = (camera_center - rotation @ board_center).reshape(3, 1)
        corners, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        samples.append(
            CalibrationSample(
                corners=corners.astype(np.float32),
                image_size=image_size,
                descriptor=np.array([index, 0, 0, 0, 1, 0, 0], np.float32),
                sharpness=100.0,
                coverage=0.1,
                source_name=f"synthetic_{index:02d}.png",
            )
        )
    return samples, camera_matrix, distortion


class BoardSpecTest(unittest.TestCase):
    def test_square_count_maps_to_inner_corners(self) -> None:
        board = BoardSpec(12, 9, 15.0)
        points = board.object_points()
        self.assertEqual(board.pattern_size, (11, 8))
        self.assertEqual(board.point_count, 88)
        self.assertEqual(points.shape, (88, 3))
        self.assertEqual(points.dtype, np.float32)
        np.testing.assert_allclose(points[0], [0, 0, 0], atol=1e-8)
        np.testing.assert_allclose(points[1], [0.015, 0, 0], atol=1e-8)
        np.testing.assert_allclose(points[-1], [0.150, 0.105, 0], atol=1e-7)

    def test_invalid_board_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoardSpec(2, 9, 15)
        with self.assertRaises(ValueError):
            BoardSpec(12, 9, 0)
        with self.assertRaises(ValueError):
            BoardSpec(12, 9, math.nan)


class DetectionTest(unittest.TestCase):
    def test_detects_rendered_12_by_9_square_board(self) -> None:
        board = BoardSpec()
        tile = 64
        margin = 120
        image = np.full(
            (board.squares_y * tile + 2 * margin,
             board.squares_x * tile + 2 * margin),
            255,
            dtype=np.uint8,
        )
        for row in range(board.squares_y):
            for column in range(board.squares_x):
                if (row + column) % 2 == 0:
                    x0 = margin + column * tile
                    y0 = margin + row * tile
                    cv2.rectangle(
                        image, (x0, y0), (x0 + tile, y0 + tile), 0, -1
                    )
        detection = detect_chessboard(image, board)
        self.assertTrue(detection.found)
        self.assertIsNotNone(detection.corners)
        self.assertEqual(detection.corners.shape, (88, 1, 2))
        self.assertTrue(np.isfinite(detection.corners).all())
        self.assertGreater(detection.coverage, 0.1)
        self.assertGreater(detection.minimum_spacing, 50)

    def test_blank_image_has_no_detection(self) -> None:
        detection = detect_chessboard(
            np.full((600, 800), 127, dtype=np.uint8), BoardSpec()
        )
        self.assertFalse(detection.found)


class CalibrationTest(unittest.TestCase):
    def test_recovers_known_intrinsics_from_synthetic_views(self) -> None:
        board = BoardSpec()
        samples, expected_matrix, expected_distortion = synthetic_samples(board)
        result = calibrate_samples(
            samples, board, minimum_samples=12, reject_outliers=False
        )
        self.assertLess(result.rms_error, 1e-3)
        np.testing.assert_allclose(
            result.camera_matrix, expected_matrix, rtol=0, atol=0.1
        )
        np.testing.assert_allclose(
            result.distortion_coefficients,
            expected_distortion,
            rtol=0,
            atol=1e-3,
        )
        self.assertEqual(len(result.rotation_vectors), len(samples))
        self.assertEqual(len(result.per_view_errors), len(samples))

    def test_mixed_resolutions_are_rejected(self) -> None:
        board = BoardSpec()
        samples, _, _ = synthetic_samples(board)
        samples[-1].image_size = (1280, 720)
        with self.assertRaisesRegex(ValueError, "has size"):
            calibrate_samples(samples, board, minimum_samples=12)

    def test_gross_outlier_view_is_rejected(self) -> None:
        board = BoardSpec()
        samples, _, _ = synthetic_samples(board)
        noise = np.random.default_rng(17).normal(
            0, 4, samples[-1].corners.shape
        )
        samples[-1].corners += noise.astype(np.float32)
        result = calibrate_samples(samples, board, minimum_samples=12)
        self.assertEqual(result.rejected_indices, (len(samples) - 1,))
        self.assertLess(result.rms_error, 1e-3)

    def test_yaml_and_json_output_round_trip(self) -> None:
        board = BoardSpec()
        samples, _, _ = synthetic_samples(board)
        result = calibrate_samples(
            samples, board, minimum_samples=12, reject_outliers=False
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "intrinsics.yaml"
            yaml_path, json_path, sample_directory = save_calibration(
                output,
                result,
                board,
                samples,
                camera_name="synthetic_camera",
                save_sample_images=False,
            )
            self.assertIsNone(sample_directory)
            yaml_document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            json_document = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(yaml_document, json_document)
            self.assertEqual(yaml_document["image_width"], 1440)
            self.assertEqual(yaml_document["image_height"], 1080)
            self.assertEqual(yaml_document["board"]["squares_x"], 12)
            self.assertEqual(yaml_document["board"]["inner_corners_x"], 11)
            self.assertEqual(yaml_document["board"]["square_size_mm"], 15.0)
            self.assertEqual(yaml_document["camera_matrix"]["rows"], 3)
            self.assertEqual(yaml_document["camera_matrix"]["cols"], 3)
            self.assertEqual(
                len(yaml_document["distortion_coefficients"]["data"]), 5
            )


if __name__ == "__main__":
    unittest.main()
