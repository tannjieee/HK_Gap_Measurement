from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from calibration_core import BoardSpec, ChessboardDetection, make_sample
from gemini335l_calibration import (
    Packet, SolveThread, camera_geometry, image_to_bgr, rejection_reason,
)
from test_calibration_core import synthetic_samples


class ImageConversionTest(unittest.TestCase):
    def test_rgb_padding_and_owned_result(self):
        data = bytearray([255, 0, 0, 0, 255, 0, 99, 99,
                          0, 0, 255, 255, 255, 255, 99, 99])
        msg = SimpleNamespace(width=2, height=2, step=8, encoding="rgb8", data=data)
        actual = image_to_bgr(msg)
        data[0] = 0
        np.testing.assert_array_equal(actual, [[[0, 0, 255], [0, 255, 0]],
                                               [[255, 0, 0], [255, 255, 255]]])

    def test_supported_encodings(self):
        for encoding, data in (("bgr8", [1, 2, 3]), ("bgra8", [1, 2, 3, 128]),
                               ("rgba8", [3, 2, 1, 128]), ("mono8", [7])):
            with self.subTest(encoding=encoding):
                msg = SimpleNamespace(width=1, height=1, step=len(data),
                                      encoding=encoding, data=bytes(data))
                np.testing.assert_array_equal(image_to_bgr(msg)[0, 0],
                                              [7, 7, 7] if encoding == "mono8" else [1, 2, 3])

    def test_malformed_or_depth_image_rejected(self):
        for encoding, step, data in (("16UC1", 2, b"00"), ("rgb8", 2, b"00"),
                                     ("rgb8", 3, b"00"), ("rgb8", 3, b"0000")):
            with self.subTest(encoding=encoding, step=step, data=data):
                with self.assertRaises(ValueError):
                    image_to_bgr(SimpleNamespace(width=1, height=1, step=step,
                                                encoding=encoding, data=data))


class SampleGuardTest(unittest.TestCase):
    def setUp(self):
        self.board = BoardSpec()
        self.packet = Packet(
            np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.uint8),
            ChessboardDetection(found=True, corners=np.zeros((88, 1, 2), np.float32),
                                sharpness=100, coverage=0.1, minimum_spacing=12,
                                descriptor=np.array([0.5, 0.5, 0.3, 0, 1, 1, 1])),
            self.board, time.monotonic(), ("color", 10, 42), ("geometry",), {})

    def reason(self, **kwargs):
        defaults = dict(samples=[], keys=[], geometry=None, automatic=False)
        defaults.update(kwargs)
        return rejection_reason(self.packet, **defaults)

    def test_valid_duplicate_and_stale(self):
        self.assertEqual(self.reason(), "")
        self.assertIn("此帧", self.reason(keys=[self.packet.key]))
        self.assertIn("新鲜", self.reason(now=self.packet.received_at + 2))

    def test_missing_info_and_changed_geometry(self):
        self.assertIn("配置已变化", self.reason(samples=[object()], geometry=("changed",)))
        self.packet.geometry = None
        self.assertIn("CameraInfo", self.reason())

    def test_blur_size_and_novelty(self):
        sample = make_sample(self.packet.detection, self.packet.image_size)
        self.assertIn("姿态相近", self.reason(samples=[sample], geometry=self.packet.geometry,
                                           automatic=True))
        self.packet.detection.sharpness = 10
        self.assertIn("不够清晰", self.reason(automatic=True))
        self.packet.detection.minimum_spacing = 4
        self.assertIn("棋盘太小", self.reason())

    def test_geometry_detects_same_size_roi_change(self):
        info = SimpleNamespace(width=640, height=480, header=SimpleNamespace(frame_id="rgb"),
                               binning_x=0, binning_y=0, distortion_model="plumb_bob",
                               k=list(range(9)), d=[0]*5, r=list(range(9)), p=list(range(12)),
                               roi=SimpleNamespace(x_offset=0, y_offset=0, width=640,
                                                   height=480, do_rectify=False))
        before = camera_geometry(info)
        info.roi.x_offset = 12
        self.assertNotEqual(before, camera_geometry(info))


class OutputTest(unittest.TestCase):
    def test_solver_recovers_intrinsics_and_saves_gemini_session(self):
        board = BoardSpec()
        samples, expected_k, _ = synthetic_samples(board)
        samples[0].gray_image = np.zeros((1080, 1440), np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            worker = SolveThread(samples, board, 12, Path(temp) / "rgb.yaml",
                                 {"image_topic": "/gemini335l/color/image_raw"}, [("color", 1, 2)])
            outputs, errors = [], []
            worker.solved.connect(outputs.append)
            worker.failed.connect(errors.append)
            worker.run()
            self.assertFalse(errors, errors)
            self.assertEqual(len(outputs), 1)
            result, paths, session = outputs[0]
            self.assertLess(result.rms_error, 0.001)
            np.testing.assert_allclose(result.camera_matrix, expected_k, atol=0.1)
            document = yaml.safe_load(paths[0].read_text())
            self.assertEqual(document["camera_name"], "gemini335l_color")
            self.assertEqual(document["image_width"], 1440)
            self.assertEqual(document["distortion_model"], "plumb_bob")
            self.assertEqual(len(document["distortion_coefficients"]["data"]), 5)
            self.assertTrue(paths[1].is_file())
            self.assertTrue((paths[2] / "view_001.png").is_file())
            self.assertEqual(json.loads(session.read_text())["sample_stamps"], [["color", 1, 2]])

    def test_solver_failure_does_not_write_success_result(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "rgb.yaml"
            worker = SolveThread([], BoardSpec(), 12, output, {}, [])
            errors = []
            worker.failed.connect(errors.append)
            worker.run()
            self.assertEqual(len(errors), 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
