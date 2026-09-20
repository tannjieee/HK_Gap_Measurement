import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from apriltag_coordinates import TAG_BASIS_OLD_FROM_NEW
from apriltag_pose_gui import load_pose_config
from gemini335l_apriltag import Processor, checked_intrinsics, frame_document, rvec_quaternion


ROOT = Path(__file__).resolve().parents[1]


def marker_message(tag_id=0):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    generate = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker
    gray = np.full((480, 640), 255, np.uint8)
    gray[140:340, 220:420] = generate(dictionary, tag_id, 200)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return SimpleNamespace(width=640, height=480, encoding="rgb8", step=640*3,
                           data=rgb.tobytes(), header=SimpleNamespace(
                               frame_id="gemini335l_color_optical_frame",
                               stamp=SimpleNamespace(sec=123, nanosec=456)))


class GeminiAprilTagTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "intrinsics.yaml"
        self.path.write_text(yaml.safe_dump({"image_width": 640, "image_height": 480,
            "distortion_model": "plumb_bob", "camera_matrix": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
            "distortion_coefficients": [0]*5}))
        self.config = replace(load_pose_config(ROOT / "config/gemini335l_apriltag.yaml"),
                              intrinsics_file=self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_generated_tag_pose_axis_remap_and_timestamp(self):
        frame = Processor(self.config).process(marker_message())
        self.assertEqual([p.tag_id for p in frame.poses], [0])
        pose = frame.poses[0]
        self.assertAlmostEqual(pose.translation_mm[2], 160, delta=2)
        np.testing.assert_allclose(pose.rotation_matrix,
                                   np.diag([1, -1, -1]) @ TAG_BASIS_OLD_FROM_NEW, atol=0.03)
        self.assertLess(pose.reprojection_error_px, 1)
        document = frame_document(frame)
        self.assertEqual(document["image_stamp"], {"sec": 123, "nanosec": 456})
        self.assertEqual(document["tags"][0]["id"], 0)
        json.dumps(document, allow_nan=False)

    def test_id_filter_and_no_stale_detections(self):
        processor = Processor(self.config)
        self.assertTrue(processor.process(marker_message()).poses)
        self.assertEqual(processor.process(marker_message(1)).poses, [])
        all_ids = Processor(replace(self.config, tag_ids=()))
        self.assertEqual(all_ids.process(marker_message(1)).poses[0].tag_id, 1)

    def test_tag_size_controls_metric_scale(self):
        message = marker_message()
        a = Processor(self.config).process(message).poses[0]
        b = Processor(replace(self.config, tag_size_mm=40)).process(message).poses[0]
        np.testing.assert_allclose(b.translation_mm, a.translation_mm / 2, atol=1e-4)

    def test_geometry_mismatch_refused(self):
        processor = Processor(self.config)
        message = marker_message()
        message.width = 1280
        with self.assertRaisesRegex(ValueError, "不一致"):
            processor.process(message)
        message.width = 640
        message.header.frame_id = "other_camera"
        with self.assertRaisesRegex(ValueError, "坐标系"):
            processor.process(message)

    def test_invalid_intrinsics_refused(self):
        document = yaml.safe_load(self.path.read_text())
        document["camera_matrix"][0][0] = float("nan")
        self.path.write_text(yaml.safe_dump(document))
        with self.assertRaises(ValueError):
            checked_intrinsics(self.path)

    def test_quaternion_preserves_rotation(self):
        for rvec in (np.zeros(3), np.array([0, np.pi, 0]), np.array([0.3, -0.7, 1.2])):
            x, y, z, w = rvec_quaternion(rvec)
            matrix = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                               [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                               [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            np.testing.assert_allclose(matrix, cv2.Rodrigues(rvec)[0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
