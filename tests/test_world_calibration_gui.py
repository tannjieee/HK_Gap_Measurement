from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from world_calibration_gui import WorldCalibrationWindow
from apriltag_pose_gui import PoseFrame, PoseRecord, _rotation_to_euler_xyz
from world_calibration_core import WorldCalibration
from PyQt5.QtWidgets import QApplication
import cv2
import numpy as np


class WorldCalibrationGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = Path(__file__).parents[1] / "config" / "world_calibration.yaml"
        # Exercise real widgets, frame handling and persistence without opening
        # a physical camera or starting a native detector / ROS process.
        with (
            patch("apriltag_pose_gui.PoseEstimator", return_value=Mock(backend_name="test-opencv")),
            patch.object(WorldCalibrationWindow, "refresh_devices"),
            patch.object(WorldCalibrationWindow, "_load_optional_feature_panel"),
        ):
            self.window = WorldCalibrationWindow(config, auto_connect=False)
        self.window.output_dir = Path(self.directory.name)
        self.window.sample_count.setValue(3)
        self.window.capture_thread = Mock()
        self.window.capture_thread.isRunning.return_value = True
        self.rotation, _ = cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))

    def tearDown(self):
        self.window.capture_thread = None
        self.window.world_timer.stop()
        self.window.close()

    def frame(self, number, *, tag_id=0, error=0.2, missing=False):
        pose = PoseRecord(
            tag_id=tag_id,
            corners=np.array([[200, 100], [400, 100], [400, 300], [200, 300]], dtype=float),
            rvec=cv2.Rodrigues(self.rotation)[0],
            tvec=np.array([[0.04], [-0.03], [0.6]]),
            rotation_matrix=self.rotation,
            euler_xyz_deg=_rotation_to_euler_xyz(self.rotation),
            translation_mm=np.array([40., -30., 600.]),
            reprojection_error_px=error,
            family="tag36h11",
        )
        return PoseFrame(np.zeros((480, 640, 3), dtype=np.uint8), [] if missing else [pose], {}, number, 1., (640, 480))

    def calibrate(self):
        self.window.begin_calibration()
        for number in range(3):
            self.window._display_frame(self.frame(number))
        self.assertIsNotNone(self.window.calibration)

    def test_collect_save_load_and_display_world_coordinates(self):
        self.calibrate()
        files = list(self.window.output_dir.glob("*.json"))
        self.assertEqual(len(files), 1)
        saved = WorldCalibration.load(files[0])
        self.assertEqual(saved.sample_count, 3)
        self.window._update_world_live(self.frame(4))
        text = self.window.world_live.toPlainText()
        self.assertIn("X=+126.1", text)
        self.assertIn("Y=-12.5", text)
        self.assertIn("Z=+142.0", text)
        np.testing.assert_allclose(saved.world_from_camera @ saved.camera_from_reference, saved.world_from_reference, atol=1e-12)
        self.window._invalidate("test")
        with patch("world_calibration_gui.QFileDialog.getOpenFileName", return_value=(str(files[0]), "")):
            self.window.load_world_calibration()
        self.assertIsNotNone(self.window.calibration)
        self.assertIn("已加载", self.window.world_status.text())

    def test_missing_wrong_duplicate_and_high_error_frames_are_not_samples(self):
        self.window.begin_calibration()
        self.window._display_frame(self.frame(0, missing=True))
        self.window._display_frame(self.frame(1, tag_id=1))
        self.window._display_frame(self.frame(2, error=9))
        self.window._display_frame(self.frame(3))
        self.window._display_frame(self.frame(3))
        self.assertEqual(len(self.window._samples), 1)
        self.assertIsNone(self.window.calibration)
        self.assertEqual(list(self.window.output_dir.glob("*.json")), [])
        self.window._deadline = 0
        self.window._check_timeout()
        self.assertIsNone(self.window._samples)
        self.assertIn("超时", self.window.world_status.text())

    def test_sampling_rejects_wrong_tag_size_and_parameter_changes(self):
        self.window.pose_config.tag_size_mm = 48.35
        self.window.begin_calibration()
        self.assertIsNone(self.window._samples)
        self.assertIn("80 mm", self.window.world_status.text())
        self.window.pose_config.tag_size_mm = 80
        self.window.begin_calibration()
        self.window.pose_config.tag_size_mm = 40
        self.window._display_frame(self.frame(0))
        self.assertIsNone(self.window._samples)
        self.assertIn("参数发生改变", self.window.world_status.text())

    def test_completed_calibration_invalidated_by_backend_or_world_pose_change(self):
        self.calibrate()
        self.window.estimator.backend_name = "different-backend"
        self.window._display_frame(self.frame(4))
        self.assertIsNone(self.window.calibration)
        self.calibrate()
        self.window.world_xyz[0].setValue(150)
        self.assertIsNone(self.window.calibration)

    def test_result_remains_fixed_when_reference_disappears(self):
        self.calibrate()
        before = self.window.calibration.world_from_camera.copy()
        self.window._display_frame(self.frame(4, missing=True))
        self.window._display_frame(self.frame(5, tag_id=1))
        np.testing.assert_array_equal(self.window.calibration.world_from_camera, before)
        self.window._update_world_live(self.frame(6, tag_id=1))
        self.assertIn("ID 1", self.window.world_live.toPlainText())


if __name__ == "__main__":
    unittest.main()
