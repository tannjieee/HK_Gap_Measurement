import unittest

import numpy as np

from dual_camera_world_core import Observation, WorldComparison, pose_difference
from world_calibration_core import inverse_transform, reference_world_pose


class WorldComparisonTest(unittest.TestCase):
    def setUp(self):
        self.settings = {"reference_xyz_mm": [126.1, -12.5, 142], "reference_rpy_deg": [0, 0, 0],
                         "sample_count": 3, "pair_tolerance_ms": 80, "stale_s": .5}
        self.model = WorldComparison(self.settings)
        self.world_camera = {
            "hik": reference_world_pose([-200, 20, -300], [5, 8, 0]),
            "gemini": reference_world_pose([200, -20, -500], [-4, -8, 0])}
        self.reference = reference_world_pose()
        self.target = reference_world_pose([200, 100, 300], [10, 20, 30])

    def observation(self, camera, number, now, target=None):
        poses = {0: inverse_transform(self.world_camera[camera]) @ self.reference}
        if target is not None:
            poses[1] = inverse_transform(self.world_camera[camera]) @ target
        return Observation((number,), now, poses, {i: .1 for i in poses})

    def calibrate(self):
        self.model.begin(0)
        for i in range(3):
            for c in self.model.cameras:
                self.model.ingest(c, self.observation(c, i, .1*i), .1*i)
        self.assertEqual(len(self.model.calibrations), 2)

    def test_frozen_extrinsics_independent_target_difference(self):
        self.calibrate()
        self.assertFalse(self.model.statistics)
        for c in self.model.cameras:
            np.testing.assert_allclose(self.model.calibrations[c].world_from_camera, self.world_camera[c], atol=1e-12)
        biased = self.target.copy()
        biased[:3, 3] += [.003, -.004, 0]
        self.model.ingest("hik", self.observation("hik", 10, 1, biased), 1)
        self.model.ingest("gemini", self.observation("gemini", 10, 1.02, self.target), 1.02)
        d = self.model.report(1.02)["current_pair"]["differences"][1]
        np.testing.assert_allclose(d["delta_xyz_mm"], [3, -4, 0], atol=1e-9)
        self.assertAlmostEqual(d["distance_mm"], 5)
        self.assertEqual(self.model.statistics[1].count, 1)
        # Moving the reference later must not refit extrinsics or erase differences.
        before = self.model.calibrations["hik"].world_from_camera.copy()
        moved = self.observation("hik", 11, 1.1)
        moved.poses[0][:3, 3] += [.01, 0, 0]
        self.model.ingest("hik", moved, 1.1)
        self.model.ingest("gemini", self.observation("gemini", 11, 1.1), 1.1)
        np.testing.assert_array_equal(before, self.model.calibrations["hik"].world_from_camera)
        self.assertGreater(self.model.pair["differences"][0]["distance_mm"], 9)

    def test_duplicate_old_and_unpaired_frames_do_not_count(self):
        self.model.begin(0)
        a = self.observation("hik", 1, 0)
        self.model.ingest("hik", a, 0)
        self.model.ingest("hik", a, 0)
        self.model.ingest("gemini", self.observation("gemini", 1, .3), .3)
        self.assertEqual(len(self.model.samples["hik"]), 0)
        self.model.ingest("hik", self.observation("hik", 2, .3), .3)
        self.assertEqual(len(self.model.samples["hik"]), 1)
        self.model.ingest("hik", self.observation("hik", 3, 0), 2)
        self.assertEqual(len(self.model.queues["hik"]), 0)

    def test_stale_comparison_disappears(self):
        self.calibrate()
        for c in self.model.cameras:
            self.model.ingest(c, self.observation(c, 10, 1, self.target), 1)
        self.assertIsNotNone(self.model.report(1)["current_pair"])
        self.assertIsNone(self.model.report(2)["current_pair"])
        self.assertEqual(self.model.world_poses(2), {})

    def test_unstable_reference_rejects_both_calibrations(self):
        self.model.begin(0)
        for i in range(3):
            for c in self.model.cameras:
                o = self.observation(c, i, .1*i)
                if c == "hik":
                    o.poses[0][:3, 3] += [i*.03, 0, 0]
                self.model.ingest(c, o, .1*i)
        self.assertFalse(self.model.calibrations)
        self.assertIn("不稳定", self.model.last_error)

    def test_missing_reference_and_timeout(self):
        self.model.begin(0)
        for c in self.model.cameras:
            self.model.ingest(c, Observation((1,), 0, {}, {}), 0)
        self.assertEqual(len(self.model.samples["hik"]), 0)
        self.model.tick(31)
        self.assertFalse(self.model.collecting)
        self.assertIn("超时", self.model.last_error)

    def test_rotation_difference_is_geodesic(self):
        a = reference_world_pose([0, 0, 0], [0, 0, 179])
        b = reference_world_pose([0, 0, 0], [0, 0, -179])
        self.assertAlmostEqual(pose_difference(a, b)["rotation_difference_deg"], 2, places=8)


if __name__ == "__main__":
    unittest.main()
