"""Known-angle, arbitrary-mounting and data-integrity regression checks."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cube_joint_measurement import JointMeasurement, cube_at_pixel, relative_sample
from cube_pose_app import packet_is_fresh
from cube_pose_core import pose_fields


def transform(angles=(0, 0, 0), translation=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    t[:3, 3] = translation
    return t


def frame(parent, child, number=1):
    return dict(calibrated=True, calibration={'sha256': 'same-calibration'},
                source={'type': 'synthetic'}, image_size=[1280, 1024],
                frame_number=number, host_receive_time_ns=number*1_000_000,
                cube_side_m=.016, tag_black_border_m=.0128, observations=[],
                cubes=[dict(cube=i, pose_valid=True, status='OK_MULTI_FACE', pose=pose_fields(t))
                       for i, t in ((1, parent), (2, child))])


class JointMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.m = JointMeasurement()
        self.m.select('parent', 1)
        self.m.select('child', 2)

    def test_arbitrary_mounts_and_common_motion_cancel(self):
        rng = np.random.default_rng(103)
        for angle in (0, 1, 30, 90, 179.9, 180, 270, -42):
            for _ in range(8):
                mount_parent = transform(rng.uniform(-180, 180, 3), rng.normal(0, .01, 3))
                mount_child = transform(rng.uniform(-180, 180, 3), rng.normal(0, .01, 3))
                world0 = transform(rng.uniform(-180, 180, 3), rng.normal(0, .1, 3))
                world1 = transform(rng.uniform(-180, 180, 3), rng.normal(0, .1, 3))
                joint0 = transform((0, 0, 17), (.025, .01, 0))
                joint1 = transform((0, 0, 17+angle), (.025, .01, 0))
                self.m.capture_initial(frame(world0@mount_parent, world0@joint0@mount_child))
                self.m.capture_final(frame(world1@mount_parent, world1@joint1@mount_child, 2))
                expected = abs((angle+180) % 360 - 180)
                self.assertAlmostEqual(self.m.result['angle_deg'], expected, places=9)
                axis = self.m.result['rotation_axis_parent']
                if expected == 0:
                    self.assertIsNone(axis)
                elif 0 < angle < 180:
                    np.testing.assert_allclose(axis, mount_parent[:3, :3].T@[0, 0, 1], atol=1e-10)

    def test_translation_only_is_not_rotation(self):
        self.m.capture_initial(frame(np.eye(4), transform((5, 6, 7))))
        self.m.capture_final(frame(np.eye(4), transform((5, 6, 7), (.03, .04, 0)), 2))
        self.assertAlmostEqual(self.m.result['angle_deg'], 0)
        self.assertAlmostEqual(self.m.result['child_center_displacement_mm'], 50)

    def test_role_reversal_preserves_angle(self):
        a = frame(transform((13, -20, 8)), transform((25, 11, 42)))
        b = frame(transform((-20, 17, 0)), transform((60, 28, -15)), 2)
        self.m.capture_initial(a); self.m.capture_final(b)
        expected = self.m.result['angle_deg']
        m = JointMeasurement(); m.select('parent', 2); m.select('child', 1)
        m.capture_initial(a); m.capture_final(b)
        self.assertAlmostEqual(m.result['angle_deg'], expected, places=10)

    def test_initial_is_immutable_and_new_initial_resets_final(self):
        a = frame(np.eye(4), np.eye(4))
        self.m.capture_initial(a)
        a['cubes'][0]['pose']['transform'][0][0] = 17
        self.m.capture_final(frame(np.eye(4), transform((0, 0, 35)), 2))
        self.assertAlmostEqual(self.m.result['angle_deg'], 35)
        export = self.m.as_dict(); export['initial']['parent_cube'] = 7
        self.assertEqual(self.m.initial['parent_cube'], 1)
        self.m.capture_initial(frame(np.eye(4), np.eye(4), 3))
        self.assertIsNone(self.m.final); self.assertIsNone(self.m.result)

    def test_invalid_observation_cannot_replace_initial_or_final(self):
        valid = frame(np.eye(4), np.eye(4))
        self.m.capture_initial(valid)
        for status in ('AMBIGUOUS', 'NOT_VISIBLE', 'DUPLICATE_ID', 'INCONSISTENT_FACES'):
            bad = frame(np.eye(4), np.eye(4), 2)
            bad['cubes'][1].update(pose_valid=False, status=status)
            with self.assertRaises(ValueError): self.m.capture_final(bad)
            with self.assertRaises(ValueError): self.m.capture_initial(bad)
            self.assertEqual(self.m.initial['frame_number'], 1)
            self.assertIsNone(self.m.result)

    def test_missing_calibration_and_invalid_transform(self):
        for calibrated, calibration in ((False, None), (True, None), (True, {})):
            bad = frame(np.eye(4), np.eye(4)); bad.update(calibrated=calibrated, calibration=calibration)
            with self.assertRaises(ValueError): self.m.capture_initial(bad)
        bad = frame(np.eye(4), np.eye(4)); bad['cubes'][0]['pose']['transform'][0][0] = 2
        with self.assertRaises(ValueError): self.m.capture_initial(bad)
        bad['cubes'] = []
        with self.assertRaises(ValueError): self.m.capture_initial(bad)

    def test_mixed_calibration_scale_source_resolution_and_same_frame_rejected(self):
        valid = frame(np.eye(4), np.eye(4)); self.m.capture_initial(valid)
        with self.assertRaises(ValueError): self.m.capture_final(valid)
        changes = dict(calibration={'sha256': 'new-calibration'}, cube_side_m=.014,
                       tag_black_border_m=.010, source={'type': 'other'}, image_size=[640, 512])
        for key, value in changes.items():
            bad = frame(np.eye(4), np.eye(4), 2); bad[key] = value
            with self.assertRaises(ValueError): self.m.capture_final(bad)
        self.assertIsNone(self.m.result)

    def test_selection_changes_clear_baseline_and_same_cube_is_rejected(self):
        self.m.capture_initial(frame(np.eye(4), np.eye(4)))
        self.m.select('parent', 1)
        self.assertIsNotNone(self.m.initial)
        with self.assertRaises(ValueError): self.m.select('child', 1)
        self.m.select('child', 3)
        self.assertIsNone(self.m.initial)
        with self.assertRaises(ValueError): self.m.capture_final(frame(np.eye(4), np.eye(4), 2))
        with self.assertRaises(ValueError): relative_sample(frame(np.eye(4), np.eye(4)), 1, 1)

    def test_single_face_quality_is_retained(self):
        a = frame(np.eye(4), np.eye(4)); a['cubes'][0]['status'] = 'OK_SINGLE_FACE'
        self.m.capture_initial(a); self.m.capture_final(frame(np.eye(4), np.eye(4), 2))
        self.assertTrue(self.m.result['single_face_used'])

    def test_clicks_follow_tag_ids_and_ignore_empty_or_overlap(self):
        result = frame(np.eye(4), np.eye(4))
        result['observations'] = [dict(tag_id=4, corners_px=[[10, 10], [30, 10], [30, 30], [10, 30]]),
                                  dict(tag_id=7, corners_px=[[20, 10], [40, 10], [40, 30], [20, 30]]),
                                  dict(tag_id=74, corners_px=[[80, 80], [90, 80], [90, 90], [80, 90]])]
        self.assertEqual(cube_at_pixel(result, 15, 20), 1)
        self.assertEqual(cube_at_pixel(result, 35, 20), 2)
        for point in ((25, 20), (1, 1), (85, 85)):
            self.assertIsNone(cube_at_pixel(result, *point))

    def test_freshness_allows_normal_full_resolution_pipeline_jitter(self):
        result = {
            'pipeline_fps': 3.0,
            'processing_ms': 350.0,
            'host_receive_time_ns': int(99.1e9),
        }
        packet = (99.3, None, None, None, result)
        self.assertTrue(packet_is_fresh(
            packet, True, monotonic_now=100.0, wall_time_ns=int(100e9)
        ))
        self.assertFalse(packet_is_fresh(
            packet, False, monotonic_now=100.0, wall_time_ns=int(100e9)
        ))
        self.assertFalse(packet_is_fresh(
            packet, True, monotonic_now=101.0, wall_time_ns=int(101e9)
        ))


if __name__ == '__main__': unittest.main()
