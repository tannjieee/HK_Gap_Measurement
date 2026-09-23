"""Camera-free GUI workflow, using real decoding of two rendered cube frames.

QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -p test_cube_joint_gui.py
Set HK_JOINT_GUI_ARTIFACTS to retain snapshots and a GUI screenshot.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cube_pose_app as application
from cube_pose_core import CubeGeometry, CubePoseEstimator, NORMALS, draw_result, load_intrinsics
from PyQt5.QtCore import QPoint, QPointF, QTimer, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def transform(angles, translation):
    t = np.eye(4); t[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    t[:3, 3] = translation
    return t


def render_cubes(transforms, k):
    geometry = CubeGeometry()
    canvas = np.full((1024, 1280, 3), 215, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for cube, t in enumerate(transforms, 1):
        faces = []
        for face, normal in enumerate(NORMALS):
            center = t[:3, :3]@(normal*geometry.side_m/2)+t[:3, 3]
            if np.dot(t[:3, :3]@normal, center) >= 0: continue
            faces.append((center[2], face))
        for _, face in sorted(faces, reverse=True):
            marker_id = (cube-1)*5+face
            generate = getattr(cv2.aruco, 'generateImageMarker', None) or cv2.aruco.drawMarker
            texture = np.full((200, 200), 255, np.uint8)
            texture[20:180, 20:180] = generate(dictionary, marker_id, 160)
            corners = geometry.corners(face)
            center = NORMALS[face]*geometry.side_m/2
            corners = center+(corners-center)*geometry.side_m/geometry.tag_size_m
            pixels = cv2.projectPoints(corners, cv2.Rodrigues(t[:3, :3])[0], t[:3, 3], k, None)[0].reshape(4, 2)
            homography = cv2.getPerspectiveTransform(np.float32([[0,0],[199,0],[199,199],[0,199]]), pixels.astype(np.float32))
            warped = cv2.warpPerspective(texture, homography, (1280, 1024), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full_like(texture, 255), homography, (1280, 1024), flags=cv2.INTER_NEAREST)
            canvas[mask > 0] = warped[mask > 0, None]
    cv2.putText(canvas, 'SYNTHETIC SOFTWARE CHECK - NOT A REAL CAMERA MEASUREMENT',
                (30, 980), cv2.FONT_HERSHEY_SIMPLEX, .7, (50, 50, 50), 2)
    return canvas


class JointGuiTests(unittest.TestCase):
    def test_click_capture_resume_compute_save_and_reject_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            artifact_dir = Path(os.environ.get('HK_JOINT_GUI_ARTIFACTS', temporary))
            artifact_dir.mkdir(parents=True, exist_ok=True)
            k = np.array([[1220., 0, 632.], [0, 1210., 508.], [0, 0, 1.]])
            calibration = folder/'synthetic.json'
            calibration.write_text(json.dumps(dict(camera_matrix=k.tolist(), distortion_coefficients=[0]*5,
                image_width=1280, image_height=1024, camera_name='synthetic_validation_only')))
            a = transform((15, -40, 12), (-.025, -.004, .19))
            b = transform((-20, -42, -18), (.025, .004, .195))
            common = transform((2, -4, 3), (.003, -.002, 0))
            common[:3, 3] += np.array([0, 0, .19])-common[:3, :3]@np.array([0, 0, .19])
            relative = np.linalg.inv(a)@b
            moved = relative.copy()
            moved[:3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()@relative[:3, :3]
            final_a, final_b = common@a, common@a@moved
            initial_image, final_image = render_cubes((a, b), k), render_cubes((final_a, final_b), k)
            image_path = folder/'initial.png'; cv2.imwrite(str(image_path), initial_image)
            camera, metadata = load_intrinsics(calibration)
            estimator = CubePoseEstimator(camera)
            final_result = estimator.detect(final_image)
            self.assertTrue(all(c['pose_valid'] for c in final_result['cubes']), final_result['cubes'])
            exceptions = []
            original_exec = QApplication.exec_
            app_ref = QApplication.instance() or QApplication([])

            def checked_exec(app):
                deadline = time.monotonic()+8

                def exercise():
                    window = next(w for w in app.topLevelWidgets() if hasattr(w, 'measurement'))
                    try:
                        if window.worker is None or window.worker.latest is None:
                            if time.monotonic() > deadline: raise AssertionError('GUI worker timeout')
                            QTimer.singleShot(50, exercise); return
                        window.refresh(); window.timer.stop()
                        initial = window.packet
                        self.assertTrue(all(c['pose_valid'] for c in initial[4]['cubes']))
                        window.resize(1480, 950); app.processEvents()
                        rect = window.video.image_rect()
                        self.assertIsNone(window.video.image_position(QPointF(-1, -1)))
                        # Explicitly exercise letterboxing at a different aspect ratio.
                        if rect.top() > 1:
                            self.assertIsNone(window.video.image_position(QPointF(rect.center().x(), 0)))
                        elif rect.left() > 1:
                            self.assertIsNone(window.video.image_position(QPointF(0, rect.center().y())))

                        def click_cube(cube):
                            obs = next(o for o in window.packet[4]['observations'] if o['tag_id']//5+1 == cube)
                            xy = np.mean(obs['corners_px'], axis=0)
                            r = window.video.image_rect()
                            p = QPoint(round(r.x()+xy[0]*r.width()/1280), round(r.y()+xy[1]*r.height()/1024))
                            QTest.mouseClick(window.video, Qt.LeftButton, pos=p)

                        click_cube(1); click_cube(2)
                        self.assertEqual((window.measurement.parent, window.measurement.child), (1, 2))
                        QTest.mouseClick(window.initial_button, Qt.LeftButton)
                        self.assertIsNotNone(window.measurement.initial, window.statusBar().currentMessage())
                        self.assertEqual(window.frozen, '初始帧')
                        self.assertFalse(window.final_button.isEnabled())
                        QTest.mouseClick(window.resume_button, Qt.LeftButton)
                        self.assertIsNone(window.frozen)
                        # Still the same offline frame: final capture must be rejected.
                        QTest.mouseClick(window.final_button, Qt.LeftButton)
                        self.assertIsNone(window.measurement.result)
                        final_result.update(host_receive_time_ns=initial[4]['host_receive_time_ns']+1_000_000_000,
                            frame_number=1, source=initial[4]['source'], calibration=metadata,
                            processing_ms=1., pipeline_fps=1.)
                        final_annotated = draw_result(final_image, final_result, camera, CubeGeometry())
                        final_packet = (time.monotonic(), final_image, final_annotated, final_annotated, final_result)
                        with window.worker.lock: window.worker.latest = final_packet
                        window.refresh()
                        QTest.mouseClick(window.final_button, Qt.LeftButton)
                        self.assertIsNotNone(window.measurement.result, window.statusBar().currentMessage())
                        angle = window.measurement.result['angle_deg']
                        self.assertAlmostEqual(angle, 30., delta=.5)
                        self.assertEqual(window.frozen, '结束帧')
                        saved = json.loads((window.measurement_folder/'measurement.json').read_text())
                        self.assertEqual(saved['parent_cube'], 1); self.assertEqual(saved['child_cube'], 2)
                        self.assertAlmostEqual(saved['measurement']['angle_deg'], angle)
                        for stage in ('initial', 'final'):
                            for name in ('raw.png', 'detected.png', 'poses.json', 'measurement.json'):
                                self.assertTrue((window.measurement_folder/saved['snapshots'][stage]/name).is_file())
                        app.processEvents(); window.grab().save(str(artifact_dir/'gui.png'))
                        application.save_json(artifact_dir/'validation.json', dict(status='PASS',
                            evidence='Synthetic raster tags, real decoding/PnP, Qt image clicks and saved two-frame measurement',
                            expected_angle_deg=30, measured_angle_deg=angle, physical_accuracy_verified=False,
                            measurement_directory=str(window.measurement_folder)))
                        QTest.mouseClick(window.view_initial, Qt.LeftButton)
                        self.assertIs(window.packet, window.initial_packet)
                        self.assertEqual(window.frozen, '初始帧')
                        QTest.mouseClick(window.view_final, Qt.LeftButton)
                        self.assertIs(window.packet, window.final_packet)
                        QTest.mouseClick(window.swap_button, Qt.LeftButton)
                        self.assertEqual((window.measurement.parent, window.measurement.child), (2, 1))
                        self.assertIsNone(window.measurement.initial); self.assertIsNone(window.measurement.result)
                        # Mark displayed and worker packets old, and switch out of static-image mode.
                        window.worker.args.image = None
                        stale = (time.monotonic()-2, *final_packet[1:])
                        with window.worker.lock: window.worker.latest = stale
                        window.packet = stale; window.refresh(); window.capture_angle('initial')
                        self.assertIsNone(window.measurement.initial)
                        self.assertIn('已过期', window.statusBar().currentMessage())
                    except BaseException:
                        exceptions.append(traceback.format_exc())
                    finally:
                        if window.worker is not None and window.worker.latest is not None:
                            window.close(); app.quit()

                QTimer.singleShot(100, exercise)
                return original_exec()

            with patch.object(QApplication, 'exec_', checked_exec), patch.object(application, 'RUNS', artifact_dir):
                application.main([
                    '--image', str(image_path),
                    '--intrinsics', str(calibration),
                    '--cubes', '1', '2',
                ])
            if exceptions: self.fail('\n'.join(exceptions))


if __name__ == '__main__': unittest.main()
