"""Two-frame relative rotation measurement, independent of camera and Qt."""
from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cube_pose_core import VALID_STATUSES, pose_fields


def cube_hulls(result):
    """Only detected faces contribute to a cube's clickable image area."""
    allowed = {c['cube'] for c in result['cubes']}
    groups = {}
    for observation in result['observations']:
        cube = observation['tag_id'] // 5 + 1
        if cube in allowed:
            groups.setdefault(cube, []).extend(observation['corners_px'])
    return {cube: cv2.convexHull(np.asarray(points, np.float32))
            for cube, points in groups.items()}


def cube_at_pixel(result, x, y):
    hits = [cube for cube, hull in cube_hulls(result).items()
            if cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0]
    # Never guess between overlapping cubes or select empty image space.
    return hits[0] if len(hits) == 1 else None


def draw_roles(image, result, parent, child, coordinate_scale=(1.0, 1.0)):
    output = image.copy()
    scale = np.asarray(coordinate_scale, dtype=float)
    for cube, hull in cube_hulls(result).items():
        if cube not in (parent, child):
            continue
        role, color = ('PARENT', (255, 200, 40)) if cube == parent else ('CHILD', (200, 80, 255))
        points = np.rint(hull * scale).astype(np.int32)
        cv2.polylines(output, [points], True, color, 3, cv2.LINE_AA)
        x, y = np.min(points.reshape(-1, 2), axis=0)
        label = f'{role} {cube:02d}'
        origin = (max(0, int(x)), max(20, int(y)-12))
        cv2.putText(output, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)
    return output


def _transform(cube):
    if not cube['pose_valid'] or cube['status'] not in VALID_STATUSES or not cube.get('pose'):
        raise ValueError(f"方块 {cube['cube']:02d} 当前位姿无效：{cube['status']}")
    t = np.asarray(cube['pose']['transform'], float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-6)):
        raise ValueError('方块位姿不是有效刚体变换')
    return t


def relative_sample(result, parent, child):
    if parent is None or child is None or parent == child:
        raise ValueError('请先点选两个不同的方块作为父 Link 和子 Link')
    if not result.get('calibrated') or not (result.get('calibration') or {}).get('sha256'):
        raise ValueError('需要当前相机的有效内参，且图像分辨率一致')
    by_id = {c['cube']: c for c in result['cubes']}
    if parent not in by_id or child not in by_id:
        raise ValueError('当前帧缺少所选父／子方块')
    a, b = by_id[parent], by_id[child]
    relative = np.linalg.inv(_transform(a)) @ _transform(b)
    return deepcopy(dict(
        parent_cube=parent, child_cube=child,
        frame_number=result['frame_number'], host_receive_time_ns=result['host_receive_time_ns'],
        source=result['source'], calibration=result['calibration'], image_size=result['image_size'],
        cube_side_m=result['cube_side_m'], tag_black_border_m=result['tag_black_border_m'],
        parent=a, child=b, parent_T_child=pose_fields(relative)))


def compare_samples(initial, final):
    """R_delta is expressed in the parent cube's axes, not the camera's."""
    for key in ('parent_cube', 'child_cube', 'source', 'image_size', 'cube_side_m', 'tag_black_border_m'):
        if initial[key] != final[key]:
            raise ValueError(f'两帧的 {key} 不一致，请重新定格初始帧')
    if initial['calibration']['sha256'] != final['calibration']['sha256']:
        raise ValueError('两帧使用了不同内参，请重新定格初始帧')
    if initial['host_receive_time_ns'] >= final['host_receive_time_ns']:
        raise ValueError('结束帧必须晚于初始帧，请恢复实时画面后重新定格')
    t0 = np.asarray(initial['parent_T_child']['transform'])
    t1 = np.asarray(final['parent_T_child']['transform'])
    r_delta = t1[:3, :3] @ t0[:3, :3].T
    rotvec = Rotation.from_matrix(r_delta).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    delta_position = (t1[:3, 3] - t0[:3, 3]) * 1000
    return dict(
        angle_deg=float(np.rad2deg(angle)), angle_range_deg=[0, 180], signed=False,
        rotation_axis_parent=(rotvec / angle).tolist() if angle > 1e-7 else None,
        axis_convention='Parent cube coordinates; undefined at zero, sign ambiguous at 180 degrees',
        rotation_matrix_parent=r_delta.tolist(),
        child_center_displacement_parent_mm=delta_position.tolist(),
        child_center_displacement_mm=float(np.linalg.norm(delta_position)),
        formula='R_delta = R_parent_child_final @ R_parent_child_initial.T',
        interpretation='Unsigned principal relative rotation, not Euler subtraction or accumulated travel',
        single_face_used=any(s[role]['status'] == 'OK_SINGLE_FACE'
                             for s in (initial, final) for role in ('parent', 'child')))


class JointMeasurement:
    def __init__(self):
        self.parent = self.child = None
        self.clear()

    def clear(self):
        self.initial = self.final = self.result = None

    def select(self, role, cube):
        if role not in ('parent', 'child'):
            raise ValueError('Unknown role')
        other = self.child if role == 'parent' else self.parent
        if cube == other:
            raise ValueError('父 Link 和子 Link 不能选同一个方块')
        if getattr(self, role) != cube:
            setattr(self, role, cube)
            self.clear()

    def capture_initial(self, result):
        sample = relative_sample(result, self.parent, self.child)
        self.clear()
        self.initial = sample

    def capture_final(self, result):
        if self.initial is None:
            raise ValueError('请先定格初始帧')
        final = relative_sample(result, self.parent, self.child)
        measurement = compare_samples(self.initial, final)
        self.final, self.result = final, measurement

    def as_dict(self):
        return deepcopy(dict(schema_version=1, parent_cube=self.parent, child_cube=self.child,
                             initial=self.initial, final=self.final, measurement=self.result))
