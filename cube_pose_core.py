"""Five-face AprilTag cube fitting in the printed cube's original coordinate frame.

T_camera_cube maps cube-centred coordinates to OpenCV camera coordinates.
No application-tag basis remapping, temporal smoothing, or stale-pose reuse.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from apriltag_pose_core import CameraIntrinsics


NORMALS = np.array([[-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]], float)
FACE_NAMES = ('center', 'left', 'right', 'top', 'bottom')
VALID_STATUSES = ('OK_MULTI_FACE', 'OK_SINGLE_FACE')


@dataclass(frozen=True)
class CubeGeometry:
    side_m: float = .016
    tag_size_m: float = .0128

    def __post_init__(self):
        if not (np.isfinite(self.side_m) and np.isfinite(self.tag_size_m)
                and 0 < self.tag_size_m < self.side_m):
            raise ValueError('Require 0 < tag black-border size < cube side')

    @property
    def square(self):
        return np.array([[-1,1,0], [1,1,0], [1,-1,0], [-1,-1,0]], float)*self.tag_size_m/2

    def cube_T_tag(self, face):
        n = NORMALS[face]
        u = np.cross([0,0,1] if abs(n[2]) < .5 else [0,1,0], n)
        u /= np.linalg.norm(u)
        t = np.eye(4)
        t[:3,:3] = np.column_stack([u, np.cross(n,u), n])
        t[:3,3] = n*self.side_m/2
        return t

    def corners(self, face):
        t = self.cube_T_tag(face)
        return self.square@t[:3,:3].T+t[:3,3]


def load_intrinsics(path, *, expected_serial=None):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    data = json.loads(raw) if path.suffix.lower()=='.json' else yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError('Camera calibration must be a JSON/YAML mapping')
    camera = CameraIntrinsics.from_mapping(data, source_path=str(path))
    k = camera.camera_matrix
    if k[0,0] <= 0 or k[1,1] <= 0 or not np.allclose(k[2], [0,0,1]) or abs(k[0,1]) > 1e-8:
        raise ValueError('Invalid pinhole camera matrix')
    if camera.image_size is None:
        raise ValueError('Calibration must contain image_width/image_height')
    if camera.distortion_model not in ('plumb_bob','rational_polynomial','opencv','pinhole'):
        raise ValueError('Only OpenCV pinhole distortion is supported, not fisheye')
    name = str(data.get('camera_name',''))
    serial = str(data.get('serial',data.get('camera_serial','')))
    if expected_serial and ((serial and serial != expected_serial)
                            or (name.startswith('MV-') and expected_serial not in name)):
        raise ValueError(f'Calibration belongs to a different camera: {name or serial}')
    return camera, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                        camera_name=name, serial=serial, image_size=list(camera.image_size))


def pose_fields(transform):
    t = np.asarray(transform, float)
    rotation = Rotation.from_matrix(t[:3,:3])
    # Explicit convention: fixed-axis XYZ, R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    r = t[:3,:3]
    pitch = np.arctan2(-r[2,0], np.hypot(r[0,0],r[1,0]))
    if abs(np.cos(pitch)) > 1e-8:
        roll, yaw = np.arctan2(r[2,1],r[2,2]), np.arctan2(r[1,0],r[0,0])
    else:
        roll, yaw = np.arctan2(-r[1,2],r[1,1]), 0.
    return dict(translation_m=t[:3,3].tolist(), translation_mm=(t[:3,3]*1000).tolist(),
                quaternion_xyzw=rotation.as_quat().tolist(),
                rpy_xyz_deg=np.rad2deg([roll,pitch,yaw]).tolist(),
                rotation_matrix=t[:3,:3].tolist(), transform=t.tolist())


class CubePoseEstimator:
    def __init__(self, camera=None, cube_ids=(1,2), geometry=None,
                 max_corner_error_px=3., ambiguity_gap_px=.15):
        self.camera = camera
        self.cube_ids = tuple(cube_ids)
        if (not self.cube_ids or len(set(self.cube_ids)) != len(self.cube_ids)
                or any(type(x) is not int or not 1 <= x <= 15 for x in self.cube_ids)):
            raise ValueError('Cube numbers must be distinct integers from 1 to 15')
        self.geometry = geometry or CubeGeometry()
        if not np.isfinite(max_corner_error_px) or max_corner_error_px <= 0:
            raise ValueError('Invalid reprojection threshold')
        self.max_error = float(max_corner_error_px)
        self.ambiguity_gap = float(ambiguity_gap_px)
        aruco = cv2.aruco
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        # OpenCV 4.6 exposes an unusable bare DetectorParameters constructor;
        # its factory must be used when present, before setting any attributes.
        self.parameters = aruco.DetectorParameters_create() if hasattr(aruco,'DetectorParameters_create') else aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        self.parameters.aprilTagQuadDecimate = 1.0
        self.detector = aruco.ArucoDetector(self.dictionary,self.parameters) if hasattr(aruco,'ArucoDetector') else None

    def detect(self, image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        if gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError('Expected an 8-bit grayscale or BGR image')
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray,self.dictionary,parameters=self.parameters)
        observations = [] if ids is None else [dict(tag_id=int(i),corners_px=np.asarray(p).reshape(4,2).tolist())
                                               for p,i in zip(corners,ids.ravel())]
        return self.estimate(observations, (gray.shape[1],gray.shape[0]))

    def estimate(self, observations, image_size):
        clean = []
        for item in observations:
            points = np.asarray(item['corners_px'],float)
            if points.shape != (4,2) or not np.isfinite(points).all():
                continue
            if not cv2.isContourConvex(points.astype(np.float32)) or abs(cv2.contourArea(points.astype(np.float32))) < 4:
                continue
            clean.append(dict(tag_id=int(item['tag_id']),corners_px=points.tolist()))
        calibrated = self.camera is not None and tuple(image_size)==self.camera.image_size
        cubes = []
        for cube in self.cube_ids:
            first = (cube-1)*5
            selected = [x for x in clean if first <= x['tag_id'] < first+5]
            base = dict(cube=cube, detected_ids=[x['tag_id'] for x in selected],
                        used_ids=[], rejected_ids=[], pose_valid=False, pose=None)
            if not selected:
                cubes.append(dict(base,status='NOT_VISIBLE'))
            elif not calibrated:
                cubes.append(dict(base,status='UNCALIBRATED' if self.camera is None else 'IMAGE_SIZE_MISMATCH'))
            elif any(n>1 for n in Counter(x['tag_id'] for x in selected).values()):
                cubes.append(dict(base,status='DUPLICATE_ID'))
            else:
                cubes.append(self._fit(base,selected))
        relative = None
        relative_pair = None
        by_id = {c['cube']:c for c in cubes}
        if len(self.cube_ids) >= 2:
            first_cube, second_cube = self.cube_ids[:2]
            if all(
                i in by_id and by_id[i]['pose_valid']
                for i in (first_cube, second_cube)
            ):
                relative = pose_fields(
                    np.linalg.inv(
                        np.asarray(by_id[first_cube]['pose']['transform'])
                    )
                    @ np.asarray(by_id[second_cube]['pose']['transform'])
                )
                relative_pair = dict(
                    from_cube=first_cube,
                    to_cube=second_cube,
                    transform=relative,
                )
        legacy_relative = (
            relative if tuple(self.cube_ids[:2]) == (1, 2) else None
        )
        return dict(frame_convention='T_camera_cube: cube centre -> camera; camera +X right, +Y down, +Z forward',
                    cube_convention='Origin at cube centre; +X blank attachment face; +Y left arm; +Z top arm',
                    rpy_convention='fixed-axis XYZ in degrees; R=Rz(yaw) Ry(pitch) Rx(roll)',
                    image_size=list(image_size), calibrated=calibrated, cubes=cubes,
                    observations=clean, cube_pair_relative=relative_pair,
                    cube01_T_cube02=legacy_relative,
                    cube_side_m=self.geometry.side_m, tag_black_border_m=self.geometry.tag_size_m)

    def _fit(self, base, observations):
        k, d = self.camera.camera_matrix, self.camera.distortion_coefficients
        faces = np.array([o['tag_id']%5 for o in observations])
        objects = np.array([self.geometry.corners(f) for f in faces])
        pixels = np.array([o['corners_px'] for o in observations])
        seeds = []
        for face, image_points in zip(faces,pixels):
            local_seeds = []
            try:
                result = cv2.solvePnPGeneric(self.geometry.square,image_points,k,d,flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if result[0]:
                    local_seeds.extend(zip(result[1],result[2]))
            except cv2.error:
                pass
            # Always run an independent planar fit, including IPPE's exact
            # fronto-parallel degeneracy; no previous-frame pose is involved.
            try:
                ok, rv, tv = cv2.solvePnP(self.geometry.square,image_points,k,d,flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    local_seeds.append((rv,tv))
            except cv2.error:
                pass
            for rv,tv in local_seeds:
                if not np.isfinite(rv).all() or not np.isfinite(tv).all():
                    continue
                t = np.eye(4); t[:3,:3] = cv2.Rodrigues(rv)[0]; t[:3,3] = tv.ravel()
                seeds.append(t@np.linalg.inv(self.geometry.cube_T_tag(face)))
        if len(faces)>1:
            try:
                result = cv2.solvePnPGeneric(objects.reshape(-1,3),pixels.reshape(-1,2),k,d,flags=cv2.SOLVEPNP_SQPNP)
                if result[0]:
                    for rv,tv in zip(result[1],result[2]):
                        t = np.eye(4); t[:3,:3]=cv2.Rodrigues(rv)[0]; t[:3,3]=tv.ravel(); seeds.append(t)
            except cv2.error:
                pass

        def evaluate(t):
            if not np.isfinite(t).all():
                return None
            points = objects@t[:3,:3].T+t[:3,3]
            rv = cv2.Rodrigues(t[:3,:3])[0]
            projected = cv2.projectPoints(objects.reshape(-1,3),rv,t[:3,3],k,d)[0].reshape(-1,4,2)
            errors = np.linalg.norm(projected-pixels,axis=2)
            outward = NORMALS[faces]@t[:3,:3].T
            front = np.sum(outward*points.mean(axis=1),axis=1)<-1e-8
            good = (points[:,:,2].min(axis=1)>.001) & front & (errors.max(axis=1)<=self.max_error)
            if not good.any() or not np.isfinite(errors).all():
                return None
            return good, errors

        fits = []
        for seed in seeds:
            t = seed.copy()
            current = evaluate(t)
            if current is None:
                continue
            for _ in range(2):
                good,_ = current
                try:
                    rv,tv = cv2.solvePnPRefineLM(objects[good].reshape(-1,3),pixels[good].reshape(-1,2),k,d,
                        cv2.Rodrigues(t[:3,:3])[0],t[:3,3].reshape(3,1).copy())
                    t[:3,:3],t[:3,3] = cv2.Rodrigues(rv)[0],tv.ravel()
                    current = evaluate(t)
                    if current is None:
                        break
                except cv2.error:
                    current = None
                    break
            if current is not None:
                good, errors = current
                fits.append((int(good.sum()),float(np.sqrt(np.mean(errors[good]**2))),t,good,errors))
        if not fits:
            return dict(base,status='POSE_FAILED')
        fits.sort(key=lambda f:(-f[0],f[1]))
        count,rms,t,good,errors = fits[0]
        alternative = None
        for other in fits[1:]:
            if other[0]!=count or not np.array_equal(other[3],good):
                continue
            angle = np.rad2deg(Rotation.from_matrix(other[2][:3,:3]@t[:3,:3].T).magnitude())
            delta = np.linalg.norm(other[2][:3,3]-t[:3,3])*1000
            if angle>3. or delta>.5:
                alternative = dict(rms_error_px=other[1],rotation_separation_deg=float(angle),
                                   translation_separation_mm=float(delta))
                break
        if len(faces)>1 and count<2:
            status = 'INCONSISTENT_FACES'
        elif alternative and alternative['rms_error_px']-rms < self.ambiguity_gap:
            status = 'AMBIGUOUS'
        else:
            status = 'OK_MULTI_FACE' if count>1 else 'OK_SINGLE_FACE'
        return dict(base,status=status,pose_valid=status in VALID_STATUSES,pose=pose_fields(t),
                    used_ids=[o['tag_id'] for o,g in zip(observations,good) if g],
                    rejected_ids=[o['tag_id'] for o,g in zip(observations,good) if not g],
                    reprojection_rms_px=rms,reprojection_max_px=float(errors[good].max()),
                    per_face_rms_px={str(o['tag_id']):float(np.sqrt(np.mean(e**2))) for o,e in zip(observations,errors)},
                    alternative_pose=alternative,
                    method='Current-frame cube corner consensus + multi-start PnP + LM; no temporal filtering')


def draw_result(image, result, camera, geometry):
    out = image.copy()
    if out.ndim==2:
        out = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
    for item in result['observations']:
        corners = np.rint(item['corners_px']).astype(np.int32)
        cv2.polylines(out,[corners],True,(70,210,255),1,cv2.LINE_AA)
        cv2.putText(out,str(item['tag_id']),tuple(corners[0]),cv2.FONT_HERSHEY_SIMPLEX,.5,(70,210,255),1,cv2.LINE_AA)
    for index,cube in enumerate(result['cubes']):
        color = (90,230,110) if cube['pose_valid'] else (70,160,255)
        label = f"Cube {cube['cube']:02d}: {cube['status']}  IDs {cube['used_ids'] or cube['detected_ids']}"
        cv2.putText(out,label,(15,28+index*30),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2,cv2.LINE_AA)
        if not cube['pose_valid'] or camera is None:
            continue
        t = np.array(cube['pose']['transform']); rv=cv2.Rodrigues(t[:3,:3])[0]
        cv2.drawFrameAxes(out,camera.camera_matrix,camera.distortion_coefficients,rv,t[:3,3],geometry.side_m*.9,2)
        vertices=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],float)*geometry.side_m/2
        uv=cv2.projectPoints(vertices,rv,t[:3,3],camera.camera_matrix,camera.distortion_coefficients)[0].reshape(-1,2)
        if not np.isfinite(uv).all() or abs(uv).max()>100000:
            continue
        uv=np.rint(uv).astype(np.int32)
        for i in range(8):
            for j in range(i+1,8):
                if np.count_nonzero(vertices[i]!=vertices[j])==1:
                    cv2.line(out,tuple(uv[i]),tuple(uv[j]),color,1,cv2.LINE_AA)
    return out
