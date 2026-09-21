"""Camera-free regression tests for the printed five-face cube geometry."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from apriltag_pose_core import CameraIntrinsics
from cube_pose_core import CubeGeometry,CubePoseEstimator,NORMALS,load_intrinsics


K=np.array([[1220.,0,632.],[0,1210.,508.],[0,0,1.]])
D=np.array([-.08,.015,.0003,-.0004,0.])


def matrix(angles=(15,-40,12),translation=(.005,-.007,.2)):
    t=np.eye(4); t[:3,:3]=Rotation.from_euler('xyz',angles,degrees=True).as_matrix(); t[:3,3]=translation
    return t


def project(geometry,t,cube=1,faces=None):
    records=[]
    for f,n in enumerate(NORMALS):
        center=t[:3,:3]@(n*geometry.side_m/2)+t[:3,3]
        if faces is None and np.dot(t[:3,:3]@n,center)>=0: continue
        if faces is not None and f not in faces: continue
        p=cv2.projectPoints(geometry.corners(f),cv2.Rodrigues(t[:3,:3])[0],t[:3,3],K,D)[0].reshape(4,2)
        records.append(dict(tag_id=(cube-1)*5+f,corners_px=p.tolist()))
    return records


class CubePoseTests(unittest.TestCase):
    def setUp(self):
        self.g=CubeGeometry(); self.camera=CameraIntrinsics(K,D,(1280,1024))
        self.est=CubePoseEstimator(self.camera)

    def test_all_five_faces_recover_the_same_cube_origin_and_axes(self):
        # Each canonical printed face is rotated into a front-facing camera pose.
        for f in range(5):
            for spin in (0,90,180,270):
                with self.subTest(face=f,spin=spin):
                    t=np.eye(4)
                    t[:3,:3]=Rotation.from_euler('xyz',[165,18,spin],degrees=True).as_matrix()@self.g.cube_T_tag(f)[:3,:3].T
                    t[:3,3]=[.004,-.006,.19]
                    result=self.est.estimate(project(self.g,t,faces=[f]),(1280,1024))['cubes'][0]
                    self.assertTrue(result['pose_valid'],result)
                    np.testing.assert_allclose(result['pose']['transform'],t,atol=2e-6)

    def test_multi_face_and_relative_cube_pose_with_distortion(self):
        a=matrix(); b=matrix((22,-35,-30),(.04,.015,.22))
        result=self.est.estimate(project(self.g,a)+project(self.g,b,2),(1280,1024))
        for actual,expected in zip(result['cubes'],[a,b]):
            self.assertEqual(actual['status'],'OK_MULTI_FACE')
            np.testing.assert_allclose(actual['pose']['transform'],expected,atol=1e-6)
        np.testing.assert_allclose(result['cube01_T_cube02']['transform'],np.linalg.inv(a)@b,atol=1e-6)

    def test_whole_face_outlier_is_rejected(self):
        truth=matrix(); obs=project(self.g,truth)
        self.assertEqual(len(obs),3)
        bad_id=obs[-1]['tag_id']; obs[-1]['corners_px']=(np.array(obs[-1]['corners_px'])+[35,-25]).tolist()
        result=self.est.estimate(obs,(1280,1024))['cubes'][0]
        self.assertEqual(result['status'],'OK_MULTI_FACE')
        self.assertEqual(result['rejected_ids'],[bad_id])
        np.testing.assert_allclose(result['pose']['transform'],truth,atol=1e-6)

    def test_two_inconsistent_faces_are_not_published_as_valid(self):
        obs=project(self.g,matrix())[:2]
        obs[-1]['corners_px']=(np.array(obs[-1]['corners_px'])+[120,80]).tolist()
        result=self.est.estimate(obs,(1280,1024))['cubes'][0]
        self.assertFalse(result['pose_valid'])
        self.assertIn(result['status'],['INCONSISTENT_FACES','POSE_FAILED'])

    def test_no_calibration_wrong_resolution_and_missing_frame_do_not_leak_pose(self):
        obs=project(self.g,matrix())
        result=self.est.estimate(obs,(1280,1024))
        self.assertTrue(result['cubes'][0]['pose_valid'])
        for estimator,size,status in [(CubePoseEstimator(),(1280,1024),'UNCALIBRATED'),
                                      (self.est,(1440,1080),'IMAGE_SIZE_MISMATCH')]:
            result=estimator.estimate(obs,size)
            self.assertEqual(result['cubes'][0]['status'],status)
            self.assertIsNone(result['cubes'][0]['pose'])
            self.assertIsNone(result['cube01_T_cube02'])
        result=self.est.estimate([],(1280,1024))
        self.assertTrue(all(x['pose'] is None and x['status']=='NOT_VISIBLE' for x in result['cubes']))

    def test_duplicate_ids_and_non_finite_corners(self):
        obs=project(self.g,matrix())
        result=self.est.estimate(obs+obs[:1],(1280,1024))['cubes'][0]
        self.assertEqual(result['status'],'DUPLICATE_ID'); self.assertIsNone(result['pose'])
        result=self.est.estimate([dict(tag_id=0,corners_px=np.full((4,2),np.nan))],(1280,1024))['cubes'][0]
        self.assertEqual(result['status'],'NOT_VISIBLE')

    def test_invalid_geometry(self):
        for side,tag in [(0,.01),(.016,.02),(float('nan'),.0128)]:
            with self.assertRaises(ValueError): CubeGeometry(side,tag)

    def test_single_face_ambiguous_pose_is_not_used_for_relative_pose(self):
        camera=CameraIntrinsics(K,np.zeros(5),(1280,1024))
        est=CubePoseEstimator(camera)
        rotation=Rotation.from_euler('xyz',[175,0,25],degrees=True).as_matrix()@self.g.cube_T_tag(0)[:3,:3].T
        pixels=cv2.projectPoints(self.g.corners(0),cv2.Rodrigues(rotation)[0],np.array([0.,0.,.5]),K,None)[0].reshape(4,2)
        pixels+=np.array([[.1,-.1],[-.1,.15],[.05,-.1],[-.05,.05]])
        result=est.estimate([dict(tag_id=0,corners_px=pixels.tolist())],(1280,1024))
        self.assertEqual(result['cubes'][0]['status'],'AMBIGUOUS')
        self.assertFalse(result['cubes'][0]['pose_valid'])
        self.assertIsNone(result['cube01_T_cube02'])

    def test_foreign_camera_and_unknown_resolution_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'camera.json'
            config=dict(camera_matrix=K.tolist(),distortion_coefficients=D.tolist(),image_width=1280,image_height=1024,
                        camera_name='MV-CS016-10UM_DB1933770')
            p.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'different camera'): load_intrinsics(p,expected_serial='DB0447679')
            config['camera_name']='MV-CU013-A0UC_DB0447679'; p.write_text(json.dumps(config))
            camera,_=load_intrinsics(p,expected_serial='DB0447679'); self.assertEqual(camera.image_size,(1280,1024))
            del config['image_width']; p.write_text(json.dumps(config))
            with self.assertRaises(ValueError): load_intrinsics(p)


if __name__=='__main__': unittest.main()
