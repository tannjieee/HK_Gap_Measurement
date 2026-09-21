"""Single HIKROBOT camera / image / video viewer for the 16 mm tag cubes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

# Restore PyQt's environment after importing the OpenCV wheel.
_qt_env = {k:os.environ.get(k) for k in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR')}
import cv2
import numpy as np
from cube_pose_core import CubeGeometry, CubePoseEstimator, draw_result, load_intrinsics
from cube_joint_measurement import JointMeasurement, cube_at_pixel, draw_roles, relative_sample
for _key,_value in _qt_env.items():
    if _value is None:
        os.environ.pop(_key,None)
    else:
        os.environ[_key]=_value

ROOT = Path(__file__).resolve().parent
DEFAULT_SERIAL = 'DB0447679'
DEFAULT_INTRINSICS = ROOT/'calibration_results/cube_MV-CU013-A0UC_DB0447679.yaml'
RUNS = ROOT/'cube_pose_runs'


def save_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class FrameSource:
    def __init__(self,args):
        self.args=args; self.controller=None; self.capture=None; self.sdk=None
        self.identity={}; self.number=0; self.image=None; self.finished=False

    def open(self):
        if self.args.image:
            self.image=cv2.imread(str(self.args.image))
            if self.image is None:
                raise ValueError(f'Cannot read image: {self.args.image}')
            self.identity=dict(type='image',path=str(self.args.image))
        elif self.args.video:
            self.capture=cv2.VideoCapture(str(self.args.video))
            if not self.capture.isOpened():
                raise ValueError(f'Cannot read video: {self.args.video}')
            self.identity=dict(type='video',path=str(self.args.video))
        else:
            import hikrobot_camera as sdk
            from hikrobot_gui import CameraController
            sdk.require_ok('MV_CC_Initialize',sdk.MvCamera.MV_CC_Initialize()); self.sdk=sdk
            self.device_refs,devices=sdk.enumerate_devices()
            selected=[d for d in devices if sdk.device_identity(d)[2]==self.args.serial]
            if len(selected)!=1:
                raise RuntimeError(f'Camera {self.args.serial} not found; connected: {[sdk.device_identity(d) for d in devices]}')
            transport,model,serial=sdk.device_identity(selected[0])
            self.identity=dict(type='hikrobot',transport=transport,model=model,serial=serial,
                               lens_focal_length_nominal_mm=self.args.lens_mm)
            self.controller=CameraController(); self.controller.open_camera(selected[0])
            # Latest-frame strategy avoids accumulating a backlog during fitting.
            if hasattr(self.controller.cam,'MV_CC_SetGrabStrategy'):
                sdk.require_ok('Set latest-image strategy',self.controller.cam.MV_CC_SetGrabStrategy(1))
            if self.args.exposure_us is not None:
                self.controller.set_enum_text('ExposureAuto','Off')
                self.controller.set_float('ExposureTime',self.args.exposure_us)
            self.controller.start_grabbing()

    def read(self):
        if self.args.image:
            if self.finished:
                return None
            self.finished=True
            return self.image.copy(),0,time.time_ns()
        if self.capture is not None:
            ok,frame=self.capture.read()
            if not ok:
                self.finished=True; return None
            self.number+=1
            return frame,self.number,time.time_ns()
        packet=self.controller.read_rgb_frame()
        if packet is None:
            return None
        data,w,h,number=packet
        frame=cv2.cvtColor(np.frombuffer(data,np.uint8).reshape(h,w,3),cv2.COLOR_RGB2BGR)
        return frame,number,time.time_ns()

    def close(self):
        if self.controller is not None:
            self.controller.close(); self.controller=None
        if self.capture is not None:
            self.capture.release(); self.capture=None
        if self.sdk is not None:
            self.sdk.MvCamera.MV_CC_Finalize(); self.sdk=None


class PoseWorker(threading.Thread):
    def __init__(self,args):
        super().__init__(daemon=True)
        self.args=args; self.stop_event=threading.Event(); self.commands=queue.Queue()
        self.lock=threading.Lock(); self.latest=None; self.error=None; self.status='Connecting'
        self.camera=None; self.calibration=None; self.record_file=None; self.csv_file=None
        self.record_path=None; self.processed=0

    def recording(self,path):
        self.close_recording()
        if path is None:
            return
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        self.record_file=path.open('x')
        try:
            self.csv_file=path.with_suffix('.csv').open('x',newline='')
        except Exception:
            self.record_file.close(); self.record_file=None; raise
        self.csv_writer=csv.writer(self.csv_file)
        self.csv_writer.writerow(['host_receive_time_ns','frame_number','cube','status','pose_valid',
                                  'x_mm','y_mm','z_mm','roll_deg','pitch_deg','yaw_deg',
                                  'qx','qy','qz','qw','reprojection_rms_px','used_ids'])
        self.record_path=str(path)

    def close_recording(self):
        for f in (self.record_file,self.csv_file):
            if f: f.close()
        self.record_file=self.csv_file=None; self.record_path=None

    def run(self):
        source=FrameSource(self.args)
        try:
            if self.args.intrinsics.is_file():
                self.camera,self.calibration=load_intrinsics(self.args.intrinsics,
                    expected_serial=None if self.args.image or self.args.video else self.args.serial)
            elif self.args.intrinsics != DEFAULT_INTRINSICS:
                raise FileNotFoundError(self.args.intrinsics)
            geometry=CubeGeometry(self.args.cube_side_mm/1000,self.args.tag_size_mm/1000)
            estimator=CubePoseEstimator(self.camera,tuple(self.args.cubes),geometry)
            source.open()
            self.status='Running'
            if self.args.record:
                self.recording(self.args.record)
            last_frame=time.monotonic()
            while not self.stop_event.is_set():
                while not self.commands.empty():
                    command,value=self.commands.get_nowait()
                    try:
                        if command=='intrinsics':
                            camera,metadata=load_intrinsics(value,expected_serial=source.identity.get('serial'))
                            self.camera,self.calibration=camera,metadata; estimator.camera=camera
                        elif command=='record':
                            self.recording(value)
                        elif command=='exposure' and source.controller:
                            source.controller.set_enum_text('ExposureAuto','Off')
                            source.controller.set_float('ExposureTime',float(value))
                        self.error=None
                    except Exception as exc:
                        self.error=f'{command}: {exc}'
                frame=source.read()
                if frame is None:
                    if source.finished: break
                    if time.monotonic()-last_frame>5:
                        raise RuntimeError('No camera frame received for 5 seconds')
                    continue
                image,number,received=frame
                started=time.perf_counter()
                result=estimator.detect(image)
                elapsed=(time.perf_counter()-started)*1000
                result.update(host_receive_time_ns=received,frame_number=number,
                              source=source.identity,calibration=self.calibration,processing_ms=elapsed)
                annotated=draw_result(image,result,self.camera,geometry)
                now=time.monotonic(); result['pipeline_fps']=1/max(now-last_frame,1e-6); last_frame=now
                with self.lock:
                    self.latest=(now,image,annotated,result)
                if self.record_file:
                    self.record_file.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                    for cube in result['cubes']:
                        pose=cube['pose'] if cube['pose_valid'] else None
                        numbers=pose['translation_mm']+pose['rpy_xyz_deg']+pose['quaternion_xyzw'] if pose else ['']*10
                        self.csv_writer.writerow([received,number,cube['cube'],cube['status'],cube['pose_valid'],
                            *numbers,cube.get('reprojection_rms_px',''),' '.join(map(str,cube['used_ids']))])
                    self.record_file.flush(); self.csv_file.flush()
                self.processed+=1
                if self.args.frames and self.processed>=self.args.frames: break
                if self.args.video:
                    fps=source.capture.get(cv2.CAP_PROP_FPS)
                    self.stop_event.wait(max(0,1/max(fps,1)-(time.monotonic()-now)))
            self.status='Stopped'
        except Exception as exc:
            self.error=str(exc); self.status='Error'
        finally:
            source.close(); self.close_recording()


def save_snapshot(packet,folder):
    _,raw,annotated,result=packet
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    if not cv2.imwrite(str(folder/'raw.png'),raw) or not cv2.imwrite(str(folder/'detected.png'),annotated):
        raise RuntimeError('Could not save snapshot image')
    save_json(folder/'poses.json',result)


def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def run_gui(args):
    from PyQt5.QtCore import QTimer,Qt
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import (QApplication,QMainWindow,QWidget,QHBoxLayout,QVBoxLayout,
        QLabel,QPushButton,QPlainTextEdit,QFileDialog,QDoubleSpinBox,QTabWidget,QScrollArea,QLayout)
    from cube_pose_widgets import CameraImageView

    class Window(QMainWindow):
        def __init__(self):
            super().__init__(); self.worker=None; self.packet=None; self.last_display=None
            self.measurement=JointMeasurement(); self.select_role='parent'; self.frozen=None
            self.initial_packet=self.final_packet=None; self.measurement_folder=None
            self.setWindowTitle('AprilTag 方块 6D 位姿 / 父子 Link 转角 · MV-CU013-A0UC')
            self.resize(1390,900)
            self.video=CameraImageView(); self.video.image_clicked.connect(self.select_at)
            panel=QWidget(); panel.setFixedWidth(460); panel_layout=QVBoxLayout(panel)
            self.info=QLabel(''); self.info.setWordWrap(True); panel_layout.addWidget(self.info)
            self.frame_state=QLabel('实时画面'); self.frame_state.setStyleSheet('font-weight:bold;color:#205dad;')
            panel_layout.addWidget(self.frame_state)
            self.tabs=QTabWidget(); panel_layout.addWidget(self.tabs)
            joint_tab=QWidget(); joint=QVBoxLayout(joint_tab); joint.setSizeConstraint(QLayout.SetMinimumSize)
            joint_scroll=QScrollArea(); joint_scroll.setWidgetResizable(True); joint_scroll.setFrameShape(0)
            joint_scroll.setWidget(joint_tab); self.tabs.addTab(joint_scroll,'父 / 子 Link 转角')
            self.role_info=QLabel(); joint.addWidget(self.role_info)
            line=QHBoxLayout()
            self.parent_button=QPushButton('① 点选父 Link'); self.parent_button.setCheckable(True)
            self.child_button=QPushButton('② 点选子 Link'); self.child_button.setCheckable(True)
            self.parent_button.clicked.connect(lambda:self.choose_role('parent'))
            self.child_button.clicked.connect(lambda:self.choose_role('child'))
            self.swap_button=QPushButton('交换父 / 子'); self.swap_button.clicked.connect(self.swap_roles)
            line.addWidget(self.parent_button); line.addWidget(self.child_button); line.addWidget(self.swap_button); joint.addLayout(line)
            self.pick_hint=QLabel(); self.pick_hint.setWordWrap(True); joint.addWidget(self.pick_hint)
            self.initial_button=QPushButton('③ 定格初始帧（设为零位）')
            self.initial_button.clicked.connect(lambda:self.capture_angle('initial')); joint.addWidget(self.initial_button)
            self.resume_button=QPushButton('恢复实时画面 → 手动移动子 Link')
            self.resume_button.clicked.connect(self.resume_live); joint.addWidget(self.resume_button)
            self.final_button=QPushButton('④ 定格结束帧 / 计算总旋转角度')
            self.final_button.clicked.connect(lambda:self.capture_angle('final')); joint.addWidget(self.final_button)
            self.angle_label=QLabel('总旋转角度：—'); self.angle_label.setStyleSheet('font-size:25px;font-weight:bold;color:#205dad;')
            self.angle_label.setToolTip('以父 Link 为参考，子 Link 从初始姿态到结束姿态的整体三维转角幅值。')
            joint.addWidget(self.angle_label)
            self.angle_detail=QLabel('先在画面中点选父方块，再点选子方块。')
            self.angle_detail.setWordWrap(True); joint.addWidget(self.angle_detail)
            line=QHBoxLayout()
            self.view_initial=QPushButton('查看初始帧'); self.view_final=QPushButton('查看结束帧')
            self.view_initial.clicked.connect(lambda:self.show_capture('initial'))
            self.view_final.clicked.connect(lambda:self.show_capture('final'))
            reset=QPushButton('重置测量'); reset.clicked.connect(self.reset_measurement)
            for button in (self.view_initial,self.view_final,reset): line.addWidget(button)
            joint.addLayout(line)
            hint=QLabel('初始 → 结束的整体转角，范围 0–180°。\n父块共同运动会被抵消；不累计中途转动。')
            hint.setWordWrap(True); joint.addWidget(hint)
            camera_tab=QWidget(); controls=QVBoxLayout(camera_tab); self.tabs.addTab(camera_tab,'相机 / 标定 / 记录')
            self.start_button=QPushButton('连接 / 开始'); self.start_button.clicked.connect(self.start); controls.addWidget(self.start_button)
            stop=QPushButton('停止并释放相机'); stop.clicked.connect(self.stop); controls.addWidget(stop)
            load=QPushButton('加载这台相机的内参…'); load.clicked.connect(self.load); controls.addWidget(load)
            calibrate=QPushButton('棋盘格标定：12×9格，15 mm'); calibrate.clicked.connect(self.calibrate); controls.addWidget(calibrate)
            line=QHBoxLayout(); self.exposure=QDoubleSpinBox(); self.exposure.setRange(10,1000000)
            self.exposure.setValue(args.exposure_us or 10000); self.exposure.setSuffix(' µs')
            line.addWidget(self.exposure); apply=QPushButton('应用手动曝光'); apply.clicked.connect(self.apply_exposure); line.addWidget(apply); controls.addLayout(line)
            snap=QPushButton('保存原图、叠加图和当前位姿'); snap.clicked.connect(self.snapshot); controls.addWidget(snap)
            self.record_button=QPushButton('开始记录 JSONL + CSV'); self.record_button.clicked.connect(self.record); controls.addWidget(self.record_button)
            self.values=QPlainTextEdit(); self.values.setReadOnly(True)
            self.values.setStyleSheet('font-family:monospace;font-size:13px;'); panel_layout.addWidget(self.values,1)
            tip=QLabel('位姿原点是方块中心。相机 X向右、Y向下、Z向前。\n红/绿/蓝轴 = 方块X/Y/Z；+X为留空安装面。\n定格时两块须同时有效，建议各自露出相邻两面。'); tip.setWordWrap(True); panel_layout.addWidget(tip)
            controls.addStretch(1)
            layout=QHBoxLayout(); layout.addWidget(self.video,1); layout.addWidget(panel)
            central=QWidget(); central.setLayout(layout); self.setCentralWidget(central)
            self.timer=QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(60)
            self.update_measurement_controls()
            QTimer.singleShot(0,self.start)

        def update_measurement_controls(self):
            m=self.measurement
            self.role_info.setText(f"父 Link：{f'方块 {m.parent:02d}' if m.parent else '未选择'}    "
                                   f"子 Link：{f'方块 {m.child:02d}' if m.child else '未选择'}")
            self.parent_button.setChecked(self.select_role=='parent')
            self.child_button.setChecked(self.select_role=='child')
            self.video.setCursor(Qt.CrossCursor if self.select_role else Qt.ArrowCursor)
            self.pick_hint.setText('请点击画面中父方块的 Tag 区域（青色）' if self.select_role=='parent' else
                                   '请点击画面中子方块的 Tag 区域（紫色）' if self.select_role=='child' else
                                   '已选好；更换任一方块会清空本次初始帧与结果。')
            selected=m.parent is not None and m.child is not None
            self.swap_button.setEnabled(selected)
            self.initial_button.setEnabled(selected and not self.frozen)
            self.final_button.setEnabled(m.initial is not None and not self.frozen)
            self.resume_button.setEnabled(bool(self.frozen))
            self.view_initial.setEnabled(self.initial_packet is not None)
            self.view_final.setEnabled(self.final_packet is not None)

        def choose_role(self,role):
            self.select_role=role; self.update_measurement_controls()

        def swap_roles(self):
            m=self.measurement
            if m.parent is None or m.child is None: return
            m.parent,m.child=m.child,m.parent
            self.reset_measurement()
            if self.packet: self.render_packet(self.packet)
            self.statusBar().showMessage('已交换父／子角色，请重新定格初始帧。')

        def select_at(self,x,y):
            if not self.select_role or self.packet is None: return
            if not self.frozen and not self.fresh_packet():
                self.statusBar().showMessage('画面已过期，请等待实时画面'); return
            cube=cube_at_pixel(self.packet[3],x,y)
            if cube is None:
                self.statusBar().showMessage('请点击一个已检测方块的 Tag 区域，避免两块重叠处'); return
            try:
                previous=(self.measurement.parent,self.measurement.child)
                self.measurement.select(self.select_role,cube)
                if previous!=(self.measurement.parent,self.measurement.child):
                    self.clear_captures()
                self.select_role=('parent' if self.measurement.parent is None else
                                  'child' if self.measurement.child is None else None)
                self.update_measurement_controls()
                if self.packet: self.render_packet(self.packet)
                self.statusBar().showMessage(f'已选择方块 {cube:02d}')
            except ValueError as exc: self.statusBar().showMessage(str(exc))

        def clear_captures(self):
            self.initial_packet=self.final_packet=None; self.measurement_folder=None
            self.angle_label.setText('总旋转角度：—')
            self.angle_detail.setText('定格初始帧后，恢复实时画面并移动子 Link。')
            self.frozen=None; self.last_display=None

        def reset_measurement(self):
            self.measurement.clear(); self.clear_captures(); self.update_measurement_controls(); self.refresh()
            self.statusBar().showMessage('已清空本次测量；已保存的文件保留，父／子选择保留。')

        def fresh_packet(self):
            return (self.packet is not None and (bool(args.image) or
                (self.worker is not None and self.worker.is_alive()
                 and time.monotonic()-self.packet[0] <= .5
                 and (time.time_ns()-self.packet[3]['host_receive_time_ns'])/1e9 <= .5)))

        def capture_angle(self,stage):
            try:
                if self.frozen: raise ValueError('请先恢复实时画面，再定格新帧')
                if not self.fresh_packet(): raise ValueError('当前没有新鲜画面，未采集；请等待两块同时有效')
                proposal=deepcopy(self.measurement)
                if stage=='initial': proposal.capture_initial(self.packet[3])
                else: proposal.capture_final(self.packet[3])
                packet=(*self.packet[:2],draw_roles(self.packet[2],self.packet[3],proposal.parent,proposal.child),self.packet[3])
                folder=RUNS/('joint_angle_'+timestamp()) if stage=='initial' else self.measurement_folder
                name='initial' if stage=='initial' else 'final_'+timestamp()
                save_snapshot(packet,folder/name)
                document=proposal.as_dict()
                document['snapshots']={'initial':'initial','final':None if stage=='initial' else name}
                save_json(folder/name/'measurement.json',document)
                save_json(folder/'measurement.json',document)
                self.measurement=proposal; self.measurement_folder=folder
                if stage=='initial':
                    self.initial_packet=self.packet; self.final_packet=None
                    self.angle_label.setText('总旋转角度：0.000°')
                    self.angle_detail.setText('初始帧已保存。恢复实时画面，移动子 Link，再定格结束帧。')
                else:
                    self.final_packet=self.packet; result=proposal.result
                    self.angle_label.setText(f"总旋转角度：{result['angle_deg']:.3f}°")
                    detail='子 Link 相对父 Link，从初始帧到结束帧。'
                    if result['single_face_used']: detail+='\n本次含单面估计；建议露出相邻两面后复测。'
                    detail+='\n两帧、位姿与角度已自动保存。'
                    self.angle_detail.setText(detail)
                self.angle_detail.setToolTip(str(folder))
                self.show_capture(stage)
                self.statusBar().showMessage('已保存：'+str(folder))
            except Exception as exc: self.statusBar().showMessage(str(exc))

        def show_capture(self,stage):
            packet=self.initial_packet if stage=='initial' else self.final_packet
            if packet is None: return
            self.frozen='初始帧' if stage=='initial' else '结束帧'
            self.packet=packet; self.render_packet(packet); self.update_measurement_controls()

        def resume_live(self):
            self.frozen=None; self.last_display=None
            self.update_measurement_controls(); self.refresh()

        def start(self):
            if self.worker and self.worker.is_alive(): return
            self.measurement.clear(); self.clear_captures(); self.update_measurement_controls()
            self.values.setPlainText('正在连接：当前无有效位姿'); self.video.setText('等待新画面')
            self.worker=PoseWorker(args); self.last_display=None; self.packet=None; self.worker.start()

        def stop(self):
            if self.worker:
                self.worker.stop_event.set(); self.worker.join(timeout=3)
                if self.worker.is_alive():
                    self.statusBar().showMessage('正在等待相机读取退出'); return False
            self.values.setPlainText('已停止：当前无有效位姿'); return True

        def load(self):
            name,_=QFileDialog.getOpenFileName(self,'加载 MV-CU013-A0UC 内参',str(ROOT/'calibration_results'),'Calibration (*.yaml *.yml *.json)')
            if name:
                try:
                    load_intrinsics(name,expected_serial=None if args.image or args.video else args.serial)
                    args.intrinsics=Path(name)
                    self.reset_measurement()
                    if self.worker and self.worker.is_alive(): self.worker.commands.put(('intrinsics',name))
                    else: self.start()
                except Exception as exc: self.statusBar().showMessage(str(exc))

        def calibrate(self):
            if not self.stop(): return
            subprocess.Popen(['bash',str(ROOT/'scripts/calibrate_cube_camera.sh'),'--serial',args.serial],cwd=ROOT)
            self.statusBar().showMessage('标定保存后关闭标定窗口，再点击“连接 / 开始”。')

        def apply_exposure(self):
            if self.worker and self.worker.is_alive(): self.worker.commands.put(('exposure',self.exposure.value()))

        def snapshot(self):
            if not self.packet or (not self.frozen and not self.fresh_packet()):
                self.statusBar().showMessage('没有新鲜画面可保存'); return
            folder=RUNS/('snapshot_'+timestamp())
            try:
                packet=(*self.packet[:2],draw_roles(self.packet[2],self.packet[3],self.measurement.parent,self.measurement.child),self.packet[3])
                save_snapshot(packet,folder); self.statusBar().showMessage(str(folder))
            except Exception as exc: self.statusBar().showMessage(str(exc))

        def record(self):
            if not self.worker or not self.worker.is_alive(): return
            path=None if self.worker.record_path else RUNS/('tracking_'+timestamp()+'.jsonl')
            self.worker.commands.put(('record',path))

        def refresh(self):
            if not self.worker: return
            worker=self.worker
            with worker.lock: packet=worker.latest
            self.record_button.setText('停止记录' if worker.record_path else '开始记录 JSONL + CSV')
            if worker.error: self.statusBar().showMessage(worker.error)
            if self.frozen: return
            if packet is None:
                self.info.setText(worker.status+('\n'+worker.error if worker.error else '')); return
            stale=(not args.image and (not worker.is_alive() or time.monotonic()-packet[0]>.5))
            if stale:
                self.frame_state.setText('画面已过期 / 已停止（禁止采集）')
                self.info.setText('STALE / 已停止：无新鲜测量'); self.values.setPlainText('当前无有效位姿'); return
            self.packet=packet
            if self.last_display==packet[0]: return
            self.last_display=packet[0]
            self.render_packet(packet)

        def render_packet(self,packet):
            _,_,annotated,result=packet
            annotated=draw_roles(annotated,result,self.measurement.parent,self.measurement.child)
            rgb=np.ascontiguousarray(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)); h,w=rgb.shape[:2]
            q=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()
            self.video.setImage(q)
            self.frame_state.setText((f'已定格：{self.frozen}（历史帧）' if self.frozen else '实时画面')+f" · 帧 {result['frame_number']}")
            cal=result['calibration']
            self.info.setText(f"{result['source'].get('model',result['source']['type'])}  {w}×{h}\n"
                f"检测+解算 {result['processing_ms']:.1f} ms | 处理 {result['pipeline_fps']:.1f} fps\n"
                +(f"内参：{Path(cal['path']).name}" if cal else '尚未标定：只检测ID，不输出米制位姿'))
            lines=[]
            for cube in result['cubes']:
                lines += [f"方块 {cube['cube']:02d}: {cube['status']}",f"检测ID {cube['detected_ids']} / 使用 {cube['used_ids']}"]
                if cube['pose_valid']:
                    p=cube['pose']; xyz=' '.join(f'{x:9.3f}' for x in p['translation_mm']); rpy=' '.join(f'{x:9.3f}' for x in p['rpy_xyz_deg'])
                    lines += [f'XYZ mm: {xyz}',f'RPY  °: {rpy}',f"重投影 RMS: {cube['reprojection_rms_px']:.3f} px"]
                else: lines += ['当前无有效6D位姿']
                lines.append('')
            try: rel=relative_sample(result,self.measurement.parent,self.measurement.child)['parent_T_child']
            except ValueError: rel=None
            lines += ['所选子 Link 相对父 Link（父块坐标系）']
            if rel is not None:
                lines += ['XYZ mm: '+' '.join(f'{x:.3f}' for x in rel['translation_mm']),
                          'RPY  °: '+' '.join(f'{x:.3f}' for x in rel['rpy_xyz_deg'])]
            else: lines += ['需要两块在同一帧均有有效位姿']
            self.values.setPlainText('\n'.join(lines))

        def closeEvent(self,event):
            if self.stop(): event.accept()
            else: event.ignore()

    app=QApplication.instance() or QApplication(sys.argv); window=Window(); window.show()
    return app.exec_()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--image',type=Path); source.add_argument('--video',type=Path)
    parser.add_argument('--serial',default=DEFAULT_SERIAL)
    parser.add_argument('--lens-mm',type=float,default=8.,help='Nominal lens focal length for records only; K is loaded from calibration')
    parser.add_argument('--intrinsics',type=Path,default=DEFAULT_INTRINSICS)
    parser.add_argument('--cube-side-mm',type=float,default=16.)
    parser.add_argument('--tag-size-mm',type=float,default=12.8)
    parser.add_argument('--cubes',nargs='+',type=int,default=[1,2])
    parser.add_argument('--exposure-us',type=float)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--frames',type=int,default=0,help='Stop after this many processed frames; 0 means continuous')
    parser.add_argument('--record',type=Path,help='New JSONL filename; also writes a CSV with the same stem')
    parser.add_argument('--output',type=Path,help='New directory for the final raw/annotated frame and JSON')
    args=parser.parse_args(argv)
    if args.frames<0: parser.error('--frames must be nonnegative')
    if not np.isfinite(args.lens_mm) or args.lens_mm<=0:
        parser.error('--lens-mm must be positive and finite')
    if args.exposure_us is not None and (not np.isfinite(args.exposure_us) or args.exposure_us<=0):
        parser.error('--exposure-us must be positive and finite')
    cv2.setNumThreads(2)
    if not args.headless: return run_gui(args)
    worker=PoseWorker(args); worker.start()
    try:
        while worker.is_alive(): worker.join(timeout=.25)
    except KeyboardInterrupt:
        worker.stop_event.set(); worker.join(timeout=3)
    if worker.error:
        print(worker.error,file=sys.stderr); return 1
    if worker.latest:
        if args.output: save_snapshot(worker.latest,args.output)
        result=worker.latest[3]
        print(json.dumps(dict(frames=worker.processed,calibrated=result['calibrated'],
             cubes=[dict(cube=x['cube'],status=x['status'],ids=x['detected_ids']) for x in result['cubes']]),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
