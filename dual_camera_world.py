#!/usr/bin/env python3
"""Two camera AprilTag previews, frozen world calibration, and RViz comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from gemini335l_apriltag import Processor, rvec_quaternion
from apriltag_pose_gui import VideoWidget, load_pose_config
from dual_camera_world_core import Observation, WorldComparison
import cv2
import numpy as np
import yaml
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent


def ensure_laser_off(node):
    """Enforce the user's emitter-off preference even when reusing a driver."""
    import rclpy
    from std_srvs.srv import SetBool
    from orbbec_camera_msgs.srv import GetBool
    for service_type, name, request in (
        (SetBool, "/gemini335l/set_laser_enable", SetBool.Request(data=False)),
        (GetBool, "/gemini335l/get_laser_status", GetBool.Request()),
    ):
        client = node.create_client(service_type, name)
        try:
            if not client.wait_for_service(timeout_sec=12):
                raise RuntimeError(f"无法确认结构光已关闭，服务不可用：{name}")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            if not future.done() or future.result() is None or not future.result().success:
                raise RuntimeError(f"关闭/读取结构光失败：{name}")
            if service_type is GetBool and future.result().data:
                raise RuntimeError("硬件仍报告结构光开启")
        finally:
            node.destroy_client(client)
    print("GEMINI 335L laser hardware status: OFF", flush=True)


class CameraSource(QThread):
    error = pyqtSignal(str)

    def __init__(self, camera, config):
        super().__init__()
        self.camera, self.config = camera, config
        self.lock = threading.Lock()
        self.latest = None

    def take_latest(self):
        with self.lock:
            frame, self.latest = self.latest, None
        return frame

    def process(self, processor, message):
        frame = processor.process(message)
        with self.lock:
            self.latest = frame

    def run(self):
        try:
            processor = Processor(self.config)
            if self.camera == "hik":
                self.run_hik(processor)
            else:
                self.run_gemini(processor)
        except Exception as exc:
            self.error.emit(f"{self.camera}：{exc}")

    def run_hik(self, processor):
        import hikrobot_camera as sdk
        from hikrobot_gui import CameraController
        sdk.require_ok("Initialize", sdk.MvCamera.MV_CC_Initialize())
        controller = CameraController()
        try:
            listing, devices = sdk.enumerate_devices()
            _, device = sdk.select_device(devices, 0, self.config.camera_serial)
            controller.open_camera(device)
            controller.start_grabbing()
            while not self.isInterruptionRequested():
                packet = controller.read_rgb_frame()
                if packet is None:
                    continue
                data, width, height, number = packet
                # SDK exposes a device frame number, not a ROS-synchronized clock.
                # Record host receive time explicitly; never claim exposure sync.
                stamp = time.time_ns()
                message = SimpleNamespace(data=data, width=width, height=height, step=width*3,
                    encoding="rgb8", header=SimpleNamespace(frame_id="hik_color_optical_frame",
                    stamp=SimpleNamespace(sec=stamp//10**9, nanosec=stamp%10**9)))
                self.process(processor, message)
        finally:
            controller.close()
            sdk.MvCamera.MV_CC_Finalize()

    def run_gemini(self, processor):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        context = Context()
        rclpy.init(args=[], context=context)
        node = rclpy.create_node(f"dual_world_gemini_{os.getpid()}", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        last_error = ""

        def receive(message):
            nonlocal last_error
            try:
                self.process(processor, message)
                last_error = ""
            except Exception as exc:
                if str(exc) != last_error:
                    self.error.emit(f"gemini：{exc}")
                    last_error = str(exc)
        topic = self.config.raw.get("ros", {}).get("image_topic", "/gemini335l/color/image_raw")
        node.create_subscription(Image, topic, receive,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        try:
            while context.ok() and not self.isInterruptionRequested():
                executor.spin_once(timeout_sec=.05)
        finally:
            executor.shutdown()
            node.destroy_node()
            context.shutdown()


class WorldPublisher:
    def __init__(self, node):
        from visualization_msgs.msg import MarkerArray
        from std_msgs.msg import String
        from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
        self.node = node
        self.markers = node.create_publisher(MarkerArray, "/dual_camera_world/markers", 1)
        self.reports = node.create_publisher(String, "/dual_camera_world/comparison_json", 1)
        self.tf = TransformBroadcaster(node)
        self.static_tf = StaticTransformBroadcaster(node)
        self.counter = 0

    def transform(self, name, matrix):
        from geometry_msgs.msg import TransformStamped
        t = TransformStamped()
        t.header.frame_id = "world"
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.child_frame_id = name
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, matrix[:3, 3])
        q = rvec_quaternion(cv2.Rodrigues(matrix[:3, :3])[0])
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        return t

    def publish(self, model, now):
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import String
        self.counter = 0
        items = []

        def marker(kind, color, position=None, scale=(.012, .012, .012), text="", points=None):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns = "dual_camera_world"
            m.id = self.counter
            self.counter += 1
            m.type = kind
            m.action = Marker.ADD
            m.pose.orientation.w = 1.
            if position is not None:
                m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, position)
            m.scale.x, m.scale.y, m.scale.z = map(float, scale)
            m.color.r, m.color.g, m.color.b, m.color.a = map(float, (*color, 1))
            # The installed Ogre font renders spaces at an excessive width.
            # Compact labels avoid that renderer issue without changing values.
            m.text = text.replace(" ", "_")
            m.lifetime.sec = 0
            m.lifetime.nanosec = 500_000_000
            if points is not None:
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
            items.append(m)

        def axes(matrix, length=.05):
            origin = matrix[:3, 3]
            for column, color in enumerate(((1, .1, .1), (.1, 1, .1), (.1, .3, 1))):
                marker(Marker.ARROW, color, scale=(.003, .006, .008),
                       points=[origin, origin + matrix[:3, column]*length])

        # DELETEALL removes tags that disappeared before the previous lifetime expires.
        clear = Marker()
        clear.action = Marker.DELETEALL
        items.append(clear)
        axes(np.eye(4), .1)
        origin = model.known[:3, 3]
        marker(Marker.SPHERE, (.8, .8, .8), origin, (.008, .008, .008))
        self.static_tf.sendTransform(self.transform("comparison/reference_known", model.known))
        report = model.report(now)
        world = model.world_poses(now)
        transforms = []
        colors = {"hik": (0, .8, 1), "gemini": (1, .5, .05)}
        for c, calibration in model.calibrations.items():
            matrix = calibration.world_from_camera
            transforms.append(self.transform(f"comparison/{c}_camera", matrix))
            axes(matrix, .08)
            marker(Marker.SPHERE, colors[c], matrix[:3, 3], (.022, .022, .022))
            marker(Marker.TEXT_VIEW_FACING, colors[c], matrix[:3, 3]+[0, 0, .05],
                   scale=(0, 0, .02), text=f"{c.upper()} camera")
        for c, poses in world.items():
            for tag_id, matrix in poses.items():
                transforms.append(self.transform(f"comparison/{c}_tag_{tag_id}", matrix))
                axes(matrix)
                marker(Marker.SPHERE, colors[c], matrix[:3, 3], (.012, .012, .012))
                offset = .035 if c == "hik" else -.035
                xyz = matrix[:3, 3]*1000
                marker(Marker.TEXT_VIEW_FACING, colors[c], matrix[:3, 3]+[0, 0, offset],
                       scale=(0, 0, .015), text=f"{c.upper()} ID {tag_id}\n({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm")
        pair = report["current_pair"]
        lines = ["HIK (cyan) - GEMINI (orange)", "World reference ID 0: (126.1, -12.5, 142) mm"]
        if pair and pair["differences"]:
            lines.append(f"Host arrival skew: {pair['arrival_skew_ms']:.1f} ms (NOT hardware synced)")
            for i, d in pair["differences"].items():
                delta = d["delta_xyz_mm"]
                lines.append(f"ID {i}: dXYZ=({delta[0]:+.2f}, {delta[1]:+.2f}, {delta[2]:+.2f}) mm")
                lines.append(f"distance={d['distance_mm']:.2f} mm; rotation={d['rotation_difference_deg']:.2f} deg")
                if i in world.get("hik", {}) and i in world.get("gemini", {}):
                    marker(Marker.LINE_LIST, (1, 1, .2), scale=(.002, 0, 0),
                           points=[world["hik"][i][:3, 3], world["gemini"][i][:3, 3]])
        elif model.collecting:
            lines.append(f"Calibrating: {len(model.samples['hik'])}/{model.settings.get('sample_count',60)} pairs")
        elif not model.calibrations:
            lines.append("Place reference tag; click Calibrate in the dual camera window")
        else:
            lines.append("Waiting for a fresh common tag in both cameras")
        lines.append("ID 0 is the calibration anchor; use another tag to evaluate agreement")
        marker(Marker.TEXT_VIEW_FACING, (1, 1, 1), origin+[0, 0, .25], scale=(0, 0, .018), text="\n".join(lines))
        if transforms:
            self.tf.sendTransform(transforms)
        self.markers.publish(MarkerArray(markers=items))
        self.reports.publish(String(data=json.dumps(report, ensure_ascii=False)))


class ComparisonWindow(QMainWindow):
    def __init__(self, path, node, load_session=None):
        super().__init__()
        self.node = node
        raw = yaml.safe_load(path.read_text())
        self.configs = {}
        metadata = {}
        for camera in ("hik", "gemini"):
            config = load_pose_config(path.parent / raw[f"{camera}_config"])
            config = replace(config, tag_ids=(), tag_size_mm=float(raw.get("tag_size_mm", 80)),
                             axis_length_mm=float(raw.get("tag_size_mm", 80))/2)
            if camera == "hik":
                config = replace(config, raw={**config.raw, "ros": {"optical_frame": "hik_color_optical_frame"}})
            self.configs[camera] = config
            metadata[camera] = {"serial": config.camera_serial, "intrinsics_file": str(config.intrinsics_file),
                "intrinsics_sha256": hashlib.sha256(config.intrinsics_file.read_bytes()).hexdigest(),
                "tag_family": config.family, "tag_size_mm": config.tag_size_mm}
        self.model = WorldComparison(raw["world"], metadata)
        if load_session:
            from world_calibration_core import WorldCalibration
            results = {c: WorldCalibration.load(load_session / f"{c}_world.json") for c in self.model.cameras}
            for c, result in results.items():
                if (result.metadata != {"camera": c, **self.model.metadata[c]} or
                        not np.allclose(result.world_from_reference, self.model.known, atol=1e-9)):
                    raise ValueError("保存外参与当前相机、内参、标签规格或世界位姿不匹配")
            self.model.calibrations = results
        self.publisher = WorldPublisher(node)
        self.saved_generation = 0
        self.session_dir = load_session.resolve() if load_session else None
        self.frames, self.sources, self.videos, self.labels = {}, {}, {}, {}
        self.rviz = None
        self.setWindowTitle("海康 + GEMINI 335L — 世界坐标标定与差异比较")
        self.resize(1540, 960)
        layout = QVBoxLayout()
        self.guide = QLabel("固定两台相机。将 ID 0 放在世界 (126.1, −12.5, 142) mm，重定义后的标签轴与世界轴同向。\n"
            "点击标定并保持静止；完成后锁定外参。海康：青色，335L：橙色。差值 = 海康 − 335L。")
        if load_session:
            self.guide.setText("已加载固定世界外参，ID 0 可以拆除。保持两台相机的位置及成像设置不变。\n"
                "海康：青色，335L：橙色。其他共同标签仍可计算世界位姿；没有共同标签时不显示实时差值。")
        self.guide.setWordWrap(True)
        layout.addWidget(self.guide)
        previews = QHBoxLayout()
        for c in self.model.cameras:
            panel = QVBoxLayout()
            self.labels[c] = QLabel(f"{c}：连接中…")
            panel.addWidget(self.labels[c])
            self.videos[c] = VideoWidget()
            self.videos[c].setMinimumSize(600, 400)
            panel.addWidget(self.videos[c])
            previews.addLayout(panel)
            source = CameraSource(c, self.configs[c])
            source.error.connect(lambda message: self.statusBar().showMessage(message))
            self.sources[c] = source
        layout.addLayout(previews, 1)
        controls = QHBoxLayout()
        self.calibrate_button = QPushButton("ID 0 已就位：标定两台相机（60 对）")
        self.calibrate_button.clicked.connect(self.begin)
        controls.addWidget(self.calibrate_button)
        reset = QPushButton("清零差异统计")
        reset.clicked.connect(self.reset_statistics)
        controls.addWidget(reset)
        save = QPushButton("保存当前比较报告")
        save.clicked.connect(self.save_report)
        controls.addWidget(save)
        rviz = QPushButton("打开 RViz")
        rviz.clicked.connect(self.open_rviz)
        controls.addWidget(rviz)
        layout.addLayout(controls)
        self.status = QLabel("等待两台相机看到 ID 0；尚未标定世界坐标")
        layout.addWidget(self.status)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(220)
        layout.addWidget(self.report)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        from std_srvs.srv import Trigger

        def calibrate_service(request, response):
            self.begin()
            response.success = True
            response.message = "采样已开始；请保持 ID 0 和两台相机静止"
            return response
        self.service = node.create_service(Trigger, "/dual_camera_world/calibrate", calibrate_service)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(40)
        self.last_publish = 0
        for source in self.sources.values():
            source.start()

    def begin(self):
        self.model.begin(time.monotonic())
        self.session_dir = None
        self.statusBar().showMessage("正在采样：相机与 ID 0 请保持静止")

    def reset_statistics(self):
        self.model.statistics.clear()
        self.model.pair = None

    def refresh(self):
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=0)
        now = time.monotonic()
        for c, source in self.sources.items():
            frame = source.take_latest()
            if frame is not None:
                self.frames[c] = frame
                self.model.ingest(c, Observation.from_frame(frame), now)
                rgb = frame.rgb
                self.videos[c].set_image(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy())
                self.labels[c].setText(f"{c.upper()} | {rgb.shape[1]}×{rgb.shape[0]} | ID {[p.tag_id for p in frame.poses]} | {frame.elapsed_ms:.1f} ms")
            if c in self.frames and now-self.frames[c].received_at > .5:
                self.videos[c].clear()
                self.labels[c].setText(f"{c.upper()}：图像超时")
        self.model.tick(now)
        self.calibrate_button.setEnabled(not self.model.collecting)
        if self.model.collecting:
            self.status.setText(f"正在标定：{len(self.model.samples['hik'])}/{self.model.settings.get('sample_count',60)} 对有效观测")
        elif self.model.last_error:
            self.status.setText(self.model.last_error)
        elif self.model.calibrations:
            self.status.setText("两台相机世界外参已锁定；只比较静止的共同标签。移动相机后须重新标定。")
        if self.model.generation != self.saved_generation:
            self.saved_generation = self.model.generation
            try:
                self.session_dir = ROOT / "calibration_results/world" / f"dual_{datetime.now():%Y%m%d_%H%M%S_%f}"
                for c, calibration in self.model.calibrations.items():
                    calibration.save(self.session_dir / f"{c}_world.json")
                self.statusBar().showMessage(f"世界外参已保存：{self.session_dir}")
            except OSError as exc:
                self.statusBar().showMessage(f"外参保存失败：{exc}")
        if now-self.last_publish >= .1:
            self.last_publish = now
            self.publisher.publish(self.model, now)
            doc = self.model.report(now)
            lines = []
            for c, poses in self.model.world_poses(now).items():
                for i, t in poses.items():
                    xyz = t[:3, 3]*1000
                    lines.append(f"{c.upper()} ID {i} 世界 XYZ = ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) mm")
            pair = doc["current_pair"]
            if pair:
                lines.append(f"主机接收时间差 {pair['arrival_skew_ms']:.1f} ms（非硬件同步）")
                for i, difference in pair["differences"].items():
                    d = difference["delta_xyz_mm"]
                    s = doc["statistics"][str(i)]
                    lines.append(f"ID {i} ΔXYZ=({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}) mm；"
                        f"距离={difference['distance_mm']:.3f} mm；角差={difference['rotation_difference_deg']:.3f}°\n"
                        f"  N={s['pair_count']}，位置差 RMS={s['rms_distance_mm']:.3f} mm，最大={s['max_distance_mm']:.3f} mm")
            if not pair or not pair["differences"]:
                lines.append("等待两台相机的共同标签，不显示过期差值。")
            lines.append("ID 0 是标定基准，其差值反映重复性；另一个共同标签可用于独立一致性检查。")
            self.report.setPlainText("\n".join(lines))

    def save_report(self):
        folder = self.session_dir or ROOT / "logs/dual_camera_world"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
            path.write_text(json.dumps(self.model.report(time.monotonic()), ensure_ascii=False, indent=2)+"\n")
            self.statusBar().showMessage(f"比较报告已保存：{path}")
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def open_rviz(self):
        if self.rviz is None or self.rviz.poll() is not None:
            logdir = ROOT / "logs/dual_camera_world"
            logdir.mkdir(parents=True, exist_ok=True)
            with (logdir / "rviz.log").open("a") as log:
                self.rviz = subprocess.Popen(["rviz2", "-d", str(ROOT / "config/dual_camera_world.rviz")],
                                             stdout=log, stderr=subprocess.STDOUT)

    def closeEvent(self, event):
        self.timer.stop()
        for source in self.sources.values():
            source.requestInterruption()
        for source in self.sources.values():
            source.wait()
        if self.model.calibrations:
            self.save_report()
        if self.rviz is not None and self.rviz.poll() is None:
            self.rviz.terminate()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/dual_camera_world.yaml")
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--calibrate-on-start", action="store_true", help="仅在 ID 0 已固定到已知世界位姿时使用")
    parser.add_argument("--load-session", type=Path, help="加载双相机外参目录；仅在两台相机未移动时使用")
    args = parser.parse_args()
    if args.calibrate_on_start and args.load_session:
        parser.error("--calibrate-on-start 与 --load-session 不能同时使用")
    if not args.calibrate_on_start and args.load_session is None:
        settings = yaml.safe_load(args.config.read_text())
        if settings.get("saved_session"):
            args.load_session = (args.config.resolve().parent / settings["saved_session"]).resolve()
    import rclpy
    rclpy.init(args=[])
    node = rclpy.create_node(f"dual_camera_world_{os.getpid()}")
    app = QApplication([sys.argv[0]])
    owned_driver = None
    try:
        # Allow discovery before deciding whether this application owns a driver.
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        if node.count_publishers("/gemini335l/color/image_raw") == 0:
            logdir = ROOT / "logs/dual_camera_world"
            logdir.mkdir(parents=True, exist_ok=True)
            with (logdir / "driver.log").open("a") as log:
                owned_driver = subprocess.Popen(["bash", str(ROOT / "scripts/run_gemini335l_ros2.sh")],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ensure_laser_off(node)
        window = ComparisonWindow(args.config.resolve(), node, args.load_session)
        window.show()
        if args.rviz:
            QTimer.singleShot(500, window.open_rviz)
        if args.calibrate_on_start:
            QTimer.singleShot(3000, window.begin)
        return app.exec_()
    finally:
        if owned_driver is not None and owned_driver.poll() is None:
            import signal
            os.killpg(owned_driver.pid, signal.SIGINT)
            try:
                owned_driver.wait(timeout=8)
            except subprocess.TimeoutExpired:
                owned_driver.terminate()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
