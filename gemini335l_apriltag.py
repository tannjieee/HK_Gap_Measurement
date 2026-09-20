#!/usr/bin/env python3
"""Gemini 335L RGB AprilTag pose viewer using calibrated intrinsics and ROS 2."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from apriltag_pose_gui import (
    OpenCVAprilTagEstimator, VideoWidget, draw_pose_overlay,
    load_intrinsics, load_pose_config, relative_poses,
)
from gemini335l_calibration import image_to_bgr
import cv2
import numpy as np
import yaml
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)


def checked_intrinsics(path):
    matrix, distortion, size = load_intrinsics(path)
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if document.get("distortion_model", "plumb_bob") != "plumb_bob":
        raise ValueError("需要 plumb_bob 模型的 RGB 内参")
    if (size is None or min(size) <= 0 or not np.isfinite(matrix).all()
            or not np.isfinite(distortion).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1]) or distortion.size != 5):
        raise ValueError("内参文件的矩阵、5 个畸变系数或图像尺寸无效")
    return matrix, distortion, size


def rvec_quaternion(rvec):
    vector = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(vector))
    xyz = vector * (np.sin(theta / 2) / theta if theta > 1e-12 else 0.5)
    return [*map(float, xyz), float(np.cos(theta / 2))]


@dataclass
class DetectionFrame:
    rgb: np.ndarray
    poses: list
    relatives: dict
    stamp: tuple
    frame_id: str
    received_at: float
    elapsed_ms: float
    config: object


def frame_document(frame):
    return {
        "image_stamp": {"sec": frame.stamp[0], "nanosec": frame.stamp[1]},
        "camera_frame": frame.frame_id,
        "image_size": [frame.rgb.shape[1], frame.rgb.shape[0]],
        "intrinsics_file": str(frame.config.intrinsics_file),
        "tag_family": frame.config.family, "tag_size_mm": frame.config.tag_size_mm,
        "tag_axes": "new +X = old +Z; new +Y = old -X; new +Z = old -Y",
        "pose_convention": "T_camera_tag; translation is in the camera optical frame",
        "tags": [{"id": p.tag_id, "translation_mm": p.translation_mm.tolist(),
                  "rotation_matrix": p.rotation_matrix.tolist(),
                  "rvec": p.rvec.reshape(3).tolist(), "quaternion_xyzw": rvec_quaternion(p.rvec),
                  "euler_xyz_deg": p.euler_xyz_deg.tolist(),
                  "reprojection_error_px": p.reprojection_error_px,
                  "corners": p.corners.tolist()} for p in frame.poses],
        "relative_poses": [{"id": p.tag_id, "reference_id": p.reference_id,
                            "translation_mm": p.translation_mm.tolist(),
                            "rotation_matrix": p.rotation_matrix.tolist(),
                            "euler_xyz_deg": p.euler_xyz_deg.tolist()}
                           for p in frame.relatives.values()],
    }


class Processor:
    def __init__(self, config):
        self.config = config
        self.matrix, self.distortion, self.size = checked_intrinsics(config.intrinsics_file)
        self.estimator = OpenCVAprilTagEstimator(self.matrix, self.distortion, config)
        self.expected_frame = config.raw.get("ros", {}).get("optical_frame", "gemini335l_color_optical_frame")

    def process(self, message):
        started = time.monotonic()
        if (message.width, message.height) != self.size:
            raise ValueError(f"图像 {message.width}×{message.height} 与内参 {self.size[0]}×{self.size[1]} 不一致，请使用对应分辨率")
        if self.expected_frame and message.header.frame_id != self.expected_frame:
            raise ValueError(f"图像坐标系 {message.header.frame_id!r} 与配置 {self.expected_frame!r} 不一致")
        bgr = image_to_bgr(message)
        poses = self.estimator.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        relatives = relative_poses(poses, self.config.reference_tag_id) if self.config.relative_enabled else {}
        overlay = draw_pose_overlay(bgr, poses, relatives, self.matrix, self.distortion, self.config)
        return DetectionFrame(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), poses, relatives,
                              (message.header.stamp.sec, message.header.stamp.nanosec),
                              message.header.frame_id, started, (time.monotonic() - started) * 1000,
                              self.config)


class RosPoseThread(QThread):
    error = pyqtSignal(str)

    def __init__(self, config, topic):
        super().__init__()
        self.topic = topic
        self.lock = threading.Lock()
        self.config = config
        self.latest = None
        self.last_error = ""

    def update_config(self, config):
        with self.lock:
            self.config, self.latest = config, None

    def take_latest(self):
        with self.lock:
            frame, self.latest = self.latest, None
        return frame

    def run(self):
        context = node = executor = None
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
            from geometry_msgs.msg import TransformStamped
            from tf2_ros import TransformBroadcaster
            from std_msgs.msg import String
            context = Context()
            rclpy.init(args=[], context=context)
            node = rclpy.create_node(f"gemini335l_apriltag_{os.getpid()}", context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            broadcaster = TransformBroadcaster(node)
            # IDs, units and camera timestamps are retained in the structured report.
            publisher = node.create_publisher(String, "/gemini335l/apriltag/poses_json", 1)
            processor = None

            def on_image(message):
                nonlocal processor
                try:
                    with self.lock:
                        config = self.config
                    if processor is None or processor.config is not config:
                        processor = Processor(config)
                    frame = processor.process(message)
                    with self.lock:
                        if self.config is not config:
                            return
                        self.latest = frame
                    ros = config.raw.get("ros", {})
                    if ros.get("publish_tf", True):
                        transforms = []
                        for pose in frame.poses:
                            transform = TransformStamped()
                            transform.header = message.header
                            transform.child_frame_id = ros.get("tag_frame_prefix", "gemini335l_tag_") + str(pose.tag_id)
                            t = pose.tvec.reshape(3)
                            transform.transform.translation.x = float(t[0])
                            transform.transform.translation.y = float(t[1])
                            transform.transform.translation.z = float(t[2])
                            q = rvec_quaternion(pose.rvec)
                            (transform.transform.rotation.x, transform.transform.rotation.y,
                             transform.transform.rotation.z, transform.transform.rotation.w) = q
                            transforms.append(transform)
                        if transforms:
                            broadcaster.sendTransform(transforms)
                    publisher.publish(String(data=json.dumps(frame_document(frame), ensure_ascii=False)))
                    self.last_error = ""
                except Exception as exc:
                    if str(exc) != self.last_error:
                        self.error.emit(str(exc))
                        self.last_error = str(exc)

            node.create_subscription(Image, self.topic, on_image,
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            while context.ok() and not self.isInterruptionRequested():
                executor.spin_once(timeout_sec=0.05)
        except Exception as exc:
            self.error.emit(f"ROS 接口失败：{exc}")
        finally:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            if context is not None and context.ok():
                context.shutdown()


class MainWindow(QMainWindow):
    def __init__(self, config, topic):
        super().__init__()
        self.config, self.topic = config, topic
        checked_intrinsics(config.intrinsics_file)
        self.latest_frame = None
        self.setWindowTitle("GEMINI 335L — AprilTag 6D 位姿")
        self.resize(1300, 820)
        self.video = VideoWidget()
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.1, 10000)
        self.size_spin.setDecimals(3)
        self.size_spin.setValue(config.tag_size_mm)
        self.ids_edit = QLineEdit(",".join(map(str, config.tag_ids)))
        self.ids_edit.setPlaceholderText("留空识别全部 ID")
        self.reference_edit = QLineEdit("" if config.reference_tag_id is None else str(config.reference_tag_id))
        self.reference_edit.setPlaceholderText("留空自动选择")
        form = QFormLayout()
        form.addRow("标签边长（mm）", self.size_spin)
        form.addRow("检测 ID（逗号分隔）", self.ids_edit)
        form.addRow("相对位姿参考 ID", self.reference_edit)
        apply_button = QPushButton("应用标签参数")
        apply_button.clicked.connect(self.apply_settings)
        load_button = QPushButton("选择 RGB 内参文件")
        load_button.clicked.connect(self.load_calibration)
        self.calibration_label = QLabel()
        self.calibration_label.setWordWrap(True)
        self.describe_calibration()
        convention = QLabel("标签轴：新 X = 旧 Z；新 Y = −旧 X；新 Z = −旧 Y\n"
                            "相机坐标：X 向右、Y 向下、Z 向前；位置显示 mm\n"
                            "RPY：标签相对相机的 XYZ 欧拉角，单位 °")
        convention.setWordWrap(True)
        self.status = QLabel("等待 ROS RGB 图像…")
        self.status.setWordWrap(True)
        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        self.save_button = QPushButton("保存当前画面和位姿 JSON")
        self.save_button.clicked.connect(self.save_frame)
        self.save_button.setEnabled(False)
        side = QVBoxLayout()
        side.addWidget(QLabel(f"字典：{config.family}\n图像：{topic}"))
        side.addLayout(form)
        for widget in (apply_button, load_button, self.calibration_label, convention,
                       self.status, self.results, self.save_button):
            side.addWidget(widget)
        panel = QWidget()
        panel.setLayout(side)
        panel.setMinimumWidth(450)
        layout = QHBoxLayout()
        layout.addWidget(self.video, 3)
        layout.addWidget(panel, 2)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.worker = RosPoseThread(config, topic)
        self.worker.error.connect(lambda message: self.statusBar().showMessage(message))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)
        self.worker.start()

    def describe_calibration(self):
        k, _, size = checked_intrinsics(self.config.intrinsics_file)
        self.calibration_label.setText(f"内参：{self.config.intrinsics_file.name}\n"
            f"{size[0]} × {size[1]} | fx={k[0,0]:.3f}, fy={k[1,1]:.3f}\n"
            f"cx={k[0,2]:.3f}, cy={k[1,2]:.3f}")

    def set_config(self, config):
        self.config = config
        self.latest_frame = None
        self.worker.update_config(config)
        self.results.clear()
        self.video.clear()
        self.save_button.setEnabled(False)
        self.describe_calibration()
        self.statusBar().showMessage("参数已应用")

    def apply_settings(self):
        try:
            ids = tuple(int(s.strip()) for s in self.ids_edit.text().replace("，", ",").split(",") if s.strip())
            reference = int(self.reference_edit.text()) if self.reference_edit.text().strip() else None
            if any(i < 0 for i in ids) or (reference is not None and reference < 0):
                raise ValueError("ID 必须为非负整数")
            self.set_config(replace(self.config, tag_ids=ids, reference_tag_id=reference,
                                    tag_size_mm=self.size_spin.value(), axis_length_mm=self.size_spin.value()/2))
        except ValueError as exc:
            QMessageBox.warning(self, "标签参数无效", str(exc))

    def load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 335L RGB 内参", str(self.config.intrinsics_file.parent),
                                             "内参 (*.yaml *.yml *.json)")
        if path:
            try:
                checked_intrinsics(path)
                self.set_config(replace(self.config, intrinsics_file=Path(path).resolve()))
            except Exception as exc:
                QMessageBox.warning(self, "内参无效", str(exc))

    def refresh(self):
        frame = self.worker.take_latest()
        if frame is not None and frame.config is self.config:
            self.latest_frame = frame
            rgb = frame.rgb
            self.video.set_image(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy())
            self.status.setText(f"{rgb.shape[1]} × {rgb.shape[0]} | 识别 {len(frame.poses)} 个标签 | 处理 {frame.elapsed_ms:.1f} ms")
            lines = []
            for p in frame.poses:
                lines += [f"ID {p.tag_id}（相对于 RGB 相机）",
                          "  ".join(f"{a}={v:+.2f}" for a, v in zip("XYZ", p.translation_mm)) + " mm",
                          "  ".join(f"{a}={v:+.2f}°" for a, v in zip(("R", "P", "Y"), p.euler_xyz_deg)),
                          f"重投影误差：{p.reprojection_error_px:.3f} px"]
                relative = frame.relatives.get(p.tag_id)
                if relative is not None:
                    lines += [f"相对于 ID {relative.reference_id}：",
                              "  ".join(f"{a}={v:+.2f}" for a, v in zip("XYZ", relative.translation_mm)) + " mm",
                              "RPY=" + ", ".join(f"{v:+.2f}°" for v in relative.euler_xyz_deg)]
                lines.append("")
            self.results.setPlainText("\n".join(lines) if lines else "未检测到所选 ID；请让完整标签进入画面，检查字典和 ID 设置。")
            self.save_button.setEnabled(True)
        if self.latest_frame is None or time.monotonic() - self.latest_frame.received_at > 1:
            self.status.setText("等待新图像，请检查驱动、图像尺寸及 ROS_DOMAIN_ID")
            self.results.clear()
            self.video.clear()
            self.save_button.setEnabled(False)

    def save_frame(self):
        frame = self.latest_frame
        if frame is None or time.monotonic() - frame.received_at > 1:
            return
        folder = Path(__file__).parent / "captures" / "gemini335l_apriltag"
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(folder / f"{stem}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
                raise OSError("图像保存失败")
            (folder / f"{stem}.json").write_text(json.dumps(frame_document(frame), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            self.statusBar().showMessage(f"已保存：{folder / stem}")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def closeEvent(self, event):
        self.timer.stop()
        self.worker.requestInterruption()
        self.worker.wait()
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/gemini335l_apriltag.yaml")
    parser.add_argument("--topic", help="覆盖配置中的 ROS RGB 图像话题")
    parser.add_argument("--intrinsics", type=Path, help="覆盖配置中的内参文件")
    args = parser.parse_args(argv)
    app = QApplication([sys.argv[0]])
    try:
        config = load_pose_config(args.config)
        if args.intrinsics:
            config = replace(config, intrinsics_file=args.intrinsics.expanduser().resolve())
        topic = args.topic or config.raw.get("ros", {}).get("image_topic", "/gemini335l/color/image_raw")
        window = MainWindow(config, topic)
    except Exception as exc:
        QMessageBox.critical(None, "无法启动 AprilTag", str(exc))
        return 1
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
