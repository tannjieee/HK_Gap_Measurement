#!/usr/bin/env python3
"""Chessboard intrinsics calibration from the Gemini 335L raw RGB ROS stream."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

_qt_keys = ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR")
_qt_env = {key: os.environ.get(key) for key in _qt_keys}
import cv2
from calibration_core import (
    BoardSpec, ChessboardDetection, calibrate_samples, calibration_quality,
    detect_chessboard, make_sample, novelty_score, save_calibration,
)
for _key, _value in _qt_env.items():
    if _value is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _value
cv2.setNumThreads(2)


def image_to_bgr(message) -> np.ndarray:
    """Decode sensor_msgs/Image, including row padding; return an owned image."""
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    if message.encoding not in channels:
        raise ValueError(f"不支持的图像编码：{message.encoding}，请订阅 RGB image_raw")
    count = channels[message.encoding]
    width, height, step = int(message.width), int(message.height), int(message.step)
    if width <= 0 or height <= 0 or step < width * count:
        raise ValueError("图像尺寸或 step 无效")
    data = np.frombuffer(message.data, dtype=np.uint8)
    if data.size != height * step:
        raise ValueError("图像 data 长度与 height × step 不一致")
    pixels = data.reshape(height, step)[:, :width * count].reshape(height, width, count)
    if message.encoding == "bgr8":
        return pixels.copy()
    code = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
            "bgra8": cv2.COLOR_BGRA2BGR, "mono8": cv2.COLOR_GRAY2BGR}[message.encoding]
    return cv2.cvtColor(pixels, code)


def camera_geometry(info) -> tuple:
    """Include same-size ROI/binning/intrinsics changes in the session identity."""
    roi = info.roi
    return (info.width, info.height, info.header.frame_id, info.binning_x,
            info.binning_y, roi.x_offset, roi.y_offset, roi.width, roi.height,
            roi.do_rectify, info.distortion_model, tuple(info.k), tuple(info.d),
            tuple(info.r), tuple(info.p))


@dataclass
class Packet:
    bgr: np.ndarray
    gray: np.ndarray
    detection: ChessboardDetection
    board: BoardSpec
    received_at: float
    key: tuple
    geometry: tuple | None
    metadata: dict

    @property
    def image_size(self):
        return self.gray.shape[1], self.gray.shape[0]


def rejection_reason(packet, samples, keys, geometry, automatic, now=None):
    now = time.monotonic() if now is None else now
    if packet is None or now - packet.received_at > 1.5:
        return "没有新鲜图像，请检查驱动与话题"
    if packet.geometry is None:
        return "等待与图像匹配的 CameraInfo"
    if samples and packet.geometry != geometry:
        return "相机成像配置已变化，请清空样本后重新采集"
    if packet.key in keys:
        return "此帧已采集，请移动棋盘后再采集"
    detection = packet.detection
    if not detection.found:
        return "未检测到完整棋盘，请检查方格数并让整块棋盘入镜"
    if detection.coverage < 0.02 or detection.minimum_spacing < 8:
        return "棋盘太小：请靠近相机，让相邻角点至少间隔 8 像素"
    if automatic and detection.sharpness < 50:
        return "画面不够清晰，请稳定棋盘或改善照明"
    if automatic and novelty_score(detection.descriptor, [s.descriptor for s in samples]) < 1:
        return "姿态相近，请改变棋盘的位置、距离或倾斜角度"
    return ""


class RosCapture(QThread):
    """Depth-one ROS subscription and latest-packet mailbox avoid GUI backlogs."""
    error = pyqtSignal(str)

    def __init__(self, topic, info_topic, board):
        super().__init__()
        self.topic, self.info_topic = topic, info_topic
        self.lock = threading.Lock()
        self.board = board
        self.latest = None
        self.info = None
        self.last_detection = 0.0
        self.last_error = ""

    def set_board(self, board):
        with self.lock:
            self.board = board
            self.latest = None

    def take_latest(self):
        with self.lock:
            packet, self.latest = self.latest, None
        return packet

    def on_image(self, message):
        received_at = time.monotonic()
        if received_at - self.last_detection < 0.15:
            return
        self.last_detection = received_at
        try:
            with self.lock:
                board = self.board
            bgr = image_to_bgr(message)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            detection = detect_chessboard(gray, board, exhaustive=False)
            info = self.info
            geometry = None
            metadata = {"image_topic": self.topic, "camera_info_topic": self.info_topic,
                        "frame_id": message.header.frame_id, "encoding": message.encoding,
                        "width": message.width, "height": message.height}
            if (info is not None and (info.width, info.height) == (message.width, message.height)
                    and info.header.frame_id == message.header.frame_id):
                geometry = (message.encoding, camera_geometry(info))
                metadata["source_camera_info"] = {
                    "K": list(info.k), "D": list(info.d), "R": list(info.r), "P": list(info.p),
                    "distortion_model": info.distortion_model,
                    "binning": [info.binning_x, info.binning_y],
                    "roi": {k: getattr(info.roi, k) for k in
                            ("x_offset", "y_offset", "width", "height", "do_rectify")}}
            stamp = message.header.stamp
            packet = Packet(bgr, gray, detection, board, received_at,
                            (message.header.frame_id, stamp.sec, stamp.nanosec), geometry, metadata)
            with self.lock:
                self.latest = packet
            self.last_error = ""
        except Exception as exc:
            if str(exc) != self.last_error:
                self.last_error = str(exc)
                self.error.emit(str(exc))

    def run(self):
        context = node = executor = None
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            from sensor_msgs.msg import Image, CameraInfo
            context = Context()
            rclpy.init(args=[], context=context)
            node = rclpy.create_node(f"gemini335l_rgb_calibration_{os.getpid()}", context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            node.create_subscription(CameraInfo, self.info_topic,
                                     lambda msg: setattr(self, "info", msg), qos)
            node.create_subscription(Image, self.topic, self.on_image, qos)
            while not self.isInterruptionRequested() and context.ok():
                executor.spin_once(timeout_sec=0.05)
        except Exception as exc:
            self.error.emit(f"ROS 接收失败：{exc}")
        finally:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            if context is not None and context.ok():
                context.shutdown()


class SolveThread(QThread):
    solved = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, samples, board, minimum, output, metadata, keys):
        super().__init__()
        self.samples, self.board, self.minimum = list(samples), board, minimum
        self.output, self.metadata, self.keys = output, metadata.copy(), list(keys)

    def run(self):
        try:
            result = calibrate_samples(self.samples, self.board, minimum_samples=self.minimum)
            paths = save_calibration(self.output, result, self.board, self.samples,
                                     camera_name="gemini335l_color")
            session = paths[0].with_suffix(".session.json")
            session.write_text(json.dumps({**self.metadata, "sample_stamps": self.keys,
                "note": "source_camera_info is the driver input, not the solved calibration"},
                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.solved.emit((result, paths, session))
        except Exception as exc:
            self.failed.emit(str(exc))


class Preview(QWidget):
    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.setMinimumSize(640, 480)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.image.isNull():
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "等待 GEMINI 335L RGB 图像…")
        else:
            size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
            x, y = (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
            painter.drawImage(x, y, self.image.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class CalibrationWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.samples, self.keys = [], []
        self.geometry = None
        self.metadata = {}
        self.packet = self.solver = None
        self.last_capture = 0.0
        self.setWindowTitle("GEMINI 335L — RGB 相机内参标定")
        self.resize(1180, 760)
        self.preview = Preview()
        self.squares_x, self.squares_y = QSpinBox(), QSpinBox()
        for spin, value in ((self.squares_x, args.squares_x), (self.squares_y, args.squares_y)):
            spin.setRange(3, 40)
            spin.setValue(value)
        self.square_mm = QDoubleSpinBox()
        self.square_mm.setRange(0.1, 1000)
        self.square_mm.setDecimals(3)
        self.square_mm.setValue(args.square_size_mm)
        form = QFormLayout()
        form.addRow("横向方格数（含黑白格）", self.squares_x)
        form.addRow("纵向方格数（含黑白格）", self.squares_y)
        form.addRow("每格边长（mm）", self.square_mm)
        self.board_label = QLabel()
        self.board_label.setWordWrap(True)
        self.count = QLabel()
        self.live = QLabel("等待图像和 CameraInfo；请先启动相机 ROS 2 驱动")
        self.live.setWordWrap(True)
        self.auto = QCheckBox(f"自动采集（目标 {args.target_samples} 张不同姿态）")
        self.capture_button = QPushButton("采集当前视图（空格）")
        self.capture_button.setShortcut("Space")
        self.undo_button = QPushButton("撤销最后一张")
        self.clear_button = QPushButton("清空样本 / 修改棋盘规格")
        self.solve_button = QPushButton("标定并保存内参")
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlainText(
            "使用平整棋盘格，确认方格数量和实测边长。\n"
            "让完整棋盘依次覆盖中心、四角和边缘，改变远近，左右/前后倾斜。\n"
            "至少采集 12 张，推荐 25–30 张；每次移动后短暂停稳。\n"
            "采样期间保持分辨率、ROI、镜像和相机设置不变。\n"
            "输出 fx/fy/cx/cy、5 个畸变系数、逐张误差和原始样本。")
        panel = QVBoxLayout()
        panel.addLayout(form)
        for widget in (self.board_label, self.live, self.count, self.auto, self.capture_button,
                       self.undo_button, self.clear_button, self.solve_button, self.report):
            panel.addWidget(widget)
        side = QWidget()
        side.setLayout(panel)
        side.setMinimumWidth(390)
        layout = QHBoxLayout()
        layout.addWidget(self.preview, 3)
        layout.addWidget(side, 2)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.receiver = RosCapture(args.topic, args.camera_info_topic, self.board())
        self.receiver.error.connect(self.on_error)
        for spin in (self.squares_x, self.squares_y, self.square_mm):
            spin.valueChanged.connect(self.board_changed)
        self.capture_button.clicked.connect(lambda: self.capture(False))
        self.undo_button.clicked.connect(self.undo)
        self.clear_button.clicked.connect(self.clear)
        self.solve_button.clicked.connect(self.solve)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(80)
        self.update_controls()
        self.receiver.start()

    def board(self):
        return BoardSpec(self.squares_x.value(), self.squares_y.value(), self.square_mm.value())

    def board_changed(self):
        self.packet = None
        self.receiver.set_board(self.board())
        self.update_controls()

    def update_controls(self):
        busy = self.solver is not None
        for spin in (self.squares_x, self.squares_y, self.square_mm):
            spin.setEnabled(not self.samples and not busy)
        b = self.board()
        self.board_label.setText(f"内角点：{b.inner_corners_x} × {b.inner_corners_y}；方格边长 {b.square_size_mm:g} mm")
        self.count.setText(f"已采集 {len(self.samples)} 张 / 最少 {self.args.minimum_samples} 张")
        self.solve_button.setEnabled(len(self.samples) >= self.args.minimum_samples and not busy)
        self.capture_button.setEnabled(not busy)
        self.auto.setEnabled(not busy)
        self.undo_button.setEnabled(bool(self.samples) and not busy)
        self.clear_button.setEnabled(not busy)

    def refresh(self):
        packet = self.receiver.take_latest()
        if packet is not None and packet.board == self.board():
            self.packet = packet
            shown = packet.bgr.copy()
            if packet.detection.found:
                cv2.drawChessboardCorners(shown, self.board().pattern_size, packet.detection.corners, True)
            rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
            self.preview.image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
            self.preview.update()
            d = packet.detection
            self.live.setText(f"{packet.image_size[0]} × {packet.image_size[1]} | "
                              f"{'检测到完整棋盘' if d.found else '未检测到完整棋盘'}\n"
                              f"覆盖率 {d.coverage:.1%} | 清晰度 {d.sharpness:.0f}\n"
                              f"最小角点间距 {d.minimum_spacing:.1f} px（采样要求 ≥ 8）")
            if self.auto.isChecked() and time.monotonic() - self.last_capture >= 0.8:
                self.capture(True)
        if self.packet is None or time.monotonic() - self.packet.received_at > 1.5:
            self.live.setText(f"等待实时图像：{self.args.topic}\n请检查驱动是否运行及 ROS_DOMAIN_ID 是否一致")

    def capture(self, automatic):
        if self.solver is not None:
            return
        if automatic and len(self.samples) >= self.args.target_samples:
            self.auto.setChecked(False)
            return
        reason = rejection_reason(self.packet, self.samples, self.keys, self.geometry, automatic)
        if reason:
            self.statusBar().showMessage(reason)
            return
        p = self.packet
        if not self.samples:
            self.geometry, self.metadata = p.geometry, p.metadata.copy()
        self.samples.append(make_sample(p.detection, p.image_size, gray_image=p.gray,
                                       frame_number=len(self.samples) + 1,
                                       source_name=f"{self.args.topic}@{p.key[1]}.{p.key[2]:09d}"))
        self.keys.append(p.key)
        self.last_capture = time.monotonic()
        self.statusBar().showMessage(f"已采集第 {len(self.samples)} 张，请改变位置、距离和倾斜角度")
        if len(self.samples) >= self.args.target_samples:
            self.auto.setChecked(False)
        self.update_controls()

    def undo(self):
        self.auto.setChecked(False)
        if self.samples:
            self.samples.pop()
            self.keys.pop()
        self.update_controls()

    def clear(self):
        self.auto.setChecked(False)
        self.samples.clear()
        self.keys.clear()
        self.geometry, self.metadata = None, {}
        self.report.clear()
        self.update_controls()

    def solve(self):
        self.auto.setChecked(False)
        default = self.args.output or (Path(__file__).parent / "calibration_results" / "gemini335l_rgb" /
            f"gemini335l_rgb_{datetime.now():%Y%m%d_%H%M%S_%f}.yaml")
        default = Path(default).expanduser().resolve()
        try:
            default.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.on_error(f"无法创建输出目录：{exc}")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 RGB 内参", str(default), "YAML (*.yaml *.yml)")
        if not path:
            return
        self.solver = SolveThread(self.samples, self.board(), self.args.minimum_samples,
                                  Path(path), self.metadata, self.keys)
        self.solver.solved.connect(self.on_solved)
        self.solver.failed.connect(self.on_error)
        self.solver.finished.connect(self.solve_finished)
        self.update_controls()
        self.statusBar().showMessage("正在后台标定、剔除异常视图并保存…")
        self.solver.start()

    def on_solved(self, payload):
        result, paths, session = payload
        k, d = result.camera_matrix, result.distortion_coefficients.reshape(-1)
        _, quality = calibration_quality(result)
        self.report.setPlainText(
            f"{quality}\nRMS = {result.rms_error:.4f} px\n"
            f"分辨率：{result.image_size[0]} × {result.image_size[1]}\n"
            f"fx = {k[0,0]:.6f}\nfy = {k[1,1]:.6f}\ncx = {k[0,2]:.6f}\ncy = {k[1,2]:.6f}\n"
            + "\n".join(f"{name} = {value:.9f}" for name, value in zip(("k1", "k2", "p1", "p2", "k3"), d))
            + f"\n使用 {len(result.used_indices)} 张，剔除 {len(result.rejected_indices)} 张\n"
            + "\n".join(f"样本 {idx+1}: {err:.4f} px" for idx, err in zip(result.used_indices, result.per_view_errors))
            + f"\n\nYAML：{paths[0]}\nJSON：{paths[1]}\n样本：{paths[2]}\n来源记录：{session}")
        self.statusBar().showMessage("标定结果已保存")

    def on_error(self, message):
        self.statusBar().showMessage(message)
        self.report.append(message)

    def solve_finished(self):
        self.solver.deleteLater()
        self.solver = None
        self.update_controls()

    def closeEvent(self, event):
        if self.solver is not None:
            QMessageBox.information(self, "正在保存", "请等待标定和文件保存完成后再关闭。")
            event.ignore()
            return
        self.timer.stop()
        self.receiver.requestInterruption()
        self.receiver.wait()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/gemini335l/color/image_raw")
    parser.add_argument("--camera-info-topic", default=None)
    parser.add_argument("--squares-x", type=int, default=12, help="横向方格数（不是内角点数）")
    parser.add_argument("--squares-y", type=int, default=9, help="纵向方格数（不是内角点数）")
    parser.add_argument("--square-size-mm", type=float, default=15.0)
    parser.add_argument("--minimum-samples", type=int, default=12)
    parser.add_argument("--target-samples", type=int, default=30)
    parser.add_argument("--output", type=Path, help="默认保存文件位置；保存前仍可在窗口修改")
    args = parser.parse_args(argv)
    if not (3 <= args.squares_x <= 40 and 3 <= args.squares_y <= 40
            and 0.1 <= args.square_size_mm <= 1000):
        parser.error("方格数必须为 3–40，边长必须为 0.1–1000 mm")
    if not 3 <= args.minimum_samples <= args.target_samples:
        parser.error("要求 3 ≤ minimum-samples ≤ target-samples")
    if args.camera_info_topic is None:
        args.camera_info_topic = args.topic.rsplit("/", 1)[0] + "/camera_info"
    return args


def main(argv=None):
    args = parse_args(argv)
    app = QApplication([sys.argv[0]])
    window = CalibrationWindow(args)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
