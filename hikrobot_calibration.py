#!/usr/bin/env python3
"""Interactive and offline intrinsic calibration for a HIKROBOT camera."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt5.QtCore import QRect, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QCloseEvent, QImage, QKeyEvent, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# The opencv-python wheel may point Qt at its bundled plugin directory. Preserve
# the desktop/PyQt5 configuration because this application never uses cv2.imshow.
_QT_ENVIRONMENT_KEYS = ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR")
_qt_environment = {key: os.environ.get(key) for key in _QT_ENVIRONMENT_KEYS}
from calibration_core import (  # noqa: E402
    BoardSpec,
    CalibrationSample,
    ChessboardDetection,
    calibrate_samples,
    calibration_quality,
    detect_chessboard,
    make_sample,
    novelty_score,
    save_calibration,
)
import cv2  # noqa: E402

for _key, _value in _qt_environment.items():
    if _value is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _value

cv2.setNumThreads(2)


@dataclass
class FramePacket:
    display_image: QImage
    gray_image: np.ndarray
    detection: ChessboardDetection
    image_size: tuple[int, int]
    frame_number: int
    received_at: float


class VideoWidget(QWidget):
    """Aspect-ratio-preserving image display."""

    def __init__(self) -> None:
        super().__init__()
        self.image = QImage()
        self.setMinimumSize(720, 576)
        self.setStyleSheet("background: #111;")

    def set_image(self, image: QImage) -> None:
        self.image = image
        self.update()

    def clear(self) -> None:
        self.image = QImage()
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        if self.image.isNull():
            painter.setPen(QColor("#aaaaaa"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待相机画面")
            return
        size = self.image.size()
        size.scale(self.size(), Qt.KeepAspectRatio)
        target = QRect(
            (self.width() - size.width()) // 2,
            (self.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(target, self.image)


class CalibrationCaptureThread(QThread):
    frame_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self, controller, board: BoardSpec
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.controller = controller
        self.board = board
        self.stop_event = threading.Event()
        self.last_exhaustive_search = 0.0

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                iteration_started = time.monotonic()
                frame = self.controller.read_rgb_frame()
                if frame is None:
                    continue
                rgb_bytes, width, height, frame_number = frame
                rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                    height, width, 3
                )
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                now = time.monotonic()
                exhaustive = now - self.last_exhaustive_search >= 3.0
                if exhaustive:
                    self.last_exhaustive_search = now
                detection = detect_chessboard(
                    gray,
                    self.board,
                    maximum_detection_dimension=960,
                    exhaustive=exhaustive,
                )
                display = np.ascontiguousarray(rgb.copy())
                if detection.found and detection.corners is not None:
                    cv2.drawChessboardCorners(
                        display,
                        self.board.pattern_size,
                        detection.corners,
                        True,
                    )
                image = QImage(
                    display.data,
                    width,
                    height,
                    int(display.strides[0]),
                    QImage.Format_RGB888,
                ).copy()
                self.frame_ready.emit(
                    FramePacket(
                        display_image=image,
                        gray_image=gray,
                        detection=detection,
                        image_size=(width, height),
                        frame_number=frame_number,
                        received_at=time.monotonic(),
                    )
                )
                remaining = 0.20 - (time.monotonic() - iteration_started)
                if remaining > 0:
                    self.stop_event.wait(remaining)
        except Exception as exc:
            self.failed.emit(str(exc))


class CalibrationWindow(QMainWindow):
    MINIMUM_COVERAGE = 0.02
    MINIMUM_CORNER_SPACING = 8.0
    AUTO_MINIMUM_SHARPNESS = 50.0
    AUTO_CAPTURE_INTERVAL_S = 0.8
    AUTO_NOVELTY_THRESHOLD = 1.0

    def __init__(
        self,
        sdk,
        board: BoardSpec,
        *,
        minimum_samples: int,
        target_samples: int,
        default_output: Path,
        device_index: int,
        serial: str | None,
        auto_capture: bool,
        reject_outliers: bool,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        from camera_features import CameraFeaturePanel
        from hikrobot_gui import CameraController

        self.sdk = sdk
        self.board = board
        self.minimum_samples = minimum_samples
        self.target_samples = target_samples
        self.default_output = default_output
        self.requested_device_index = device_index
        self.requested_serial = serial
        self.reject_outliers = reject_outliers

        self.controller = CameraController()
        self.device_list = None
        self.devices = []
        self.capture_thread: CalibrationCaptureThread | None = None
        self._capture_generation = 0
        self.latest_packet: FramePacket | None = None
        self.samples: list[CalibrationSample] = []
        self.last_auto_capture_at = 0.0
        self.camera_name = "hikrobot_camera"
        self._feature_change_failed = False
        self._feature_change_stopped_stream = False
        self._feature_change_invalidates_samples = False

        self.setWindowTitle("HIKROBOT 棋盘格相机内参标定")
        self.resize(1380, 860)
        self.video = VideoWidget()
        self.device_combo = QComboBox()
        self.refresh_button = QPushButton("刷新设备")
        self.connect_button = QPushButton("连接")
        self.stream_button = QPushButton("暂停取流")
        self.board_label = QLabel()
        self.board_label.setWordWrap(True)
        self.detection_label = QLabel("尚未检测")
        self.detection_label.setWordWrap(True)
        self.auto_capture = QCheckBox("自动采集不同姿态")
        self.auto_capture.setChecked(auto_capture)
        self.capture_button = QPushButton("采集当前视图（空格）")
        self.capture_button.setEnabled(False)
        self.undo_button = QPushButton("撤销上一张")
        self.clear_button = QPushButton("清空样本")
        self.lens_adjusted_button = QPushButton("镜头已调整，清空样本")
        self.progress = QProgressBar()
        self.progress.setRange(0, target_samples)
        self.sample_label = QLabel()
        self.calibrate_button = QPushButton("标定并保存内参…")
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setPlaceholderText("完成至少 12 个不同姿态后，可计算内参。")
        self.feature_panel = CameraFeaturePanel()
        self.feature_panel.before_change = self._before_feature_change
        self.feature_panel.after_change = self._after_feature_change

        self._build_ui()
        self._connect_signals()
        self._update_board_text()
        self._update_sample_state()
        self._set_connected_state(False)
        self.refresh_devices()
        if self.devices:
            QTimer.singleShot(0, self.connect_camera)

    def _build_ui(self) -> None:
        connection_box = QGroupBox("相机")
        connection_layout = QVBoxLayout(connection_box)
        connection_layout.addWidget(self.device_combo)
        connection_buttons = QHBoxLayout()
        connection_buttons.addWidget(self.refresh_button)
        connection_buttons.addWidget(self.connect_button)
        connection_layout.addLayout(connection_buttons)
        connection_layout.addWidget(self.stream_button)

        board_box = QGroupBox("标定板")
        board_layout = QVBoxLayout(board_box)
        board_layout.addWidget(self.board_label)
        tip = QLabel(
            "采样建议：让标定板覆盖画面中心、四角和边缘，并改变远近、旋转和倾斜角度。"
        )
        tip.setWordWrap(True)
        board_layout.addWidget(tip)
        lens_tip = QLabel(
            "<b>6 mm 定焦镜头：</b>软件参数不能增加光学景深。请先在实际工作"
            "距离完成物理调焦；有光圈环时适当收小光圈并补光，尽量保持低增益、"
            "短曝光。标定期间不要再转动对焦环或光圈环。"
        )
        lens_tip.setWordWrap(True)
        lens_tip.setStyleSheet(
            "padding: 7px; background: #fff7df; border: 1px solid #e6cf8b;"
        )
        board_layout.addWidget(lens_tip)
        board_layout.addWidget(self.lens_adjusted_button)

        sample_box = QGroupBox("采样")
        sample_layout = QVBoxLayout(sample_box)
        sample_layout.addWidget(self.detection_label)
        sample_layout.addWidget(self.auto_capture)
        sample_layout.addWidget(self.capture_button)
        sample_layout.addWidget(self.progress)
        sample_layout.addWidget(self.sample_label)
        sample_buttons = QHBoxLayout()
        sample_buttons.addWidget(self.undo_button)
        sample_buttons.addWidget(self.clear_button)
        sample_layout.addLayout(sample_buttons)
        sample_layout.addWidget(self.calibrate_button)

        result_box = QGroupBox("标定结果")
        result_layout = QVBoxLayout(result_box)
        result_layout.addWidget(self.results)

        calibration_layout = QVBoxLayout()
        calibration_layout.addWidget(board_box)
        calibration_layout.addWidget(sample_box)
        calibration_layout.addWidget(result_box, 1)
        calibration_tab = QWidget()
        calibration_tab.setLayout(calibration_layout)

        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(calibration_tab, "标定")
        self.main_tabs.addTab(self.feature_panel, "相机参数")

        side_layout = QVBoxLayout()
        side_layout.addWidget(connection_box)
        side_layout.addWidget(self.main_tabs, 1)
        side_panel = QWidget()
        side_panel.setLayout(side_layout)
        side_panel.setMinimumWidth(460)
        side_panel.setMaximumWidth(610)

        layout = QHBoxLayout()
        layout.addWidget(self.video, 1)
        layout.addWidget(side_panel)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("未连接")

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.connect_button.clicked.connect(self._toggle_connection)
        self.stream_button.clicked.connect(self._toggle_stream)
        self.capture_button.clicked.connect(self.capture_current_view)
        self.undo_button.clicked.connect(self.undo_sample)
        self.clear_button.clicked.connect(self.clear_samples)
        self.lens_adjusted_button.clicked.connect(self.mark_lens_adjusted)
        self.calibrate_button.clicked.connect(self.calibrate_and_save)
        self.feature_panel.status_changed.connect(self.statusBar().showMessage)
        self.feature_panel.change_failed.connect(self._feature_change_failure)

    def _update_board_text(self) -> None:
        self.board_label.setText(
            f"<b>方格数：{self.board.squares_x} × {self.board.squares_y}</b><br>"
            f"OpenCV 内角点：{self.board.inner_corners_x} × "
            f"{self.board.inner_corners_y} = {self.board.point_count} 点<br>"
            f"方格边长：{self.board.square_size_mm:g} mm"
        )

    def _set_connected_state(self, connected: bool) -> None:
        self.connect_button.setText("断开" if connected else "连接")
        self.device_combo.setEnabled(not connected)
        self.refresh_button.setEnabled(not connected)
        self.stream_button.setEnabled(connected)
        if not connected:
            self.stream_button.setText("开始取流")
            self.capture_button.setEnabled(False)

    def _update_sample_state(self) -> None:
        count = len(self.samples)
        self.progress.setMaximum(max(self.target_samples, count, 1))
        self.progress.setValue(count)
        self.sample_label.setText(
            f"已采集 {count} / 推荐 {self.target_samples} 张；"
            f"至少需要 {self.minimum_samples} 张"
        )
        self.undo_button.setEnabled(count > 0)
        self.clear_button.setEnabled(count > 0)
        self.calibrate_button.setEnabled(count >= self.minimum_samples)

    def refresh_devices(self) -> None:
        if self.controller.device_open:
            return
        try:
            self.device_list, self.devices = self.sdk.enumerate_devices()
            self.device_combo.clear()
            selected_combo_index = -1
            for combo_index, device in enumerate(self.devices):
                transport, model, serial = self.sdk.device_identity(device)
                self.device_combo.addItem(
                    f"[{combo_index}] {model} ({serial}) - {transport}", combo_index
                )
                if self.requested_serial and serial == self.requested_serial:
                    selected_combo_index = combo_index
                elif (
                    not self.requested_serial
                    and combo_index == self.requested_device_index
                ):
                    selected_combo_index = combo_index
            if selected_combo_index >= 0:
                self.device_combo.setCurrentIndex(selected_combo_index)
            if self.devices:
                self.statusBar().showMessage(f"发现 {len(self.devices)} 台相机")
            else:
                self.statusBar().showMessage("未发现相机")
        except Exception as exc:
            self._show_error("刷新设备失败", exc)

    def _toggle_connection(self) -> None:
        if self.controller.device_open:
            self.disconnect_camera()
        else:
            self.connect_camera()

    def _toggle_stream(self) -> None:
        if self.capture_thread is not None and self.capture_thread.isRunning():
            if self.stop_stream():
                self.feature_panel.refresh_features()
        else:
            if self.start_stream():
                self.feature_panel.refresh_features()

    def connect_camera(self) -> None:
        selected = self.device_combo.currentData()
        if selected is None or not self.devices:
            self._show_error("连接失败", RuntimeError("没有可连接的相机"))
            return
        try:
            device = self.devices[int(selected)]
            _transport, model, serial = self.sdk.device_identity(device)
            if self.requested_serial and serial != self.requested_serial:
                raise RuntimeError(
                    f"未找到序列号为 {self.requested_serial} 的相机"
                )
            self.camera_name = f"{model}_{serial}" if serial else model
            self.controller.open_camera(device)
            self._set_connected_state(True)
            if not self.start_stream():
                self.controller.close()
                self._set_connected_state(False)
                return
            self.feature_panel.bind_controller(self.controller)
            self.statusBar().showMessage(f"已连接 {model}，正在检测棋盘格")
        except Exception as exc:
            try:
                self.disconnect_camera()
            except Exception:
                self.controller.close()
                self._set_connected_state(False)
            self._show_error("连接相机失败", exc)

    def start_stream(self) -> bool:
        """Start SDK acquisition and chessboard detection."""
        if not self.controller.device_open:
            return False
        if self.capture_thread is not None and self.capture_thread.isRunning():
            return True
        sdk_started = False
        try:
            self.controller.start_grabbing()
            sdk_started = True
            thread = CalibrationCaptureThread(self.controller, self.board)
            self._capture_generation += 1
            generation = self._capture_generation
            thread.frame_ready.connect(
                lambda packet, token=generation: self._frame_ready_guarded(
                    token, packet
                )
            )
            thread.failed.connect(
                lambda message, token=generation: self._capture_failed_guarded(
                    token, message
                )
            )
            thread.start()
            self.capture_thread = thread
            self.stream_button.setText("暂停取流")
            self.statusBar().showMessage("正在取流并检测棋盘格…")
            return True
        except Exception as exc:
            self.capture_thread = None
            self._capture_generation += 1
            if sdk_started and self.controller.grabbing:
                try:
                    self.controller.stop_grabbing()
                except Exception as stop_exc:
                    self.statusBar().showMessage(
                        f"启动失败且停止取流失败：{stop_exc}"
                    )
            self._show_error("开始取流失败", exc)
            return False

    def stop_stream(self) -> bool:
        """Stop detection first, then stop SDK acquisition."""
        thread = self.capture_thread
        previous_generation = self._capture_generation
        self._capture_generation += 1
        if thread is not None and thread.isRunning():
            thread.request_stop()
            if not thread.wait(5000):
                self._capture_generation = previous_generation
                self._show_error(
                    "停止失败", RuntimeError("角点检测线程未在 5 秒内退出")
                )
                return False
        self.capture_thread = None
        if self.controller.grabbing:
            try:
                self.controller.stop_grabbing()
            except Exception as exc:
                self.latest_packet = None
                self.capture_button.setEnabled(False)
                self.stream_button.setText("开始取流")
                self._show_error("停止取流失败", exc)
                return False
        self.latest_packet = None
        self.capture_button.setEnabled(False)
        self.stream_button.setText("开始取流")
        self.statusBar().showMessage("取流已暂停，可调整 ROI/像素格式等参数")
        return True

    def _frame_ready_guarded(self, generation: int, packet: FramePacket) -> None:
        if generation != self._capture_generation:
            return
        self._frame_ready(packet)

    def _capture_failed_guarded(self, generation: int, message: str) -> None:
        if generation != self._capture_generation:
            return
        self._capture_failed(message)

    def disconnect_camera(self) -> bool:
        if self.controller.device_open and not self.stop_stream():
            return False
        self.feature_panel.bind_controller(None)
        self.controller.close()
        self.latest_packet = None
        self.video.clear()
        self._set_connected_state(False)
        self.statusBar().showMessage("已断开")
        return True

    def _frame_ready(self, packet: FramePacket) -> None:
        self.latest_packet = packet
        self.video.set_image(packet.display_image)
        detection = packet.detection
        if not detection.found:
            self.detection_label.setText(
                f"<span style='color:#c44'>未检测到 "
                f"{self.board.inner_corners_x}×{self.board.inner_corners_y} 内角点</span>"
            )
            self.capture_button.setEnabled(False)
            return

        self.capture_button.setEnabled(True)
        self.detection_label.setText(
            "<span style='color:#198754'><b>棋盘格检测成功</b></span><br>"
            f"覆盖率 {detection.coverage * 100:.1f}% · "
            f"清晰度 {detection.sharpness:.0f} · "
            f"最小角点间距 {detection.minimum_spacing:.1f} px"
        )
        if self.auto_capture.isChecked() and len(self.samples) < self.target_samples:
            if (
                packet.received_at - self.last_auto_capture_at
                >= self.AUTO_CAPTURE_INTERVAL_S
            ):
                self._accept_packet(packet, automatic=True)

    def _hard_rejection_reason(self, packet: FramePacket) -> str | None:
        detection = packet.detection
        if detection.coverage < self.MINIMUM_COVERAGE:
            return "标定板太小，请靠近相机（覆盖率至少 2%）"
        if detection.minimum_spacing < self.MINIMUM_CORNER_SPACING:
            return "角点间距太小，请靠近相机"
        if self.samples and packet.image_size != self.samples[0].image_size:
            return (
                f"分辨率已变化：当前 {packet.image_size}，"
                f"样本为 {self.samples[0].image_size}；请清空后重采"
            )
        return None

    def _accept_packet(self, packet: FramePacket, *, automatic: bool) -> bool:
        detection = packet.detection
        if not detection.found or detection.descriptor is None:
            return False
        reason = self._hard_rejection_reason(packet)
        if reason:
            self.statusBar().showMessage(reason)
            return False
        if automatic and detection.sharpness < self.AUTO_MINIMUM_SHARPNESS:
            self.statusBar().showMessage("画面偏模糊，自动采集已跳过")
            return False
        if automatic:
            score = novelty_score(
                detection.descriptor, [sample.descriptor for sample in self.samples]
            )
            if score < self.AUTO_NOVELTY_THRESHOLD:
                self.statusBar().showMessage("姿态与已有样本相近，请移动或倾斜标定板")
                return False

        sample = make_sample(
            detection,
            packet.image_size,
            gray_image=packet.gray_image,
            frame_number=packet.frame_number,
            source_name=f"camera_frame_{packet.frame_number}",
        )
        self.samples.append(sample)
        self.last_auto_capture_at = packet.received_at
        self._update_sample_state()
        suffix = "（清晰度偏低，建议补拍）" if detection.sharpness < 50 else ""
        self.statusBar().showMessage(f"已采集第 {len(self.samples)} 张{suffix}")
        if len(self.samples) >= self.target_samples:
            self.auto_capture.setChecked(False)
            self.statusBar().showMessage("已达到推荐样本数，可以执行标定")
        return True

    def capture_current_view(self) -> None:
        packet = self.latest_packet
        if packet is None or time.monotonic() - packet.received_at > 1.0:
            self.statusBar().showMessage("没有可用的最新检测结果")
            return
        self._accept_packet(packet, automatic=False)

    @staticmethod
    def _feature_invalidates_calibration(name: str) -> bool:
        from camera_features import is_structural_feature

        broad_configuration_names = {
            "SensorMode",
            "UserSetLoad",
            "DeviceReset",
        }
        optical_prefixes = (
            "Focus",
            "Focal",
            "Zoom",
            "Iris",
        )
        return (
            is_structural_feature(name)
            or name in broad_configuration_names
            or name.startswith(optical_prefixes)
        )

    @staticmethod
    def _feature_requires_stopped_stream(name: str) -> bool:
        from camera_features import is_structural_feature

        return is_structural_feature(name) or name in {
            "DeviceReset",
            "UserSetLoad",
            "UserSetSave",
            "LUTSave",
            "ActivateShading",
            "DPCSave",
            "FPNCSave",
            "PRNUCSave",
        } or name.startswith(("Focus", "Focal", "Zoom", "Iris"))

    def _before_feature_change(self, spec) -> bool:  # type: ignore[no-untyped-def]
        """Approve a feature write and establish a safe stopped-stream state."""
        from camera_features import is_dangerous_feature

        name = spec.name
        if name in {"AcquisitionStart", "AcquisitionStop"}:
            QMessageBox.information(
                self,
                "请使用取流按钮",
                f"{name} 由标定工具统一管理。请使用窗口顶部的“开始/暂停取流”按钮，"
                "避免 SDK 状态与检测线程不一致。",
            )
            return False

        is_command = spec.effective_interface_type == "Command"
        external_io_change = (
            "DigitalIOControl" in spec.category_path
            and name
            not in {"LineSelector", "LineStatus", "LineStatusAll", "LineDebouncerTime"}
        )
        if is_command or is_dangerous_feature(name) or external_io_change:
            answer = QMessageBox.warning(
                self,
                "确认高风险相机操作",
                f"{spec.display_name} ({name}) 可能中断取流、改变多个参数，"
                "或影响外部 I/O。\n\n确定执行吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return False

        invalidates = self._feature_invalidates_calibration(name)
        if invalidates and self.samples:
            answer = QMessageBox.question(
                self,
                "成像配置将改变",
                f"修改 {spec.display_name} ({name}) 后，现有的 "
                f"{len(self.samples)} 张样本不能继续使用。\n\n"
                "是否应用参数并在设置成功后清空样本？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return False

        self._feature_change_failed = False
        self._feature_change_invalidates_samples = invalidates
        self._feature_change_stopped_stream = False
        if self._feature_requires_stopped_stream(name) and self.controller.grabbing:
            if not self.stop_stream():
                return False
            self._feature_change_stopped_stream = True
        return True

    def _feature_change_failure(
        self, spec, message: str
    ) -> None:  # type: ignore[no-untyped-def]
        self._feature_change_failed = True
        self.statusBar().showMessage(f"设置 {spec.name} 失败：{message}")

    def _after_feature_change(self, spec) -> None:  # type: ignore[no-untyped-def]
        """Restore acquisition and invalidate samples only after a successful set."""
        from camera_features import feature_change_requires_refresh

        succeeded = not self._feature_change_failed
        invalidates = self._feature_change_invalidates_samples
        restart = self._feature_change_stopped_stream
        self._feature_change_invalidates_samples = False
        self._feature_change_stopped_stream = False

        if succeeded and invalidates:
            self._discard_samples(
                f"相机参数 {spec.name} 已改变，旧标定样本和结果已清空"
            )

        if succeeded and spec.name == "DeviceReset":
            self.statusBar().showMessage("相机已复位，正在断开旧连接…")
            QTimer.singleShot(0, self.disconnect_camera)
            return

        if restart and self.controller.device_open:
            self.start_stream()
        if succeeded and (restart or feature_change_requires_refresh(spec)):
            QTimer.singleShot(0, self.feature_panel.refresh_features)

    def _discard_samples(self, status_message: str) -> None:
        self.samples.clear()
        self.results.clear()
        self.latest_packet = None
        self.last_auto_capture_at = 0.0
        self.video.clear()
        self._update_sample_state()
        self.statusBar().showMessage(status_message)

    def undo_sample(self) -> None:
        if not self.samples:
            return
        self.samples.pop()
        self._update_sample_state()
        self.statusBar().showMessage("已撤销上一张样本")

    def clear_samples(self) -> None:
        if not self.samples:
            return
        answer = QMessageBox.question(
            self,
            "清空样本",
            f"确定清空已采集的 {len(self.samples)} 张样本吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._discard_samples("样本已清空")

    def mark_lens_adjusted(self) -> None:
        if self.samples:
            answer = QMessageBox.question(
                self,
                "镜头状态已改变",
                f"物理调焦或改变光圈会使现有内参不再对应当前成像状态。\n\n"
                f"确定清空 {len(self.samples)} 张样本和旧结果吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._discard_samples("已记录镜头调整，请在当前焦点和光圈下重新采样")

    def calibrate_and_save(self) -> None:
        if len(self.samples) < self.minimum_samples:
            return
        default_path = self.default_output.expanduser().resolve()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "保存相机内参",
            str(default_path),
            "YAML calibration (*.yaml *.yml)",
        )
        if not filename:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = calibrate_samples(
                self.samples,
                self.board,
                minimum_samples=self.minimum_samples,
                reject_outliers=self.reject_outliers,
            )
            yaml_path, json_path, sample_directory = save_calibration(
                Path(filename),
                result,
                self.board,
                self.samples,
                camera_name=self.camera_name,
            )
        except Exception as exc:
            self._show_error("标定失败", exc)
            return
        finally:
            QApplication.restoreOverrideCursor()

        matrix = result.camera_matrix
        distortion = result.distortion_coefficients
        quality_code, quality_message = calibration_quality(result)
        color = {"good": "#198754", "review": "#b7791f", "poor": "#c44"}[
            quality_code
        ]
        coefficient_names = ("k1", "k2", "p1", "p2", "k3")
        coefficient_text = "\n".join(
            f"  {name} = {value:.9g}"
            for name, value in zip(coefficient_names, distortion, strict=False)
        )
        self.results.setHtml(
            f"<b style='color:{color}'>RMS = {result.rms_error:.4f} px</b><br>"
            f"{quality_message}<br><br>"
            f"fx = {matrix[0, 0]:.6f}<br>"
            f"fy = {matrix[1, 1]:.6f}<br>"
            f"cx = {matrix[0, 2]:.6f}<br>"
            f"cy = {matrix[1, 2]:.6f}<br><br>"
            f"使用 {len(result.used_indices)} 张；剔除 "
            f"{len(result.rejected_indices)} 张<br>"
            f"平均/最大单视图误差：{result.mean_view_error:.4f} / "
            f"{result.maximum_view_error:.4f} px<br><br>"
            f"YAML：{yaml_path}<br>JSON：{json_path}<br>"
            f"样本：{sample_directory or '未保存'}"
        )
        print("Camera matrix:\n", matrix)
        print("Distortion coefficients:")
        print(coefficient_text)
        print(f"RMS reprojection error: {result.rms_error:.6f} px")
        print(f"Saved YAML: {yaml_path}")
        print(f"Saved JSON: {json_path}")
        QMessageBox.information(
            self,
            "标定完成",
            f"{quality_message}\n\nRMS：{result.rms_error:.4f} px\n"
            f"内参已保存到：\n{yaml_path}",
        )

    def _capture_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"取流/检测失败：{message}")
        QMessageBox.critical(self, "取流/检测失败", message)
        QTimer.singleShot(0, self.disconnect_camera)

    def _show_error(self, title: str, error: Exception) -> None:
        self.statusBar().showMessage(str(error))
        QMessageBox.critical(self, title, str(error))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space:
            self.capture_current_view()
            event.accept()
            return
        if event.key() == Qt.Key_Backspace:
            self.undo_sample()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.disconnect_camera():
            event.accept()
        else:
            event.ignore()


def parse_grid(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX*×]\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("尺寸必须写成 列数x行数，例如 12x9")
    columns, rows = (int(item) for item in match.groups())
    if columns < 2 or rows < 2:
        raise argparse.ArgumentTypeError("行列数必须至少为 2")
    return columns, rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate HIKROBOT camera intrinsics from a live chessboard or images. "
            "The default 12x9 value means squares, hence 11x8 inner corners."
        )
    )
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument(
        "--squares",
        type=parse_grid,
        metavar="COLSxROWS",
        help="number of chessboard squares (default: 12x9)",
    )
    grid.add_argument(
        "--inner-corners",
        type=parse_grid,
        metavar="COLSxROWS",
        help="explicit OpenCV inner-corner count instead of square count",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=15.0,
        help="chessboard square edge length in millimetres (default: 15)",
    )
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=12,
        help="minimum accepted views required to calibrate (default: 12)",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=20,
        help="recommended/automatic sample target (default: 20)",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--serial", help="select a live camera by serial number")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="disable automatic diverse-view collection",
    )
    parser.add_argument(
        "--no-reject-outliers",
        action="store_true",
        help="do not reject gross per-view reprojection outliers",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        metavar="PATH",
        help="calibrate offline from image files, directories, or glob patterns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output YAML path (a JSON report is written alongside it)",
    )
    return parser


def board_from_args(args: argparse.Namespace) -> BoardSpec:
    if args.inner_corners:
        corners_x, corners_y = args.inner_corners
        return BoardSpec(corners_x + 1, corners_y + 1, args.square_size_mm)
    squares_x, squares_y = args.squares or (12, 9)
    return BoardSpec(squares_x, squares_y, args.square_size_mm)


def collect_image_paths(inputs: Sequence[str]) -> list[Path]:
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths: list[Path] = []
    for item in inputs:
        candidate = Path(item).expanduser()
        if candidate.is_dir():
            paths.extend(
                path
                for path in sorted(candidate.iterdir())
                if path.is_file() and path.suffix.lower() in supported
            )
        elif candidate.is_file():
            paths.append(candidate)
        else:
            paths.extend(
                Path(match)
                for match in sorted(glob.glob(str(candidate)))
                if Path(match).is_file()
            )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.suffix.lower() in supported and resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def run_offline(args: argparse.Namespace, board: BoardSpec) -> int:
    paths = collect_image_paths(args.images)
    if not paths:
        print("Error: no supported input images were found", file=sys.stderr)
        return 2
    samples: list[CalibrationSample] = []
    image_size: tuple[int, int] | None = None
    skipped = 0
    print(
        f"Board: {board.squares_x}x{board.squares_y} squares -> "
        f"{board.inner_corners_x}x{board.inner_corners_y} inner corners, "
        f"square={board.square_size_mm:g} mm"
    )
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Skip unreadable image: {path}", file=sys.stderr)
            skipped += 1
            continue
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(
                f"Error: image {path} has size {current_size}; expected {image_size}",
                file=sys.stderr,
            )
            return 2
        detection = detect_chessboard(gray, board)
        if not detection.found:
            print(f"No complete chessboard: {path}", file=sys.stderr)
            skipped += 1
            continue
        samples.append(
            make_sample(
                detection,
                current_size,
                gray_image=gray,
                source_name=str(path),
            )
        )
        print(
            f"Accepted {path.name}: coverage={detection.coverage * 100:.1f}% "
            f"sharpness={detection.sharpness:.0f}"
        )

    print(f"Detected {len(samples)} valid view(s); skipped {skipped}")
    if len(samples) < args.minimum_samples:
        print(
            f"Error: need at least {args.minimum_samples} valid views",
            file=sys.stderr,
        )
        return 2
    try:
        result = calibrate_samples(
            samples,
            board,
            minimum_samples=args.minimum_samples,
            reject_outliers=not args.no_reject_outliers,
        )
        output = args.output or Path("calibration_results") / (
            f"hikrobot_intrinsics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        yaml_path, json_path, sample_directory = save_calibration(
            output,
            result,
            board,
            samples,
            camera_name="hikrobot_offline",
        )
    except Exception as exc:
        print(f"Error: calibration failed: {exc}", file=sys.stderr)
        return 1

    print("Camera matrix:")
    print(result.camera_matrix)
    print("Distortion coefficients [k1, k2, p1, p2, k3]:")
    print(result.distortion_coefficients)
    print(f"RMS reprojection error: {result.rms_error:.6f} px")
    print(
        f"Used {len(result.used_indices)} view(s), rejected "
        f"{len(result.rejected_indices)}"
    )
    print(f"Saved YAML: {yaml_path}")
    print(f"Saved JSON: {json_path}")
    if sample_directory:
        print(f"Saved samples: {sample_directory}")
    return 0


def run_gui(args: argparse.Namespace, board: BoardSpec) -> int:
    import hikrobot_camera as sdk

    initialized = False
    window: CalibrationWindow | None = None
    try:
        sdk.require_ok("MV_CC_Initialize", sdk.MvCamera.MV_CC_Initialize())
        initialized = True
        app = QApplication(sys.argv)
        app.setApplicationName("HIKROBOT Intrinsic Calibration")
        output = args.output or Path("calibration_results") / (
            f"hikrobot_intrinsics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        window = CalibrationWindow(
            sdk,
            board,
            minimum_samples=args.minimum_samples,
            target_samples=args.target_samples,
            default_output=output,
            device_index=args.device_index,
            serial=args.serial,
            auto_capture=not args.manual,
            reject_outliers=not args.no_reject_outliers,
        )
        window.show()
        return app.exec_()
    except Exception as exc:
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "启动失败", str(exc))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if window is not None:
            window.disconnect_camera()
        if initialized:
            sdk.MvCamera.MV_CC_Finalize()


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.minimum_samples < 3:
        parser.error("--minimum-samples must be at least 3")
    if args.target_samples < args.minimum_samples:
        parser.error("--target-samples must be at least --minimum-samples")
    try:
        board = board_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.images:
        return run_offline(args, board)
    return run_gui(args, board)


if __name__ == "__main__":
    raise SystemExit(main())
