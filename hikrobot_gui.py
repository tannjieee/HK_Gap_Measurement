#!/usr/bin/env python3
"""PyQt5 live-view GUI for HIKROBOT cameras through the MVS SDK."""

from __future__ import annotations

import math
import sys
import threading
import time
from ctypes import byref, c_bool, c_ubyte, memset, sizeof, string_at
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QRect, QSignalBlocker, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import hikrobot_camera as sdk
from MvCameraControl_class import (  # type: ignore[import-not-found]
    MVCC_ENUMVALUE,
    MVCC_FLOATVALUE,
    MV_CC_PIXEL_CONVERT_PARAM_EX,
    MV_E_NODATA,
    PixelType_Gvsp_RGB8_Packed,
)


AUTO_VALUE_TO_TEXT = {
    "ExposureAuto": {0: "Off", 1: "Once", 2: "Continuous"},
    "GainAuto": {0: "Off", 1: "Once", 2: "Continuous"},
    "BalanceWhiteAuto": {0: "Off", 1: "Continuous", 2: "Once"},
}


class FloatControl(QWidget):
    """A synchronized slider and spin box for a floating-point feature."""

    value_edited = pyqtSignal(float)

    def __init__(self, suffix: str, logarithmic: bool = False) -> None:
        super().__init__()
        self.minimum = 0.0
        self.maximum = 1.0
        self.logarithmic = logarithmic
        self._updating = False

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(2)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(130)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def configure(self, minimum: float, maximum: float, current: float) -> None:
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("Feature range must be finite")
        if maximum <= minimum:
            maximum = minimum + 1.0
        if self.logarithmic and minimum <= 0:
            minimum = max(maximum / 1_000_000.0, 0.001)

        self.minimum = minimum
        self.maximum = maximum
        step = max((maximum - minimum) / 1000.0, 0.01)
        with QSignalBlocker(self.spin):
            self.spin.setRange(minimum, maximum)
            self.spin.setSingleStep(step)
        self.set_value(current)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, value: float) -> None:
        value = min(max(value, self.minimum), self.maximum)
        self._updating = True
        try:
            with QSignalBlocker(self.spin):
                self.spin.setValue(value)
            with QSignalBlocker(self.slider):
                self.slider.setValue(self._value_to_slider(value))
        finally:
            self._updating = False

    def _slider_to_value(self, slider_value: int) -> float:
        ratio = slider_value / 1000.0
        if self.logarithmic:
            return self.minimum * (self.maximum / self.minimum) ** ratio
        return self.minimum + (self.maximum - self.minimum) * ratio

    def _value_to_slider(self, value: float) -> int:
        if self.logarithmic:
            ratio = math.log(value / self.minimum) / math.log(
                self.maximum / self.minimum
            )
        else:
            ratio = (value - self.minimum) / (self.maximum - self.minimum)
        return round(min(max(ratio, 0.0), 1.0) * 1000)

    def _slider_changed(self, slider_value: int) -> None:
        if self._updating:
            return
        value = self._slider_to_value(slider_value)
        with QSignalBlocker(self.spin):
            self.spin.setValue(value)
        self.value_edited.emit(value)

    def _spin_changed(self, value: float) -> None:
        if self._updating:
            return
        with QSignalBlocker(self.slider):
            self.slider.setValue(self._value_to_slider(value))
        self.value_edited.emit(value)


class VideoWidget(QWidget):
    """Aspect-ratio-preserving image display widget."""

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
            painter.drawText(self.rect(), Qt.AlignCenter, "等待视频流")
            return

        scaled_size = self.image.size()
        scaled_size.scale(self.size(), Qt.KeepAspectRatio)
        target_x = (self.width() - scaled_size.width()) // 2
        target_y = (self.height() - scaled_size.height()) // 2
        target = QRect(
            target_x,
            target_y,
            scaled_size.width(),
            scaled_size.height(),
        )
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(target, self.image)


class CameraController:
    """Thread-safe ownership and feature access for one MVS camera handle."""

    def __init__(self) -> None:
        self.cam = sdk.MvCamera()
        self.lock = threading.RLock()
        self.handle_created = False
        self.device_open = False
        self.grabbing = False
        self.rgb_buffer = None
        self.rgb_buffer_size = 0

    def open_camera(self, device) -> None:  # type: ignore[no-untyped-def]
        with self.lock:
            if self.device_open:
                return
            try:
                sdk.require_ok("MV_CC_CreateHandle", self.cam.MV_CC_CreateHandle(device))
                self.handle_created = True
                sdk.require_ok(
                    "MV_CC_OpenDevice",
                    self.cam.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0),
                )
                self.device_open = True
                sdk.require_ok(
                    "Set TriggerMode=Off",
                    self.cam.MV_CC_SetEnumValue(
                        "TriggerMode", sdk.MV_TRIGGER_MODE_OFF
                    ),
                )
                sdk.require_ok(
                    "MV_CC_SetBayerCvtQuality", self.cam.MV_CC_SetBayerCvtQuality(1)
                )
            except Exception:
                self.close()
                raise

    def start_grabbing(self) -> None:
        with self.lock:
            if not self.device_open:
                raise sdk.CameraError("Camera is not connected")
            if self.grabbing:
                return
            sdk.require_ok("MV_CC_StartGrabbing", self.cam.MV_CC_StartGrabbing())
            self.grabbing = True

    def stop_grabbing(self) -> None:
        with self.lock:
            if not self.grabbing:
                return
            result = self.cam.MV_CC_StopGrabbing()
            self.grabbing = False
            sdk.require_ok("MV_CC_StopGrabbing", result)

    def close(self) -> None:
        with self.lock:
            if self.grabbing:
                self.cam.MV_CC_StopGrabbing()
                self.grabbing = False
            if self.device_open:
                self.cam.MV_CC_CloseDevice()
                self.device_open = False
            if self.handle_created:
                self.cam.MV_CC_DestroyHandle()
                self.handle_created = False
            self.rgb_buffer = None
            self.rgb_buffer_size = 0

    def get_float(self, key: str) -> tuple[float, float, float]:
        value = MVCC_FLOATVALUE()
        with self.lock:
            sdk.require_ok(key, self.cam.MV_CC_GetFloatValue(key, value))
        return value.fCurValue, value.fMin, value.fMax

    def set_float(self, key: str, value: float) -> None:
        with self.lock:
            sdk.require_ok(key, self.cam.MV_CC_SetFloatValue(key, value))

    def get_enum(self, key: str) -> int:
        value = MVCC_ENUMVALUE()
        with self.lock:
            sdk.require_ok(key, self.cam.MV_CC_GetEnumValue(key, value))
        return value.nCurValue

    def set_enum_text(self, key: str, value: str) -> None:
        with self.lock:
            sdk.require_ok(key, self.cam.MV_CC_SetEnumValueByString(key, value))

    def get_bool(self, key: str) -> bool:
        value = c_bool()
        with self.lock:
            sdk.require_ok(key, self.cam.MV_CC_GetBoolValue(key, value))
        return value.value

    def set_bool(self, key: str, value: bool) -> None:
        with self.lock:
            sdk.require_ok(key, self.cam.MV_CC_SetBoolValue(key, value))

    def read_rgb_frame(self) -> tuple[bytes, int, int, int] | None:
        with self.lock:
            if not self.grabbing:
                return None

            frame = sdk.MV_FRAME_OUT()
            memset(byref(frame), 0, sizeof(frame))
            result = self.cam.MV_CC_GetImageBuffer(frame, 200)
            if result == MV_E_NODATA:
                return None
            sdk.require_ok("MV_CC_GetImageBuffer", result)
            if not frame.pBufAddr:
                raise sdk.CameraError("The SDK returned an empty image buffer")

            try:
                info = frame.stFrameInfo
                width = int(info.nWidth)
                height = int(info.nHeight)
                required_size = width * height * 3
                if required_size != self.rgb_buffer_size:
                    self.rgb_buffer = (c_ubyte * required_size)()
                    self.rgb_buffer_size = required_size

                conversion = MV_CC_PIXEL_CONVERT_PARAM_EX()
                memset(byref(conversion), 0, sizeof(conversion))
                conversion.nWidth = width
                conversion.nHeight = height
                conversion.enSrcPixelType = info.enPixelType
                conversion.pSrcData = frame.pBufAddr
                conversion.nSrcDataLen = info.nFrameLenEx or info.nFrameLen
                conversion.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                conversion.pDstBuffer = self.rgb_buffer
                conversion.nDstBufferSize = required_size
                sdk.require_ok(
                    "MV_CC_ConvertPixelTypeEx",
                    self.cam.MV_CC_ConvertPixelTypeEx(conversion),
                )
                if conversion.nDstLen < required_size:
                    raise sdk.CameraError(
                        f"RGB conversion returned {conversion.nDstLen} bytes; "
                        f"expected {required_size}"
                    )
                rgb = string_at(self.rgb_buffer, required_size)
                return rgb, width, height, int(info.nFrameNum)
            finally:
                self.cam.MV_CC_FreeImageBuffer(frame)


class CaptureThread(QThread):
    frame_ready = pyqtSignal(object, int)
    statistics = pyqtSignal(float, int, int)
    failed = pyqtSignal(str)

    def __init__(self, controller: CameraController) -> None:
        super().__init__()
        self.controller = controller
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        count = 0
        sample_started = time.monotonic()
        try:
            while not self.stop_event.is_set():
                frame = self.controller.read_rgb_frame()
                if frame is None:
                    continue
                rgb, width, height, frame_number = frame
                image = QImage(
                    rgb,
                    width,
                    height,
                    width * 3,
                    QImage.Format_RGB888,
                ).copy()
                self.frame_ready.emit(image, frame_number)

                count += 1
                now = time.monotonic()
                elapsed = now - sample_started
                if elapsed >= 1.0:
                    self.statistics.emit(count / elapsed, width, height)
                    count = 0
                    sample_started = now
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HIKROBOT 相机实时控制")
        self.resize(1280, 800)

        self.controller = CameraController()
        self.device_list = None
        self.devices = []
        self.capture_thread: CaptureThread | None = None
        self.latest_image = QImage()
        self.loading_parameters = False

        self.video = VideoWidget()
        self.device_combo = QComboBox()
        self.refresh_button = QPushButton("刷新设备")
        self.connect_button = QPushButton("连接")
        self.stream_button = QPushButton("开始取流")
        self.read_button = QPushButton("读取参数")
        self.save_button = QPushButton("保存当前帧")
        self.save_button.setEnabled(False)

        self.exposure_auto = self._auto_combo()
        self.exposure = FloatControl(" µs", logarithmic=True)
        self.gain_auto = self._auto_combo()
        self.gain = FloatControl(" dB")
        self.white_balance_auto = self._auto_combo()
        self.frame_rate_enable = QCheckBox("启用帧率限制")
        self.frame_rate = FloatControl(" fps")

        self.exposure_timer = self._feature_timer(self._apply_exposure)
        self.gain_timer = self._feature_timer(self._apply_gain)
        self.frame_rate_timer = self._feature_timer(self._apply_frame_rate)
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(1000)
        self.poll_timer.timeout.connect(self._poll_auto_parameters)

        self._build_ui()
        self._connect_signals()
        self._set_connected_state(False)
        self.refresh_devices()
        if self.devices:
            QTimer.singleShot(0, self.connect_camera)

    @staticmethod
    def _auto_combo() -> QComboBox:
        combo = QComboBox()
        combo.addItems(["Off", "Once", "Continuous"])
        return combo

    def _feature_timer(self, callback) -> QTimer:  # type: ignore[no-untyped-def]
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(80)
        timer.timeout.connect(callback)
        return timer

    def _build_ui(self) -> None:
        connection_box = QGroupBox("相机")
        connection_layout = QVBoxLayout(connection_box)
        connection_layout.addWidget(self.device_combo)
        connection_buttons = QHBoxLayout()
        connection_buttons.addWidget(self.refresh_button)
        connection_buttons.addWidget(self.connect_button)
        connection_layout.addLayout(connection_buttons)
        connection_layout.addWidget(self.stream_button)

        parameter_box = QGroupBox("采集参数（取流中可动态调整）")
        parameter_layout = QFormLayout(parameter_box)
        parameter_layout.addRow("自动曝光", self.exposure_auto)
        parameter_layout.addRow("曝光时间", self.exposure)
        parameter_layout.addRow("自动增益", self.gain_auto)
        parameter_layout.addRow("增益", self.gain)
        parameter_layout.addRow("自动白平衡", self.white_balance_auto)
        parameter_layout.addRow(self.frame_rate_enable)
        parameter_layout.addRow("采集帧率", self.frame_rate)
        parameter_layout.addRow(self.read_button)

        side_layout = QVBoxLayout()
        side_layout.addWidget(connection_box)
        side_layout.addWidget(parameter_box)
        side_layout.addWidget(self.save_button)
        side_layout.addStretch(1)

        side_panel = QWidget()
        side_panel.setLayout(side_layout)
        side_panel.setMaximumWidth(390)
        side_panel.setMinimumWidth(350)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video, 1)
        main_layout.addWidget(side_panel)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("未连接")

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.connect_button.clicked.connect(self._toggle_connection)
        self.stream_button.clicked.connect(self._toggle_stream)
        self.read_button.clicked.connect(self.refresh_parameters)
        self.save_button.clicked.connect(self.save_current_frame)

        self.exposure_auto.currentTextChanged.connect(
            lambda text: self._set_auto("ExposureAuto", text)
        )
        self.gain_auto.currentTextChanged.connect(
            lambda text: self._set_auto("GainAuto", text)
        )
        self.white_balance_auto.currentTextChanged.connect(
            lambda text: self._set_auto("BalanceWhiteAuto", text)
        )
        self.exposure.value_edited.connect(lambda _: self.exposure_timer.start())
        self.gain.value_edited.connect(lambda _: self.gain_timer.start())
        self.frame_rate.value_edited.connect(lambda _: self.frame_rate_timer.start())
        self.frame_rate_enable.toggled.connect(self._set_frame_rate_enable)

    def _set_connected_state(self, connected: bool) -> None:
        self.connect_button.setText("断开" if connected else "连接")
        self.device_combo.setEnabled(not connected)
        self.refresh_button.setEnabled(not connected)
        self.stream_button.setEnabled(connected)
        self.read_button.setEnabled(connected)
        for widget in (
            self.exposure_auto,
            self.gain_auto,
            self.white_balance_auto,
            self.frame_rate_enable,
        ):
            widget.setEnabled(connected)
        if not connected:
            self.stream_button.setText("开始取流")
            self.exposure.setEnabled(False)
            self.gain.setEnabled(False)
            self.frame_rate.setEnabled(False)

    def refresh_devices(self) -> None:
        if self.controller.device_open:
            return
        try:
            self.device_list, self.devices = sdk.enumerate_devices()
            self.device_combo.clear()
            for index, device in enumerate(self.devices):
                transport, model, serial = sdk.device_identity(device)
                self.device_combo.addItem(
                    f"[{index}] {model} ({serial}) - {transport}", index
                )
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

    def connect_camera(self) -> None:
        index = self.device_combo.currentData()
        if index is None or not self.devices:
            self._show_error("连接失败", sdk.CameraError("没有可连接的相机"))
            return
        try:
            self.controller.open_camera(self.devices[int(index)])
            self._set_connected_state(True)
            self.refresh_parameters()
            self.start_stream()
            self.poll_timer.start()
        except Exception as exc:
            self.controller.close()
            self._set_connected_state(False)
            self._show_error("连接相机失败", exc)

    def disconnect_camera(self) -> None:
        self.poll_timer.stop()
        self.stop_stream()
        self.controller.close()
        self.video.clear()
        self.latest_image = QImage()
        self.save_button.setEnabled(False)
        self._set_connected_state(False)
        self.statusBar().showMessage("已断开")

    def _toggle_stream(self) -> None:
        if self.capture_thread and self.capture_thread.isRunning():
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self) -> None:
        if not self.controller.device_open:
            return
        if self.capture_thread and self.capture_thread.isRunning():
            return
        try:
            self.controller.start_grabbing()
            self.capture_thread = CaptureThread(self.controller)
            self.capture_thread.frame_ready.connect(self._display_frame)
            self.capture_thread.statistics.connect(self._display_statistics)
            self.capture_thread.failed.connect(self._capture_failed)
            self.capture_thread.start()
            self.stream_button.setText("停止取流")
            self.statusBar().showMessage("正在取流…")
        except Exception as exc:
            self._show_error("开始取流失败", exc)

    def stop_stream(self) -> None:
        thread = self.capture_thread
        if thread and thread.isRunning():
            thread.request_stop()
            if not thread.wait(1500):
                self.statusBar().showMessage("等待取流线程退出…")
                thread.wait(2000)
        self.capture_thread = None
        if self.controller.grabbing:
            try:
                self.controller.stop_grabbing()
            except Exception as exc:
                self._show_error("停止取流失败", exc)
        self.stream_button.setText("开始取流")

    def refresh_parameters(self) -> None:
        if not self.controller.device_open:
            return
        self.loading_parameters = True
        try:
            exposure, exposure_min, exposure_max = self.controller.get_float(
                "ExposureTime"
            )
            gain, gain_min, gain_max = self.controller.get_float("Gain")
            frame_rate, frame_rate_min, frame_rate_max = self.controller.get_float(
                "AcquisitionFrameRate"
            )
            self.exposure.configure(exposure_min, exposure_max, exposure)
            self.gain.configure(gain_min, gain_max, gain)
            self.frame_rate.configure(
                frame_rate_min, min(frame_rate_max, 500.0), frame_rate
            )
            self._set_combo_from_camera("ExposureAuto", self.exposure_auto)
            self._set_combo_from_camera("GainAuto", self.gain_auto)
            self._set_combo_from_camera(
                "BalanceWhiteAuto", self.white_balance_auto
            )
            enabled = self.controller.get_bool("AcquisitionFrameRateEnable")
            with QSignalBlocker(self.frame_rate_enable):
                self.frame_rate_enable.setChecked(enabled)
            self._update_manual_control_states()
            self.statusBar().showMessage("参数已读取")
        except Exception as exc:
            self._show_error("读取参数失败", exc)
        finally:
            self.loading_parameters = False

    def _set_combo_from_camera(self, key: str, combo: QComboBox) -> None:
        current = self.controller.get_enum(key)
        text = AUTO_VALUE_TO_TEXT[key].get(current, "Off")
        with QSignalBlocker(combo):
            combo.setCurrentText(text)

    def _set_auto(self, key: str, value: str) -> None:
        if self.loading_parameters or not self.controller.device_open:
            return
        try:
            self.controller.set_enum_text(key, value)
            self._update_manual_control_states()
            self.statusBar().showMessage(f"{key} = {value}")
        except Exception as exc:
            self._show_error("设置自动参数失败", exc)

    def _update_manual_control_states(self) -> None:
        connected = self.controller.device_open
        self.exposure.setEnabled(connected and self.exposure_auto.currentText() == "Off")
        self.gain.setEnabled(connected and self.gain_auto.currentText() == "Off")
        self.frame_rate.setEnabled(connected and self.frame_rate_enable.isChecked())

    def _apply_exposure(self) -> None:
        self._apply_float("ExposureTime", self.exposure.value())

    def _apply_gain(self) -> None:
        self._apply_float("Gain", self.gain.value())

    def _apply_frame_rate(self) -> None:
        self._apply_float("AcquisitionFrameRate", self.frame_rate.value())

    def _apply_float(self, key: str, value: float) -> None:
        if self.loading_parameters or not self.controller.device_open:
            return
        try:
            self.controller.set_float(key, value)
            self.statusBar().showMessage(f"{key} = {value:.2f}")
        except Exception as exc:
            self._show_error(f"设置 {key} 失败", exc)

    def _set_frame_rate_enable(self, enabled: bool) -> None:
        if self.loading_parameters or not self.controller.device_open:
            return
        try:
            self.controller.set_bool("AcquisitionFrameRateEnable", enabled)
            self._update_manual_control_states()
            if enabled:
                self.frame_rate_timer.start()
        except Exception as exc:
            self._show_error("设置帧率控制失败", exc)

    def _poll_auto_parameters(self) -> None:
        if not self.controller.device_open:
            return
        self.loading_parameters = True
        try:
            self._set_combo_from_camera("ExposureAuto", self.exposure_auto)
            self._set_combo_from_camera("GainAuto", self.gain_auto)
            if self.exposure_auto.currentText() != "Off":
                current, _, _ = self.controller.get_float("ExposureTime")
                self.exposure.set_value(current)
            if self.gain_auto.currentText() != "Off":
                current, _, _ = self.controller.get_float("Gain")
                self.gain.set_value(current)
            self._update_manual_control_states()
        except Exception as exc:
            self.statusBar().showMessage(f"自动参数轮询失败：{exc}")
        finally:
            self.loading_parameters = False

    def _display_frame(self, image: QImage, frame_number: int) -> None:
        self.latest_image = image
        self.video.set_image(image)
        self.save_button.setEnabled(True)
        self.save_button.setToolTip(f"当前相机帧号：{frame_number}")

    def _display_statistics(self, fps: float, width: int, height: int) -> None:
        self.statusBar().showMessage(f"取流正常 | {width}×{height} | 显示 {fps:.1f} fps")

    def _capture_failed(self, message: str) -> None:
        self.stop_stream()
        self._show_error("视频流错误", sdk.CameraError(message))

    def save_current_frame(self) -> None:
        if self.latest_image.isNull():
            return
        capture_dir = Path(__file__).resolve().parent / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        default_name = capture_dir / (
            f"gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "保存当前帧",
            str(default_name),
            "PNG image (*.png)",
        )
        if not filename:
            return
        if not self.latest_image.save(filename, "PNG"):
            self._show_error("保存失败", sdk.CameraError(f"无法写入 {filename}"))
            return
        self.statusBar().showMessage(f"已保存：{filename}")

    def _show_error(self, title: str, error: Exception) -> None:
        self.statusBar().showMessage(str(error))
        QMessageBox.critical(self, title, str(error))

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.disconnect_camera()
        event.accept()


def main() -> int:
    initialized = False
    window = None
    try:
        sdk.require_ok("MV_CC_Initialize", sdk.MvCamera.MV_CC_Initialize())
        initialized = True
        app = QApplication(sys.argv)
        app.setApplicationName("HIKROBOT Camera GUI")
        window = MainWindow()
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
            window.controller.close()
        if initialized:
            sdk.MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    raise SystemExit(main())
