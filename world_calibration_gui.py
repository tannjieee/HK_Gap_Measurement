#!/usr/bin/env python3
"""HIKROBOT world calibration using an 80 mm tag36h11 ID 0.

Run: python world_calibration_gui.py
The reference origin is (126.1, -12.5, 142) mm in world coordinates. The
application's remapped tag axes are assumed aligned with the world axes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Sequence

# Import the existing GUI first to preserve its OpenCV / Qt environment setup.
from apriltag_pose_gui import (
    MainWindow, PoseFrame, _format_xyz, _format_vector,
    _rotation_to_euler_xyz, main as run_pose_gui,
)
import numpy as np
import yaml
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from world_calibration_core import (
    WorldCalibration, calibrate_world, reference_world_pose, rigid_transform,
)


class WorldCalibrationWindow(MainWindow):
    def __init__(self, config_path: str | Path, **kwargs: Any) -> None:
        self.calibration: WorldCalibration | None = None
        self._samples: list[np.ndarray] | None = None
        self._sample_errors: list[float] = []
        self._sample_context: dict[str, Any] = {}
        self._seen_frames: set[int] = set()
        self._world_ready = False
        self._world_last_refresh = 0.0
        self._deadline = 0.0
        super().__init__(config_path, **kwargs)
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        settings = raw.get("world_calibration", {})
        self.output_dir = (
            self.config_path.parent / settings.get("output_dir", "../calibration_results/world")
        ).resolve()
        self.max_translation_rms_mm = float(settings.get("max_translation_rms_mm", 2.0))
        self.max_rotation_rms_deg = float(settings.get("max_rotation_rms_deg", 1.0))
        self.timeout_s = float(settings.get("timeout_s", 20.0))
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("采样超时必须为有限正数")
        self.setWindowTitle("HIKROBOT 世界坐标系标定 — ID 0 / 80 mm")
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "参考：tag36h11 / ID 0 / 边长 80 mm\n"
            "新 +X = 旧 +Z；新 +Y = 旧 −X；新 +Z = 旧 −Y\n"
            "三个新正轴与世界系同向时，下面的 R/P/Y 均为 0°。\n"
            "采样时固定相机和标签；相机移动后需重新标定。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.world_xyz = self._vector_fields(
            form, "ID 0 世界原点", "XYZ", settings.get("reference_xyz_mm", [126.1, -12.5, 142]), " mm"
        )
        self.world_rpy = self._vector_fields(
            form, "ID 0 世界旋转", "RPY", settings.get("reference_rpy_deg", [0, 0, 0]), "°"
        )
        self.sample_count = QSpinBox()
        self.sample_count.setRange(3, 1000)
        self.sample_count.setValue(int(settings.get("sample_count", 60)))
        form.addRow("有效采样帧数", self.sample_count)
        self.max_error = QDoubleSpinBox()
        self.max_error.setRange(0.01, 20.0)
        self.max_error.setValue(float(settings.get("max_reprojection_error_px", 1.5)))
        self.max_error.setSuffix(" px")
        form.addRow("最大重投影误差", self.max_error)
        layout.addLayout(form)
        self.world_start = QPushButton("开始标定并保存")
        self.world_cancel = QPushButton("取消采样")
        self.world_load = QPushButton("加载标定")
        self.world_save = QPushButton("另存标定")
        row = QHBoxLayout()
        for button in (self.world_start, self.world_cancel, self.world_load, self.world_save):
            row.addWidget(button)
        layout.addLayout(row)
        self.world_progress = QProgressBar()
        self.world_progress.setRange(0, self.sample_count.value())
        self.world_progress.setValue(0)
        layout.addWidget(self.world_progress)
        self.world_status = QLabel("尚未标定。连接相机并取流后，将 ID 0 放在已知位置，再开始标定。")
        self.world_status.setWordWrap(True)
        layout.addWidget(self.world_status)
        self.world_result = QPlainTextEdit()
        self.world_result.setReadOnly(True)
        self.world_result.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.world_result, 1)
        self.world_live = QPlainTextEdit()
        self.world_live.setReadOnly(True)
        self.world_live.setStyleSheet("font-family: monospace;")
        layout.addWidget(QLabel("实时标签位姿（世界坐标系，mm / deg）"))
        layout.addWidget(self.world_live, 1)
        self.right_tabs.addTab(page, "世界标定")
        self.right_tabs.setCurrentWidget(page)
        self.world_start.clicked.connect(self.begin_calibration)
        self.world_cancel.clicked.connect(lambda: self._invalidate("已取消采样"))
        self.world_load.clicked.connect(self.load_world_calibration)
        self.world_save.clicked.connect(self.save_world_calibration)
        for field in (*self.world_xyz, *self.world_rpy):
            field.valueChanged.connect(lambda: self._invalidate("世界参考位姿已修改，请重新标定或加载匹配结果"))
        self._world_ready = True
        self._set_world_controls()
        self.world_timer = QTimer(self)
        self.world_timer.setInterval(200)
        self.world_timer.timeout.connect(self._check_timeout)
        self.world_timer.start()

    @staticmethod
    def _vector_fields(form, title, names, values, suffix):
        values = np.asarray(values, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"{title} 必须包含三个有限数值")
        row = QHBoxLayout()
        fields = []
        for name, value in zip(names, values):
            field = QDoubleSpinBox()
            field.setRange(-1000000, 1000000)
            field.setDecimals(3)
            field.setValue(float(value))
            field.setSuffix(suffix)
            row.addWidget(QLabel(name))
            row.addWidget(field)
            fields.append(field)
        form.addRow(title, row)
        return fields

    def _known_pose(self) -> np.ndarray:
        return reference_world_pose(
            [field.value() for field in self.world_xyz],
            [field.value() for field in self.world_rpy],
        )

    def _context(self) -> dict[str, Any]:
        serial = self.serial_hint or ""
        index = self.device_combo.currentData()
        if self.sdk is not None and index is not None and self.devices:
            serial = self.sdk.device_identity(self.devices[int(index)])[2]
        return {
            "reference_tag_id": 0,
            "tag_family": self.pose_config.family,
            "tag_size_mm": self.pose_config.tag_size_mm,
            "camera_serial": serial,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion": self.distortion.tolist(),
            "intrinsics_image_size": list(self.source_size) if self.source_size else None,
            "backend": self.estimator.backend_name,
        }

    def _set_world_controls(self) -> None:
        if not self._world_ready:
            return
        active = self._samples is not None
        self.world_start.setEnabled(not active)
        self.world_cancel.setEnabled(active)
        self.world_load.setEnabled(not active)
        self.world_save.setEnabled(self.calibration is not None and not active)
        for field in (*self.world_xyz, *self.world_rpy, self.sample_count, self.max_error):
            field.setEnabled(not active)

    def _invalidate(self, reason: str) -> None:
        self._samples = None
        self.calibration = None
        if self._world_ready:
            self.world_status.setText(reason)
            self.world_result.clear()
            self.world_live.clear()
            self.world_progress.setValue(0)
            self._set_world_controls()

    def begin_calibration(self) -> None:
        if not self.capture_thread or not self.capture_thread.isRunning():
            self.world_status.setText("请先在 AprilTag 页连接相机并开始取流")
            return
        config = self.pose_config
        if config.family != "tag36h11" or not np.isclose(config.tag_size_mm, 80.0):
            self.world_status.setText("标定需要 tag36h11、边长 80 mm，请先修改检测参数")
            return
        if config.tag_ids and 0 not in config.tag_ids:
            self.world_status.setText("检测配置的 tag.ids 必须包含 ID 0")
            return
        self._invalidate("正在采样 ID 0，请保持相机和标签静止")
        self._samples = []
        self._sample_errors = []
        self._seen_frames = set()
        self._sample_context = self._context()
        self._deadline = time.monotonic() + self.timeout_s
        self.world_progress.setRange(0, self.sample_count.value())
        self._set_world_controls()

    def _check_timeout(self) -> None:
        if self._samples is not None and time.monotonic() > self._deadline:
            count = len(self._samples)
            self._invalidate(f"采样超时，仅获得 {count} 帧有效 ID 0；请检查可见性、清晰度和误差阈值后重试")

    def _display_frame(self, packet: PoseFrame) -> None:
        super()._display_frame(packet)
        if not self._world_ready:
            return
        context = self._context()
        if self.calibration is not None:
            metadata = self.calibration.metadata
            if (context != metadata.get("capture_context")
                    or list(packet.image_size) != metadata.get("image_size")):
                self._invalidate("相机、检测后端或成像参数已改变，请重新标定")
        if self._samples is not None:
            self._check_timeout()
            if self._samples is not None:
                self._sample_frame(packet, context)
        now = time.monotonic()
        if now - self._world_last_refresh >= 0.1:
            self._world_last_refresh = now
            self._update_world_live(packet)

    def _sample_frame(self, packet: PoseFrame, context: dict[str, Any]) -> None:
        if context != self._sample_context:
            self._invalidate("采样期间相机或检测参数发生改变，请重新开始")
            return
        if packet.frame_number in self._seen_frames:
            return
        self._seen_frames.add(packet.frame_number)
        references = [pose for pose in packet.poses if pose.tag_id == 0 and pose.family == "tag36h11"]
        if len(references) != 1:
            self.world_status.setText("等待唯一的 ID 0；请保持标签完整可见")
            return
        pose = references[0]
        error = pose.reprojection_error_px
        if not np.isfinite(error) or error > self.max_error.value():
            self.world_status.setText(f"跳过 ID 0：重投影误差 {error:.2f} px 超过阈值")
            return
        try:
            sample = rigid_transform(pose.rotation_matrix, pose.tvec)
            if sample[2, 3] <= 0:
                raise ValueError("标签深度必须为正")
        except ValueError as exc:
            self.world_status.setText(f"跳过无效位姿：{exc}")
            return
        if self._samples and list(packet.image_size) != self._sample_image_size:
            self._invalidate("采样期间图像尺寸发生改变，请重新开始")
            return
        self._sample_image_size = list(packet.image_size)
        self._samples.append(sample)
        self._sample_errors.append(float(error))
        self.world_progress.setValue(len(self._samples))
        self.world_status.setText(f"已采集 {len(self._samples)}/{self.sample_count.value()} 帧；请保持静止")
        if len(self._samples) >= self.sample_count.value():
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        try:
            self.calibration = calibrate_world(
                self._samples, self._known_pose(),
                max_translation_rms_mm=self.max_translation_rms_mm,
                max_rotation_rms_deg=self.max_rotation_rms_deg,
                metadata={
                    "capture_context": self._sample_context,
                    "image_size": self._sample_image_size,
                    "mean_reprojection_error_px": float(np.mean(self._sample_errors)),
                },
            )
        except ValueError as exc:
            self._invalidate(str(exc))
            return
        self._samples = None
        self._show_world_result()
        self._set_world_controls()
        path = self.output_dir / f"world_calibration_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        try:
            self.calibration.save(path)
            self.world_status.setText(f"标定完成，已保存：{path}")
        except (OSError, ValueError) as exc:
            self.world_status.setText(f"标定已算出，但保存失败：{exc}；可点击另存标定")

    def _show_world_result(self) -> None:
        result = self.calibration
        if result is None:
            return
        self.world_result.setPlainText(
            f"有效帧：{result.sample_count}\n"
            f"位置 RMS：{result.translation_rms_mm:.4f} mm\n"
            f"角度 RMS：{result.rotation_rms_deg:.4f} deg\n"
            "RMS 表示采样重复性，不代表绝对精度。\n"
            "相机原点（世界系）：\n" + _format_xyz(result.world_from_camera[:3, 3] * 1000)
            + "\nT_world_camera（平移单位 m）：\n"
            + np.array2string(result.world_from_camera, precision=7, suppress_small=True)
        )

    def _update_world_live(self, packet: PoseFrame) -> None:
        if self.calibration is None:
            self.world_live.setPlainText("完成或加载标定后显示世界坐标")
            return
        lines = []
        for pose in packet.poses:
            try:
                world_pose = self.calibration.world_pose(rigid_transform(pose.rotation_matrix, pose.tvec))
            except ValueError:
                continue
            lines.extend([
                f"ID {pose.tag_id}  world:",
                _format_xyz(world_pose[:3, 3] * 1000),
                "RPY=" + _format_vector(_rotation_to_euler_xyz(world_pose[:3, :3]), " deg"),
                "",
            ])
        self.world_live.setPlainText("\n".join(lines) if lines else "当前未检测到标签；相机到世界的标定变换保持固定")

    def save_world_calibration(self) -> None:
        if self.calibration is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存世界标定", str(self.output_dir / "world_calibration.json"), "JSON (*.json)"
        )
        if filename:
            try:
                self.calibration.save(filename)
                self.world_status.setText(f"已保存：{filename}")
            except (OSError, ValueError) as exc:
                self._show_error("保存世界标定失败", exc)

    def load_world_calibration(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "加载世界标定", str(self.output_dir), "JSON (*.json)")
        if not filename:
            return
        try:
            result = WorldCalibration.load(filename)
            if not np.allclose(result.world_from_reference, self._known_pose(), atol=1e-8, rtol=0):
                raise ValueError("文件中的 ID 0 世界位姿与当前输入不匹配")
            if result.metadata.get("capture_context") != self._context():
                raise ValueError("文件中的相机、内参、标签尺寸或检测后端与当前设置不匹配")
            if self.latest_frame is not None and result.metadata.get("image_size") != list(self.latest_frame.image_size):
                raise ValueError("文件中的图像尺寸与当前视频不匹配")
            self.calibration = result
            self._show_world_result()
            self._set_world_controls()
            self.world_status.setText(f"已加载：{filename}；请确保相机仍在标定时的固定位置")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._show_error("加载世界标定失败", exc)

    def stop_stream(self) -> None:
        if self._samples is not None:
            self._invalidate("取流已停止，采样取消")
        super().stop_stream()

    def disconnect_camera(self) -> None:
        self._invalidate("相机已断开，请连接后重新标定或加载匹配结果")
        super().disconnect_camera()

    def _feature_changed(self, spec: Any) -> None:
        super()._feature_changed(spec)
        from camera_features import is_structural_feature
        if is_structural_feature(str(getattr(spec, "name", ""))):
            self._invalidate("成像几何已改变，请加载匹配内参并重新标定")

    def _settings_changed(self, *args: Any) -> None:
        previous = self._context() if self._world_ready else None
        super()._settings_changed(*args)
        if previous is not None and previous != self._context():
            self._invalidate("检测参数已改变，请重新标定或加载匹配结果")

    def reload_config(self) -> None:
        super().reload_config()
        if self._world_ready:
            self._invalidate("配置已重新加载，请重新标定或加载匹配结果")


def main(argv: Sequence[str] | None = None) -> int:
    return run_pose_gui(
        argv, window_class=WorldCalibrationWindow,
        default_config_path=Path(__file__).parent / "config" / "world_calibration.yaml",
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
