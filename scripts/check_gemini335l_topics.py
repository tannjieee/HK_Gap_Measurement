#!/usr/bin/env python3
"""Receive and validate live Gemini 335L ROS topics, optionally starting the driver."""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-driver", action="store_true")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--camera-name", default="gemini335l")
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.seconds < 3:
        parser.error("--seconds must be at least 3")
    if args.domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)

    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from tf2_msgs.msg import TFMessage

    project = Path(__file__).resolve().parents[1]
    output = args.output or project / "logs" / "gemini335l" / f"check_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "/" + args.camera_name.strip("/")
    metrics: dict[str, dict] = {}
    frames = set()
    tf_edges = set()
    driver = None
    driver_log = None
    node = None
    initialized = False

    def record(topic, message, kind):
        now = time.monotonic()
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        item = metrics.setdefault(topic, {"count": 0, "first": now, "last": now, "errors": []})
        if "stamp" in item and stamp <= item["stamp"]:
            if "timestamps_not_increasing" not in item["errors"]:
                item["errors"].append("timestamps_not_increasing")
        item.update(count=item["count"] + 1, last=now, stamp=stamp, frame_id=message.header.frame_id)
        if kind == "image":
            item.update(width=message.width, height=message.height, encoding=message.encoding)
            if not message.width or not message.height or len(message.data) != message.height * message.step:
                if "invalid_image_layout" not in item["errors"]:
                    item["errors"].append("invalid_image_layout")
            if not message.header.frame_id:
                if "missing_frame_id" not in item["errors"]:
                    item["errors"].append("missing_frame_id")
            frames.add(message.header.frame_id)
            if topic.endswith("/depth/image_raw") and message.encoding == "16UC1":
                values = array("H")
                values.frombytes(bytes(message.data))
                if bool(message.is_bigendian) != (sys.byteorder == "big"):
                    values.byteswap()
                # A sparse depth check keeps validation overhead low.
                sampled = values[::max(1, len(values) // 2048)]
                fraction = sum(value > 0 for value in sampled) / max(1, len(sampled))
                item["valid_depth_fraction"] = fraction
                item["max_valid_depth_fraction"] = max(fraction, item.get("max_valid_depth_fraction", 0))
        elif kind == "info":
            item.update(width=message.width, height=message.height, distortion_model=message.distortion_model,
                        k=list(message.k), d=list(message.d), p=list(message.p))
            if not message.width or not message.height or message.k[0] <= 0 or message.k[4] <= 0:
                if "invalid_intrinsics" not in item["errors"]:
                    item["errors"].append("invalid_intrinsics")
        else:
            item.update(width=message.width, height=message.height, fields=[field.name for field in message.fields])
            if not {"x", "y", "z"}.issubset(item["fields"]):
                if "missing_xyz_fields" not in item["errors"]:
                    item["errors"].append("missing_xyz_fields")
            item["max_points"] = max(item.get("max_points", 0), message.width * message.height)

    def record_tf(message):
        for transform in message.transforms:
            tf_edges.add((transform.header.frame_id, transform.child_frame_id))

    report = {}
    try:
        rclpy.init()
        initialized = True
        node = rclpy.create_node("gemini335l_interface_check")
        subscriptions = []
        expected = []
        for stream in ("color", "depth", "left_ir", "right_ir"):
            for suffix, message_type, kind in (("image_raw", Image, "image"), ("camera_info", CameraInfo, "info")):
                topic = f"{prefix}/{stream}/{suffix}"
                expected.append(topic)
                subscriptions.append(node.create_subscription(
                    message_type, topic, lambda message, t=topic, k=kind: record(t, message, k), qos_profile_sensor_data
                ))
        point_topic = f"{prefix}/depth/points"
        expected.append(point_topic)
        subscriptions.append(node.create_subscription(
            PointCloud2, point_topic, lambda message: record(point_topic, message, "points"), qos_profile_sensor_data
        ))
        static_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        subscriptions.append(node.create_subscription(TFMessage, "/tf_static", record_tf, static_qos))
        subscriptions.append(node.create_subscription(TFMessage, "/tf", record_tf, qos_profile_sensor_data))
        if args.start_driver:
            driver_log = output.with_suffix(".driver.log").open("w", encoding="utf-8")
            driver = subprocess.Popen(
                [str(project / "scripts" / "run_gemini335l_ros2.sh"), f"camera_name:={args.camera_name}"],
                stdout=driver_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if driver is not None and driver.poll() is not None:
                break

        failures = []
        for topic in expected:
            item = metrics.get(topic)
            if item is None or item["count"] < 3:
                failures.append(f"未收到足够数据：{topic}")
                continue
            item["received_hz"] = (item["count"] - 1) / max(item["last"] - item["first"], 1e-6)
            failures.extend(f"{topic}: {error}" for error in item["errors"])
        for stream in ("color", "depth", "left_ir", "right_ir"):
            image = metrics.get(f"{prefix}/{stream}/image_raw")
            info = metrics.get(f"{prefix}/{stream}/camera_info")
            if image and info and (image["width"], image["height"], image["frame_id"]) != (info["width"], info["height"], info["frame_id"]):
                failures.append(f"{stream} 图像与 CameraInfo 的尺寸/坐标系不一致")
        depth = metrics.get(f"{prefix}/depth/image_raw", {})
        if depth and depth.get("max_valid_depth_fraction", 0) <= 0:
            failures.append("深度图没有有效深度值")
        if point_topic in metrics and metrics[point_topic].get("max_points", 0) <= 0:
            failures.append("点云为空")
        # Every optical frame must be connected to the named camera's root.
        connected = {args.camera_name + "_link"}
        for _ in range(len(tf_edges) + 1):
            for parent, child in tf_edges:
                if parent in connected:
                    connected.add(child)
        if not frames or not frames.issubset(connected):
            failures.append("相机 TF 未覆盖全部图像光学坐标系")
        if driver is not None and driver.poll() is not None:
            failures.append(f"驱动提前退出，返回码 {driver.returncode}")
        for item in metrics.values():
            item.pop("first", None)
            item.pop("last", None)
        report = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "passed": not failures, "failures": failures,
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
            "fastdds_builtin_transports": os.environ.get("FASTDDS_BUILTIN_TRANSPORTS", "DEFAULT"),
            "topics": metrics, "tf_edges": sorted(tf_edges),
        }
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"验证报告：{output}")
        return 0 if not failures else 1
    finally:
        if driver is not None and driver.poll() is None:
            os.killpg(driver.pid, signal.SIGINT)
            try:
                driver.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(driver.pid, signal.SIGTERM)
                try:
                    driver.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(driver.pid, signal.SIGKILL)
                    driver.wait(timeout=3)
        if driver_log is not None:
            driver_log.close()
        if node is not None:
            node.destroy_node()
        if initialized:
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
