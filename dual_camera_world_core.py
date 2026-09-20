"""Frozen extrinsics and approximately paired two-camera world pose comparison."""
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from world_calibration_core import calibrate_world, reference_world_pose, rigid_transform


@dataclass
class Observation:
    key: tuple
    received_at: float
    poses: dict
    errors: dict

    @classmethod
    def from_frame(cls, frame):
        return cls(frame.stamp, frame.received_at,
                   {p.tag_id: rigid_transform(p.rotation_matrix, p.tvec) for p in frame.poses},
                   {p.tag_id: p.reprojection_error_px for p in frame.poses})


def pose_difference(hik_world, gemini_world):
    delta = (hik_world[:3, 3] - gemini_world[:3, 3]) * 1000
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(
        gemini_world[:3, :3].T @ hik_world[:3, :3])[0])))
    return {"delta_xyz_mm": delta.tolist(), "distance_mm": float(np.linalg.norm(delta)),
            "rotation_difference_deg": angle}


@dataclass
class DifferenceStatistics:
    count: int = 0
    sum_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sum_squared_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sum_squared_angle: float = 0
    maximum_distance: float = 0

    def add(self, difference):
        delta = np.array(difference["delta_xyz_mm"])
        self.count += 1
        self.sum_delta += delta
        self.sum_squared_delta += delta * delta
        self.sum_squared_angle += difference["rotation_difference_deg"] ** 2
        self.maximum_distance = max(self.maximum_distance, difference["distance_mm"])

    def document(self):
        n = max(1, self.count)
        mean = self.sum_delta / n
        return {"pair_count": self.count, "mean_delta_xyz_mm": mean.tolist(),
                "std_delta_xyz_mm": np.sqrt(np.maximum(0, self.sum_squared_delta/n - mean*mean)).tolist(),
                "rms_distance_mm": float(np.sqrt(self.sum_squared_delta.sum()/n)),
                "max_distance_mm": self.maximum_distance,
                "rms_rotation_difference_deg": float(np.sqrt(self.sum_squared_angle/n))}


class WorldComparison:
    cameras = ("hik", "gemini")

    def __init__(self, settings, metadata=None):
        self.settings = settings
        self.metadata = metadata or {}
        self.known = reference_world_pose(settings["reference_xyz_mm"], settings["reference_rpy_deg"])
        self.reference_id = settings.get("reference_id", 0)
        self.queues = {c: deque(maxlen=12) for c in self.cameras}
        self.last_keys = {c: None for c in self.cameras}
        self.latest = {}
        self.calibrations = {}
        self.samples = {c: [] for c in self.cameras}
        self.statistics = {}
        self.pair = None
        self.collecting = False
        self.started = 0
        self.last_error = ""
        self.generation = 0

    def begin(self, now):
        self.calibrations.clear()
        self.samples = {c: [] for c in self.cameras}
        self.statistics.clear()
        self.pair = None
        for queue in self.queues.values():
            queue.clear()
        self.collecting = True
        self.started = now
        self.last_error = ""

    def tick(self, now):
        if self.collecting and now - self.started > self.settings.get("timeout_s", 30):
            self.collecting = False
            self.last_error = "标定超时：请让两台相机同时看到固定的 ID 0，然后重新开始"

    def ingest(self, camera, observation, now):
        self.tick(now)
        if (observation.key == self.last_keys[camera] or now - observation.received_at > self.settings.get("stale_s", .5)):
            return False
        self.last_keys[camera] = observation.key
        self.latest[camera] = observation
        self.queues[camera].append(observation)
        a, b = (self.queues[c] for c in self.cameras)
        tolerance = self.settings.get("pair_tolerance_ms", 80) / 1000
        changed = False
        while a and b:
            if abs(a[0].received_at - b[0].received_at) > tolerance:
                (a if a[0].received_at < b[0].received_at else b).popleft()
                continue
            pair = {"hik": a.popleft(), "gemini": b.popleft()}
            if any(now-o.received_at > self.settings.get("stale_s", .5) for o in pair.values()):
                continue
            if self.collecting:
                ref = self.reference_id
                if all(ref in o.poses and np.isfinite(o.errors[ref]) and
                       o.errors[ref] <= self.settings.get("max_reprojection_error_px", 1.5)
                       for o in pair.values()):
                    for c, o in pair.items():
                        self.samples[c].append(o.poses[ref])
                    if len(self.samples["hik"]) >= self.settings.get("sample_count", 60):
                        self.collecting = False
                        try:
                            results = {c: calibrate_world(self.samples[c], self.known,
                                max_translation_rms_mm=self.settings.get("max_translation_rms_mm", 2),
                                max_rotation_rms_deg=self.settings.get("max_rotation_rms_deg", 1),
                                metadata={"camera": c, **self.metadata.get(c, {})}) for c in self.cameras}
                            self.calibrations = results
                            self.generation += 1
                        except ValueError as exc:
                            self.last_error = str(exc)
                # Calibration observations never contribute to evaluation statistics.
                continue
            if len(self.calibrations) == 2:
                world = {c: {i: self.calibrations[c].world_pose(t) for i, t in o.poses.items()}
                         for c, o in pair.items()}
                differences = {}
                for i in world["hik"].keys() & world["gemini"].keys():
                    if any(pair[c].errors[i] > self.settings.get("max_reprojection_error_px", 1.5) for c in self.cameras):
                        continue
                    difference = pose_difference(world["hik"][i], world["gemini"][i])
                    self.statistics.setdefault(i, DifferenceStatistics()).add(difference)
                    differences[i] = difference
                self.pair = {"received_at": min(o.received_at for o in pair.values()),
                             "arrival_skew_ms": abs(pair["hik"].received_at - pair["gemini"].received_at)*1000,
                             "stamps": {c: o.key for c, o in pair.items()}, "differences": differences,
                             "world_poses": {c: {str(i): t.tolist() for i, t in poses.items()
                                if pair[c].errors[i] <= self.settings.get("max_reprojection_error_px", 1.5)}
                                for c, poses in world.items()}}
                changed = True
        return changed

    def world_poses(self, now):
        # Display the exact paired estimates used by the numeric difference.
        if self.pair and now-self.pair["received_at"] <= self.settings.get("stale_s", .5):
            return {c: {int(i): np.array(t) for i, t in poses.items()}
                    for c, poses in self.pair["world_poses"].items()}
        return {c: {i: self.calibrations[c].world_pose(t) for i, t in o.poses.items()
                    if o.errors[i] <= self.settings.get("max_reprojection_error_px", 1.5)}
                for c, o in self.latest.items() if c in self.calibrations and
                now - o.received_at <= self.settings.get("stale_s", .5)}

    def report(self, now):
        fresh = self.pair is not None and now-self.pair["received_at"] <= self.settings.get("stale_s", .5)
        return {"calibrated": len(self.calibrations) == 2, "collecting": self.collecting,
                "samples": len(self.samples["hik"]), "error": self.last_error,
                "world_reference_xyz_mm": (self.known[:3, 3]*1000).tolist(),
                "difference_convention": "HIK minus Gemini",
                "pairing": "host monotonic arrival times; not hardware synchronized; compare stationary targets",
                "current_pair": self.pair if fresh else None,
                "world_poses": {c: {str(i): t.tolist() for i, t in poses.items()}
                                for c, poses in self.world_poses(now).items()},
                "statistics": {str(i): s.document() for i, s in self.statistics.items()}}
