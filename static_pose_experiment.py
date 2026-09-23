#!/usr/bin/env python3
"""Analyze unsmoothed static-pose repeatability from cube-pose JSONL records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def load_frames(path: Path) -> list[dict]:
    frames = []
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(frame, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            frames.append(frame)
    if not frames:
        raise ValueError("The JSONL file contains no frames")
    return frames


def _mean_rotation(matrices: np.ndarray) -> Rotation:
    return Rotation.from_matrix(matrices).mean()


def summarize(samples: list[dict]) -> tuple[dict, list[dict]]:
    translations = np.asarray([item["translation_mm"] for item in samples], float)
    matrices = np.asarray([item["rotation_matrix"] for item in samples], float)
    mean_translation = translations.mean(axis=0)
    translation_delta = translations - mean_translation
    translation_norm = np.linalg.norm(translation_delta, axis=1)

    mean_rotation = _mean_rotation(matrices)
    rotation_delta = mean_rotation.inv() * Rotation.from_matrix(matrices)
    rotation_vectors_deg = np.rad2deg(rotation_delta.as_rotvec())
    rotation_magnitude_deg = np.linalg.norm(rotation_vectors_deg, axis=1)

    window = min(10, max(1, len(samples) // 5))
    first_translation = translations[:window].mean(axis=0)
    last_translation = translations[-window:].mean(axis=0)
    first_rotation = _mean_rotation(matrices[:window])
    last_rotation = _mean_rotation(matrices[-window:])
    drift_rotation = math.degrees((first_rotation.inv() * last_rotation).magnitude())

    ddof = 1 if len(samples) > 1 else 0
    summary = {
        "valid_sample_count": len(samples),
        "mean_translation_mm": mean_translation.tolist(),
        "translation_std_xyz_mm": translations.std(axis=0, ddof=ddof).tolist(),
        "translation_peak_to_peak_xyz_mm": np.ptp(translations, axis=0).tolist(),
        "translation_radial_rms_mm": float(np.sqrt(np.mean(translation_norm**2))),
        "translation_radial_p95_mm": float(np.percentile(translation_norm, 95)),
        "translation_radial_max_mm": float(translation_norm.max()),
        "mean_quaternion_xyzw": mean_rotation.as_quat().tolist(),
        "rotation_std_rotvec_xyz_deg": rotation_vectors_deg.std(
            axis=0, ddof=ddof
        ).tolist(),
        "rotation_rms_deg": float(
            np.sqrt(np.mean(rotation_magnitude_deg**2))
        ),
        "rotation_p95_deg": float(np.percentile(rotation_magnitude_deg, 95)),
        "rotation_max_deg": float(rotation_magnitude_deg.max()),
        "first_last_window": {
            "window_sample_count": window,
            "translation_delta_xyz_mm": (
                last_translation - first_translation
            ).tolist(),
            "translation_delta_norm_mm": float(
                np.linalg.norm(last_translation - first_translation)
            ),
            "rotation_delta_deg": float(drift_rotation),
        },
    }
    rows = []
    for item, delta, delta_norm, rotation_vector, rotation_magnitude in zip(
        samples,
        translation_delta,
        translation_norm,
        rotation_vectors_deg,
        rotation_magnitude_deg,
        strict=True,
    ):
        rows.append(
            {
                **item,
                "translation_delta_mm": delta.tolist(),
                "translation_delta_norm_mm": float(delta_norm),
                "rotation_delta_rotvec_deg": rotation_vector.tolist(),
                "rotation_delta_deg": float(rotation_magnitude),
            }
        )
    return summary, rows


def extract_entities(frames: list[dict]) -> dict[str, list[dict]]:
    entities: dict[str, list[dict]] = {"cube_03": [], "cube_04": [], "cube_03_to_04": []}
    for frame in frames:
        base = {
            "frame_number": int(frame["frame_number"]),
            "host_receive_time_ns": int(frame["host_receive_time_ns"]),
        }
        for cube in frame.get("cubes", []):
            cube_number = int(cube.get("cube", -1))
            key = f"cube_{cube_number:02d}"
            if key not in entities or not cube.get("pose_valid") or not cube.get("pose"):
                continue
            pose = cube["pose"]
            entities[key].append(
                {
                    **base,
                    "translation_mm": pose["translation_mm"],
                    "rotation_matrix": pose["rotation_matrix"],
                    "status": cube["status"],
                    "used_ids": cube.get("used_ids", []),
                    "reprojection_rms_px": cube.get("reprojection_rms_px"),
                }
            )
        pair = frame.get("cube_pair_relative")
        if pair and pair.get("from_cube") == 3 and pair.get("to_cube") == 4:
            pose = pair.get("transform")
            if pose:
                entities["cube_03_to_04"].append(
                    {
                        **base,
                        "translation_mm": pose["translation_mm"],
                        "rotation_matrix": pose["rotation_matrix"],
                        "status": "VALID_RELATIVE",
                        "used_ids": [],
                        "reprojection_rms_px": None,
                    }
                )
    return entities


def _write_csv(path: Path, rows_by_entity: dict[str, list[dict]], start_ns: int) -> None:
    fields = [
        "entity", "frame_number", "time_s", "x_mm", "y_mm", "z_mm",
        "dx_mm", "dy_mm", "dz_mm", "translation_delta_norm_mm",
        "rotation_delta_x_deg", "rotation_delta_y_deg", "rotation_delta_z_deg",
        "rotation_delta_deg", "status", "used_ids", "reprojection_rms_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entity, rows in rows_by_entity.items():
            for row in rows:
                xyz = row["translation_mm"]
                delta = row["translation_delta_mm"]
                rdelta = row["rotation_delta_rotvec_deg"]
                writer.writerow(
                    {
                        "entity": entity,
                        "frame_number": row["frame_number"],
                        "time_s": (row["host_receive_time_ns"] - start_ns) / 1e9,
                        "x_mm": xyz[0], "y_mm": xyz[1], "z_mm": xyz[2],
                        "dx_mm": delta[0], "dy_mm": delta[1], "dz_mm": delta[2],
                        "translation_delta_norm_mm": row["translation_delta_norm_mm"],
                        "rotation_delta_x_deg": rdelta[0],
                        "rotation_delta_y_deg": rdelta[1],
                        "rotation_delta_z_deg": rdelta[2],
                        "rotation_delta_deg": row["rotation_delta_deg"],
                        "status": row["status"],
                        "used_ids": " ".join(map(str, row["used_ids"])),
                        "reprojection_rms_px": row["reprojection_rms_px"],
                    }
                )


def _write_plot(path: Path, rows_by_entity: dict[str, list[dict]], start_ns: int) -> None:
    labels = {
        "cube_03": "Cube 03 vs camera",
        "cube_04": "Cube 04 vs camera",
        "cube_03_to_04": "Cube 04 vs Cube 03",
    }
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    for row_index, (entity, rows) in enumerate(rows_by_entity.items()):
        times = np.asarray(
            [(row["host_receive_time_ns"] - start_ns) / 1e9 for row in rows]
        )
        delta = np.asarray([row["translation_delta_mm"] for row in rows])
        angle = np.asarray([row["rotation_delta_deg"] for row in rows])
        for component, name in enumerate(("X", "Y", "Z")):
            axes[row_index, 0].plot(times, delta[:, component], label=name, lw=1)
        axes[row_index, 0].axhline(0, color="black", lw=0.6)
        axes[row_index, 0].set_ylabel(f"{labels[entity]}\nΔ position (mm)")
        axes[row_index, 0].legend(loc="upper right", ncol=3, fontsize=8)
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].plot(times, angle, color="#9c2f5f", lw=1)
        axes[row_index, 1].set_ylabel("Rotation deviation (deg)")
        axes[row_index, 1].grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("CH120-10GM-1 static pose repeatability (no smoothing)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(input_path: Path, output_dir: Path) -> dict:
    frames = load_frames(input_path)
    entities = extract_entities(frames)
    if any(not rows for rows in entities.values()):
        missing = [name for name, rows in entities.items() if not rows]
        raise ValueError(f"No valid samples for: {', '.join(missing)}")
    start_ns = min(int(frame["host_receive_time_ns"]) for frame in frames)
    end_ns = max(int(frame["host_receive_time_ns"]) for frame in frames)
    summaries = {}
    normalized_rows = {}
    for entity, samples in entities.items():
        summaries[entity], normalized_rows[entity] = summarize(samples)
        if entity != "cube_03_to_04":
            summaries[entity]["status_counts"] = dict(
                Counter(item["status"] for item in samples)
            )
            summaries[entity]["used_id_sets"] = dict(
                Counter(" ".join(map(str, item["used_ids"])) for item in samples)
            )
            errors = np.asarray(
                [item["reprojection_rms_px"] for item in samples], float
            )
            summaries[entity]["reprojection_rms_px"] = {
                "mean": float(errors.mean()),
                "std": float(errors.std(ddof=1 if len(errors) > 1 else 0)),
                "max": float(errors.max()),
            }
    first = frames[0]
    report = {
        "schema_version": 1,
        "experiment": "static_pose_repeatability",
        "interpretation": (
            "Repeatability/jitter with stationary objects; no external ground truth, "
            "so this is not absolute pose accuracy."
        ),
        "input_jsonl": str(input_path.expanduser().resolve()),
        "frame_count": len(frames),
        "duration_s": (end_ns - start_ns) / 1e9,
        "image_size": first.get("image_size"),
        "source": first.get("source"),
        "calibration": first.get("calibration"),
        "cube_side_m": first.get("cube_side_m"),
        "tag_black_border_m": first.get("tag_black_border_m"),
        "smoothing": "none",
        "entities": summaries,
    }
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "static_repeatability_report.json"
    csv_path = output_dir / "static_repeatability_samples.csv"
    plot_path = output_dir / "static_repeatability_plot.png"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, normalized_rows, start_ns)
    _write_plot(plot_path, normalized_rows, start_ns)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="cube pose JSONL recording")
    parser.add_argument("--output-dir", type=Path, help="defaults to the JSONL directory")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.expanduser().resolve().parent
    try:
        report = analyze(args.input, output_dir)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(report["entities"], indent=2, ensure_ascii=False))
    print(f"Saved: {output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
