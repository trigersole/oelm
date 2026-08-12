"""Convert frame-level DAiSEE blendshape CSVs into fixed-length sequences.

The input CSVs are expected to be grouped by ``video_name`` and to contain the
52 MediaPipe blendshape columns followed by timestamp/frame metadata. Clip IDs
are normalized so ``123.avi`` in AllLabels.csv matches ``123`` in video_name.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


LABEL_NAMES = ("Boredom", "Engagement", "Confusion", "Frustration")
METADATA_COLUMNS = {"timestamp_ms", "original_frame", "video_name"}


def normalize_clip_id(value: object) -> str:
    clip_id = str(value).strip()
    if clip_id.lower().endswith(".avi"):
        clip_id = clip_id[:-4]
    return clip_id


def load_labels(path: Path) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        required = {"ClipID", *LABEL_NAMES}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for raw in reader:
            row = {key.strip(): value.strip() for key, value in raw.items()}
            clip_id = normalize_clip_id(row["ClipID"])
            target = np.asarray([int(row[name]) for name in LABEL_NAMES], dtype=np.int64)
            if np.any((target < 0) | (target > 3)):
                raise ValueError(f"Invalid labels for {clip_id}: {target.tolist()}")
            previous = labels.get(clip_id)
            if previous is not None and not np.array_equal(previous, target):
                raise ValueError(f"Conflicting duplicate label for {clip_id}")
            labels[clip_id] = target
    return labels


def resample_clip(
    timestamps: np.ndarray,
    values: np.ndarray,
    frames: int = 50,
    interval_ms: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample one clip and return values plus an observed-frame mask."""
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("A clip must contain at least one frame")
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order].astype(np.float64, copy=False)
    values = values[order].astype(np.float32, copy=False)

    unique_timestamps, unique_indices = np.unique(timestamps, return_index=True)
    values = values[unique_indices]
    target_times = np.arange(frames, dtype=np.float64) * interval_ms
    output = np.empty((frames, values.shape[1]), dtype=np.float32)
    for feature_index in range(values.shape[1]):
        output[:, feature_index] = np.interp(
            target_times,
            unique_timestamps,
            values[:, feature_index],
            left=float(values[0, feature_index]),
            right=float(values[-1, feature_index]),
        )
    observed = np.isclose(
        target_times[:, None], unique_timestamps[None, :], atol=1.0
    ).any(axis=1)
    return output, observed.astype(np.uint8)


def iter_grouped_clips(
    path: Path,
) -> tuple[list[str], Iterable[tuple[str, np.ndarray, np.ndarray]]]:
    """Return feature names and a generator over contiguous clip groups."""
    handle = path.open(newline="", encoding="utf-8-sig")
    reader = csv.reader(handle)
    try:
        header = next(reader)
    except StopIteration as exc:
        handle.close()
        raise ValueError(f"Empty CSV: {path}") from exc

    required = {"timestamp_ms", "video_name"}
    if not required.issubset(header):
        handle.close()
        raise ValueError(f"{path} must contain {sorted(required)}")
    feature_names = [name for name in header if name not in METADATA_COLUMNS]
    if len(feature_names) != 52:
        handle.close()
        raise ValueError(f"Expected 52 blendshapes in {path}, found {len(feature_names)}")
    feature_indices = [header.index(name) for name in feature_names]
    timestamp_index = header.index("timestamp_ms")
    video_index = header.index("video_name")

    def generator() -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
        current_id: str | None = None
        timestamps: list[float] = []
        rows: list[list[float]] = []
        completed: set[str] = set()
        try:
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(
                        f"{path}:{line_number} has {len(row)} fields; expected {len(header)}"
                    )
                clip_id = normalize_clip_id(row[video_index])
                if current_id is None:
                    current_id = clip_id
                if clip_id != current_id:
                    completed.add(current_id)
                    yield current_id, np.asarray(timestamps), np.asarray(rows, dtype=np.float32)
                    if clip_id in completed:
                        raise ValueError(
                            f"{path} is not grouped by video_name; {clip_id} reappeared"
                        )
                    current_id, timestamps, rows = clip_id, [], []
                timestamps.append(float(row[timestamp_index]))
                rows.append([float(row[index]) for index in feature_indices])
            if current_id is not None:
                yield current_id, np.asarray(timestamps), np.asarray(rows, dtype=np.float32)
        finally:
            handle.close()

    return feature_names, generator()


def prepare_split(
    csv_path: Path,
    output_path: Path,
    labels: dict[str, np.ndarray],
    expected_features: list[str] | None,
    frames: int,
    interval_ms: float,
) -> tuple[dict[str, object], list[str], set[str]]:
    feature_names, clips = iter_grouped_clips(csv_path)
    if expected_features is not None and feature_names != expected_features:
        raise ValueError(f"Feature order differs in {csv_path}")

    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    clip_ids: list[str] = []
    missing_labels: list[str] = []
    original_frame_counts: list[int] = []

    for clip_id, timestamps, values in clips:
        if clip_id not in labels:
            missing_labels.append(clip_id)
            continue
        sequence, observed = resample_clip(timestamps, values, frames, interval_ms)
        sequences.append(sequence)
        targets.append(labels[clip_id])
        masks.append(observed)
        clip_ids.append(clip_id)
        original_frame_counts.append(len(values))

    if not sequences:
        raise ValueError(f"No labeled clips found in {csv_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=np.stack(sequences).astype(np.float32),
        y=np.stack(targets).astype(np.int64),
        observed=np.stack(masks).astype(np.uint8),
        clip_ids=np.asarray(clip_ids),
    )

    distributions = {
        name: {str(level): int(count) for level, count in sorted(Counter(
            int(target[index]) for target in targets
        ).items())}
        for index, name in enumerate(LABEL_NAMES)
    }
    report: dict[str, object] = {
        "source": str(csv_path),
        "output": str(output_path),
        "labeled_clips": len(sequences),
        "missing_label_count": len(missing_labels),
        "missing_label_clip_ids": missing_labels,
        "original_frames_min": min(original_frame_counts),
        "original_frames_max": max(original_frame_counts),
        "fully_observed_clips": int(sum(mask.all() for mask in masks)),
        "label_distribution": distributions,
    }
    return report, feature_names, set(clip_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ml/artifacts/data")
    )
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--interval-ms", type=float, default=200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    labels = load_labels(root / "AllLabels.csv")
    split_files = {
        "train": root / "train_blendshapes.csv",
        "validation": root / "validation_blendshapes.csv",
        "test": root / "test_blendshapes.csv",
    }

    reports: dict[str, object] = {}
    split_ids: dict[str, set[str]] = {}
    feature_names: list[str] | None = None
    for split_name, csv_path in split_files.items():
        report, current_features, ids = prepare_split(
            csv_path,
            output_dir / f"{split_name}.npz",
            labels,
            feature_names,
            args.frames,
            args.interval_ms,
        )
        feature_names = current_features
        reports[split_name] = report
        split_ids[split_name] = ids
        print(
            f"{split_name}: {report['labeled_clips']} labeled clips, "
            f"{report['missing_label_count']} without labels"
        )

    overlaps = {}
    split_names = list(split_ids)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(split_ids[left].intersection(split_ids[right]))
            overlaps[f"{left}_{right}"] = overlap
            if overlap:
                raise ValueError(f"Clip leakage between {left} and {right}: {overlap[:5]}")

    metadata = {
        "frames": args.frames,
        "interval_ms": args.interval_ms,
        "feature_names": feature_names,
        "label_names": LABEL_NAMES,
        "total_labels": len(labels),
        "split_overlaps": overlaps,
        "splits": reports,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Prepared sequences in {output_dir}")


if __name__ == "__main__":
    main()

