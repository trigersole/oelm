"""Extract lightweight behavioural signals from DAiSEE videos at 5 FPS.

The extractor deliberately follows clip IDs already present in the prepared
blendshape NPZ files. This keeps the train/validation/test population identical
for a fair ablation. MediaPipe provides 478 face/iris landmarks and a facial
transformation matrix; no label information is used during extraction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np


FEATURE_NAMES = (
    "head_pitch",
    "head_yaw",
    "head_roll",
    "gaze_x",
    "gaze_y",
    "eye_aspect_left",
    "eye_aspect_right",
    "mouth_aspect",
    "face_center_x",
    "face_center_y",
    "face_area",
    "face_detected",
)

_LANDMARKER = None
cv2 = None
mp = None
_FRAMES = 50
_INTERVAL_MS = 200.0


def _initialize_worker(model_path: str, frames: int, interval_ms: float) -> None:
    global _LANDMARKER, _FRAMES, _INTERVAL_MS, cv2, mp
    import cv2 as cv2_module
    import mediapipe as mediapipe_module

    cv2 = cv2_module
    mp = mediapipe_module
    _FRAMES = frames
    _INTERVAL_MS = interval_ms
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    _LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    cv2.setNumThreads(1)


def _point(landmarks: list, index: int) -> np.ndarray:
    item = landmarks[index]
    return np.asarray((item.x, item.y), dtype=np.float64)


def _distance(landmarks: list, left: int, right: int) -> float:
    return float(np.linalg.norm(_point(landmarks, left) - _point(landmarks, right)))


def _eye_aspect_ratio(landmarks: list, indices: tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = indices
    width = max(_distance(landmarks, p1, p4), 1e-6)
    return (_distance(landmarks, p2, p6) + _distance(landmarks, p3, p5)) / (2.0 * width)


def _gaze_ratio(
    landmarks: list, iris_indices: tuple[int, ...], eye_indices: tuple[int, ...]
) -> tuple[float, float]:
    iris = np.mean([_point(landmarks, index) for index in iris_indices], axis=0)
    eye = np.stack([_point(landmarks, index) for index in eye_indices])
    low, high = eye.min(axis=0), eye.max(axis=0)
    scale = np.maximum(high - low, 1e-6)
    ratio = (iris - low) / scale
    return float(np.clip(ratio[0], -0.5, 1.5)), float(np.clip(ratio[1], -0.5, 1.5))


def rotation_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return pitch/yaw/roll in radians from a 3x3 or 4x4 transform."""
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = 0.0
    return pitch, yaw, roll


def landmarks_to_features(result: object) -> np.ndarray | None:
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]
    if len(landmarks) < 478:
        return None

    pitch = yaw = roll = 0.0
    if result.facial_transformation_matrixes:
        pitch, yaw, roll = rotation_to_euler(result.facial_transformation_matrixes[0])

    right_gaze = _gaze_ratio(
        landmarks,
        (468, 469, 470, 471, 472),
        (33, 133, 159, 145, 160, 144, 158, 153),
    )
    left_gaze = _gaze_ratio(
        landmarks,
        (473, 474, 475, 476, 477),
        (263, 362, 386, 374, 387, 373, 385, 380),
    )
    gaze_x = (right_gaze[0] + left_gaze[0]) / 2.0
    gaze_y = (right_gaze[1] + left_gaze[1]) / 2.0
    left_ear = _eye_aspect_ratio(landmarks, (263, 386, 387, 362, 373, 380))
    right_ear = _eye_aspect_ratio(landmarks, (33, 160, 158, 133, 153, 144))
    mouth_width = max(_distance(landmarks, 61, 291), 1e-6)
    mouth_aspect = _distance(landmarks, 13, 14) / mouth_width

    coordinates = np.asarray([(item.x, item.y) for item in landmarks[:468]])
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    center = (low + high) / 2.0
    area = max(float((high[0] - low[0]) * (high[1] - low[1])), 0.0)
    return np.asarray(
        (
            pitch,
            yaw,
            roll,
            gaze_x,
            gaze_y,
            left_ear,
            right_ear,
            mouth_aspect,
            center[0],
            center[1],
            area,
            1.0,
        ),
        dtype=np.float32,
    )


def interpolate_missing(features: np.ndarray) -> np.ndarray:
    detected = features[:, -1] > 0.5
    if not np.any(detected):
        return features
    indices = np.arange(len(features))
    for column in range(features.shape[1] - 1):
        features[:, column] = np.interp(
            indices, indices[detected], features[detected, column]
        )
    return features


def extract_clip(item: tuple[str, str]) -> tuple[str, np.ndarray, bool, str | None]:
    clip_id, raw_path = item
    output = np.zeros((_FRAMES, len(FEATURE_NAMES)), dtype=np.float32)
    if not raw_path:
        return clip_id, output, False, "video_not_found"

    capture = cv2.VideoCapture(raw_path)
    if not capture.isOpened():
        return clip_id, output, False, "video_open_failed"
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    target_frames = np.rint(
        np.arange(_FRAMES, dtype=np.float64) * (_INTERVAL_MS / 1000.0) * fps
    ).astype(np.int64)
    targets: dict[int, list[int]] = {}
    for output_index, frame_index in enumerate(target_frames):
        targets.setdefault(int(frame_index), []).append(output_index)

    next_target = min(targets) if targets else None
    frame_index = 0
    try:
        while next_target is not None:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index == next_target:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = _LANDMARKER.detect(image)
                values = landmarks_to_features(result)
                if values is not None:
                    for output_index in targets[frame_index]:
                        output[output_index] = values
                remaining = [value for value in targets if value > frame_index]
                next_target = min(remaining) if remaining else None
            frame_index += 1
    except Exception as exc:  # preserve the rest of a long extraction run
        return clip_id, interpolate_missing(output), True, type(exc).__name__
    finally:
        capture.release()
    return clip_id, interpolate_missing(output), True, None


def build_video_index(split_dir: Path) -> dict[str, str]:
    candidates: dict[str, list[Path]] = {}
    for extension in ("*.avi", "*.mp4"):
        for path in split_dir.rglob(extension):
            candidates.setdefault(path.stem, []).append(path)
    index: dict[str, str] = {}
    for clip_id, paths in candidates.items():
        paths.sort(key=lambda path: (path.suffix.lower() != ".avi", str(path)))
        index[clip_id] = str(paths[0].resolve())
    return index


def load_clip_ids(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data["clip_ids"].astype(str)


def process_split(
    split_name: str,
    source_path: Path,
    video_dir: Path,
    output_dir: Path,
    model_path: Path,
    frames: int,
    interval_ms: float,
    workers: int,
    limit: int | None,
) -> dict[str, object]:
    clip_ids = load_clip_ids(source_path)
    if limit is not None:
        clip_ids = clip_ids[:limit]
    video_index = build_video_index(video_dir)
    items = [(clip_id, video_index.get(clip_id, "")) for clip_id in clip_ids]
    started = time.perf_counter()
    sequences: list[np.ndarray] = []
    found = 0
    errors: dict[str, int] = {}

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(str(model_path), frames, interval_ms),
    ) as executor:
        results: Iterable[tuple[str, np.ndarray, bool, str | None]] = executor.map(
            extract_clip, items, chunksize=1
        )
        for index, (clip_id, values, video_found, error) in enumerate(results, start=1):
            if clip_id != items[index - 1][0]:
                raise RuntimeError("Worker results changed clip order")
            sequences.append(values)
            found += int(video_found)
            if error:
                errors[error] = errors.get(error, 0) + 1
            if index % 50 == 0 or index == len(items):
                elapsed = time.perf_counter() - started
                print(
                    f"{split_name}: {index}/{len(items)} clips "
                    f"({index / max(elapsed, 1e-6):.2f} clips/s)",
                    flush=True,
                )

    values = np.stack(sequences).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(output_path, X=values, clip_ids=clip_ids)
    detection_rate = float(values[..., -1].mean())
    return {
        "split": split_name,
        "clips": len(clip_ids),
        "videos_found": found,
        "frame_detection_rate": detection_rate,
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/artifacts/data"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("DAiSEE/DataSet"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/behavior"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("ml/artifacts/face_landmarker.task")
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--interval-ms", type=float, default=200.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Per-split smoke-test limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    reports = []
    split_directories = {"train": "Train", "validation": "Validation", "test": "Test"}
    for split_name in args.splits:
        report = process_split(
            split_name,
            args.data_dir / f"{split_name}.npz",
            args.dataset_dir / split_directories[split_name],
            args.output_dir,
            args.model_path.resolve(),
            args.frames,
            args.interval_ms,
            max(args.workers, 1),
            args.limit,
        )
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    metadata = {
        "feature_names": FEATURE_NAMES,
        "frames": args.frames,
        "interval_ms": args.interval_ms,
        "reports": reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
