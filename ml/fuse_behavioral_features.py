"""Append extracted behavioural features to prepared blendshape sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fuse_split(base_path: Path, behavior_path: Path, output_path: Path) -> dict[str, object]:
    with np.load(base_path, allow_pickle=False) as data:
        base_x = data["X"]
        targets = data["y"]
        observed = data["observed"]
        clip_ids = data["clip_ids"].astype(str)
    with np.load(behavior_path, allow_pickle=False) as data:
        behavior_x = data["X"]
        behavior_ids = data["clip_ids"].astype(str)

    if not np.array_equal(clip_ids, behavior_ids):
        raise ValueError(f"Clip IDs differ between {base_path} and {behavior_path}")
    if base_x.shape[:2] != behavior_x.shape[:2]:
        raise ValueError(f"Sequence shapes differ: {base_x.shape} vs {behavior_x.shape}")
    fused = np.concatenate((base_x, behavior_x), axis=-1).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=fused,
        y=targets,
        observed=observed,
        clip_ids=clip_ids,
    )
    return {
        "clips": len(clip_ids),
        "frames": int(fused.shape[1]),
        "blendshape_features": int(base_x.shape[-1]),
        "behavioral_features": int(behavior_x.shape[-1]),
        "total_features": int(fused.shape[-1]),
        "behavior_frame_detection_rate": float(behavior_x[..., -1].mean()),
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("ml/artifacts/data"))
    parser.add_argument("--behavior-dir", type=Path, default=Path("ml/artifacts/behavior"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/data-behavior"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_metadata = json.loads(
        (args.base_dir / "metadata.json").read_text(encoding="utf-8")
    )
    behavior_metadata = json.loads(
        (args.behavior_dir / "metadata.json").read_text(encoding="utf-8")
    )
    feature_names = [
        *base_metadata["feature_names"],
        *behavior_metadata["feature_names"],
    ]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Duplicate names in fused feature list")
    reports = {}
    for split_name in ("train", "validation", "test"):
        reports[split_name] = fuse_split(
            args.base_dir / f"{split_name}.npz",
            args.behavior_dir / f"{split_name}.npz",
            args.output_dir / f"{split_name}.npz",
        )
        print(f"{split_name}: {reports[split_name]}")
    metadata = {
        "feature_names": feature_names,
        "blendshape_feature_names": base_metadata["feature_names"],
        "behavioral_feature_names": behavior_metadata["feature_names"],
        "frames": base_metadata["frames"],
        "interval_ms": base_metadata["interval_ms"],
        "splits": reports,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
