"""Benchmark four combined-feature binary XGBoost heads on CPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

try:
    from ml.evaluate_xgboost_baseline import aggregate_sequences
    from ml.train_tcn_corn import LABEL_NAMES
except ModuleNotFoundError:
    from evaluate_xgboost_baseline import aggregate_sequences
    from train_tcn_corn import LABEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/artifacts/data-behavior"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("ml/artifacts/model-binary-xgboost-behavior")
    )
    parser.add_argument("--repetitions", type=int, default=200)
    return parser.parse_args()


def measure(function, repetitions: int) -> np.ndarray:
    for _ in range(10):
        function()
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        durations.append((time.perf_counter() - started) * 1000)
    return np.asarray(durations)


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_order = json.loads((args.model_dir / "feature_order.json").read_text(encoding="utf-8"))
    with np.load(args.data_dir / "test.npz", allow_pickle=False) as data:
        raw = data["X"][:30]
    columns = aggregate_sequences(raw, metadata["feature_names"])
    matrix = np.column_stack([columns[name] for name in feature_order]).astype(np.float32)
    data_matrix = xgb.DMatrix(matrix, feature_names=feature_order)
    models = []
    for label_name in LABEL_NAMES:
        model = xgb.Booster()
        model.load_model(args.model_dir / f"model_binary_{label_name}.ubj")
        models.append(model)

    def prediction_only() -> None:
        for model in models:
            model.predict(data_matrix)

    def matrix_and_prediction() -> None:
        current = xgb.DMatrix(matrix, feature_names=feature_order)
        for model in models:
            model.predict(current)

    prediction = measure(prediction_only, args.repetitions)
    end_to_end = measure(matrix_and_prediction, args.repetitions)
    print(
        f"four-head prediction batch30 median={np.median(prediction):.3f} ms "
        f"p95={np.percentile(prediction, 95):.3f} ms"
    )
    print(
        f"DMatrix+prediction batch30 median={np.median(end_to_end):.3f} ms "
        f"p95={np.percentile(end_to_end, 95):.3f} ms"
    )
    print(f"throughput={30_000 / np.median(end_to_end):.1f} sequences/s")


if __name__ == "__main__":
    main()
