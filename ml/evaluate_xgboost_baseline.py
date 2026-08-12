"""Evaluate the existing XGBoost models on the prepared labeled test clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

try:
    from ml.train_tcn_corn import LABEL_NAMES, calculate_metrics
except ModuleNotFoundError:
    from train_tcn_corn import LABEL_NAMES, calculate_metrics


STATISTIC_NAMES = ("mean", "std", "min", "max", "median", "skew", "kurt")


def aggregate_sequences(raw: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    mean = raw.mean(axis=1)
    std = raw.std(axis=1)
    normalized = np.divide(
        raw - mean[:, None, :],
        std[:, None, :],
        out=np.zeros_like(raw),
        where=std[:, None, :] != 0,
    )
    values = {
        "mean": mean,
        "std": std,
        "min": raw.min(axis=1),
        "max": raw.max(axis=1),
        "median": np.median(raw, axis=1),
        "skew": (normalized**3).mean(axis=1),
        "kurt": (normalized**4).mean(axis=1) - 3,
    }
    return {
        f"{feature}_{statistic}": values[statistic][:, feature_index]
        for feature_index, feature in enumerate(feature_names)
        for statistic in STATISTIC_NAMES
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/artifacts/data"))
    parser.add_argument("--models-dir", type=Path, default=Path("Models"))
    parser.add_argument("--feature-order", type=Path, default=Path("feature_order.pkl"))
    parser.add_argument(
        "--output", type=Path, default=Path("ml/artifacts/model/xgboost_baseline_metrics.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.data_dir / "test.npz", allow_pickle=False) as data:
        raw, targets = data["X"], data["y"]
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    columns = aggregate_sequences(raw, metadata["feature_names"])
    feature_order = joblib.load(args.feature_order)
    missing = [name for name in feature_order if name not in columns]
    if missing:
        raise ValueError(f"Missing XGBoost input features: {missing[:5]}")
    matrix = np.column_stack([columns[name] for name in feature_order]).astype(np.float32)
    data_matrix = xgb.DMatrix(matrix, feature_names=feature_order)

    predictions = []
    for label_name in LABEL_NAMES:
        model = xgb.Booster()
        model.load_model(args.models_dir / f"model_tuned_smote_{label_name}.ubj")
        output = np.asarray(model.predict(data_matrix))
        predictions.append(output.argmax(axis=1) if output.ndim > 1 else output.astype(int))
    metrics = calculate_metrics(targets, np.column_stack(predictions))
    report = {
        "note": (
            "Existing XGBoost evaluated on the same resampled, labeled test clips as "
            "TCN-CORN. Test clips without labels are excluded from both models."
        ),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

