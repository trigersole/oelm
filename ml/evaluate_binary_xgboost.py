"""Collapse the existing four-class XGBoost probabilities into Low/High."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

try:
    from ml.evaluate_xgboost_baseline import aggregate_sequences
    from ml.train_tcn_corn import LABEL_NAMES, calculate_metrics, tune_binary_thresholds
except ModuleNotFoundError:
    from evaluate_xgboost_baseline import aggregate_sequences
    from train_tcn_corn import LABEL_NAMES, calculate_metrics, tune_binary_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/artifacts/data"))
    parser.add_argument("--models-dir", type=Path, default=Path("Models"))
    parser.add_argument("--feature-order", type=Path, default=Path("feature_order.pkl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/artifacts/binary/xgboost_collapsed_metrics.json"),
    )
    return parser.parse_args()


def load_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["X"], (data["y"] >= 2).astype(np.int64)


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_names = metadata["feature_names"]
    feature_order = joblib.load(args.feature_order)
    validation_x, validation_y = load_raw(args.data_dir / "validation.npz")
    test_x, test_y = load_raw(args.data_dir / "test.npz")

    models = []
    for label_name in LABEL_NAMES:
        model = xgb.Booster()
        model.load_model(args.models_dir / f"model_tuned_smote_{label_name}.ubj")
        models.append(model)

    def predict(raw: np.ndarray) -> np.ndarray:
        columns = aggregate_sequences(raw, feature_names)
        matrix = np.column_stack([columns[name] for name in feature_order]).astype(np.float32)
        data_matrix = xgb.DMatrix(matrix, feature_names=feature_order)
        high_probabilities = []
        for model in models:
            output = np.asarray(model.predict(data_matrix))
            if output.ndim != 2 or output.shape[1] != 4:
                raise ValueError(f"Expected four-class probabilities, found {output.shape}")
            high_probabilities.append(output[:, 2:].sum(axis=1))
        high = np.column_stack(high_probabilities)
        return np.stack([1.0 - high, high], axis=-1)

    validation_probabilities = predict(validation_x)
    test_probabilities = predict(test_x)
    thresholds = tune_binary_thresholds(validation_y, validation_probabilities)
    validation_predictions = (
        validation_probabilities[..., 1] >= thresholds[None, :]
    ).astype(np.int64)
    test_predictions = (
        test_probabilities[..., 1] >= thresholds[None, :]
    ).astype(np.int64)
    default_predictions = (test_probabilities[..., 1] >= 0.5).astype(np.int64)
    report = {
        "model": "Existing four-class XGBoost collapsed to binary",
        "target_definition": "Low=levels 0/1; High=levels 2/3",
        "threshold_selection": "Per-label validation macro-F1",
        "decision_thresholds": thresholds.tolist(),
        "validation": calculate_metrics(validation_y, validation_predictions, levels=2),
        "test": calculate_metrics(test_y, test_predictions, levels=2),
        "test_at_default_threshold": calculate_metrics(
            test_y, default_predictions, levels=2
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

