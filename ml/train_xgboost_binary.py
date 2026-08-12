"""Train four balanced binary XGBoost models for Low (0/1) vs High (2/3)."""

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
    parser.add_argument("--feature-order", type=Path, default=Path("feature_order.pkl"))
    parser.add_argument(
        "--include-extra-features",
        action="store_true",
        help=(
            "Append aggregate columns not listed in feature_order.pkl. Use this "
            "for fused blendshape + behavioural sequences."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ml/artifacts/model-binary-xgboost")
    )
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["X"], (data["y"] >= 2).astype(np.int64), data["clip_ids"]


def balanced_weights(targets: np.ndarray) -> np.ndarray:
    counts = np.bincount(targets, minlength=2).astype(np.float64)
    return np.where(
        targets == 0,
        len(targets) / (2 * counts[0]),
        len(targets) / (2 * counts[1]),
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_order = list(joblib.load(args.feature_order))

    raw_train, train_y, _ = load_split(args.data_dir / "train.npz")
    raw_validation, validation_y, _ = load_split(args.data_dir / "validation.npz")
    raw_test, test_y, test_ids = load_split(args.data_dir / "test.npz")

    train_columns = aggregate_sequences(raw_train, metadata["feature_names"])
    if args.include_extra_features:
        known = set(feature_order)
        feature_order.extend(name for name in train_columns if name not in known)
    missing = [name for name in feature_order if name not in train_columns]
    if missing:
        raise ValueError(f"Missing aggregate features: {missing[:5]}")

    def matrix(raw: np.ndarray, columns: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if columns is None:
            columns = aggregate_sequences(raw, metadata["feature_names"])
        return np.column_stack([columns[name] for name in feature_order]).astype(np.float32)

    train_x = matrix(raw_train, train_columns)
    validation_x = matrix(raw_validation)
    test_x = matrix(raw_test)
    validation_probabilities = np.empty((len(validation_x), 4, 2), dtype=np.float32)
    test_probabilities = np.empty((len(test_x), 4, 2), dtype=np.float32)
    best_iterations: dict[str, int] = {}
    class_weights: dict[str, dict[str, float]] = {}

    parameters = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "alpha": 0.1,
        "tree_method": "hist",
        "seed": args.seed,
        "nthread": 4,
    }
    for task, label_name in enumerate(LABEL_NAMES):
        weights = balanced_weights(train_y[:, task])
        class_weights[label_name] = {
            "low": float(weights[train_y[:, task] == 0][0]),
            "high": float(weights[train_y[:, task] == 1][0]),
        }
        train_matrix = xgb.DMatrix(
            train_x,
            label=train_y[:, task],
            weight=weights,
            feature_names=feature_order,
        )
        validation_matrix = xgb.DMatrix(
            validation_x, label=validation_y[:, task], feature_names=feature_order
        )
        test_matrix = xgb.DMatrix(test_x, feature_names=feature_order)
        model = xgb.train(
            parameters,
            train_matrix,
            num_boost_round=args.rounds,
            evals=[(validation_matrix, "validation")],
            early_stopping_rounds=args.early_stopping,
            verbose_eval=False,
        )
        best_iterations[label_name] = int(model.best_iteration)
        model.save_model(args.output_dir / f"model_binary_{label_name}.ubj")
        validation_high = model.predict(validation_matrix, iteration_range=(0, model.best_iteration + 1))
        test_high = model.predict(test_matrix, iteration_range=(0, model.best_iteration + 1))
        validation_probabilities[:, task] = np.column_stack([1 - validation_high, validation_high])
        test_probabilities[:, task] = np.column_stack([1 - test_high, test_high])

    thresholds = tune_binary_thresholds(validation_y, validation_probabilities)
    validation_predictions = (
        validation_probabilities[..., 1] >= thresholds[None, :]
    ).astype(np.int64)
    test_predictions = (
        test_probabilities[..., 1] >= thresholds[None, :]
    ).astype(np.int64)
    report = {
        "model": "Balanced binary XGBoost",
        "seed": args.seed,
        "raw_feature_count": int(raw_train.shape[-1]),
        "aggregate_feature_count": len(feature_order),
        "include_extra_features": args.include_extra_features,
        "target_definition": "Low=levels 0/1; High=levels 2/3",
        "training_balance": "Each side contributes 50% of training loss per label",
        "threshold_selection": "Per-label validation macro-F1",
        "decision_thresholds": thresholds.tolist(),
        "class_weights": class_weights,
        "best_iterations": best_iterations,
        "parameters": parameters,
        "validation": calculate_metrics(validation_y, validation_predictions, levels=2),
        "test": calculate_metrics(test_y, test_predictions, levels=2),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "feature_order.json").write_text(
        json.dumps(feature_order, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        clip_ids=test_ids,
        targets=test_y,
        predictions=test_predictions,
        probabilities=test_probabilities,
        decision_thresholds=thresholds,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
