"""Benchmark conventional binary classifiers on combined DAiSEE features.

Each affective target gets an independent balanced classifier. Decision
thresholds are selected on validation macro-F1 and then frozen for test
evaluation, matching the binary XGBoost/TCN protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from ml.evaluate_xgboost_baseline import aggregate_sequences
    from ml.train_tcn_corn import LABEL_NAMES, calculate_metrics, tune_binary_thresholds
except ModuleNotFoundError:
    from evaluate_xgboost_baseline import aggregate_sequences
    from train_tcn_corn import LABEL_NAMES, calculate_metrics, tune_binary_thresholds


MODEL_NAMES = ("logistic_regression", "linear_svm", "random_forest", "extra_trees")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/artifacts/data-behavior"))
    parser.add_argument("--feature-order", type=Path, default=Path("feature_order.pkl"))
    parser.add_argument("--output-root", type=Path, default=Path("ml/artifacts"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benchmark-repetitions", type=int, default=100)
    return parser.parse_args()


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["X"], (data["y"] >= 2).astype(np.int64), data["clip_ids"]


def make_estimator(name: str, seed: int, trees: int, threads: int):
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=seed,
            ),
        )
    if name == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=0.25,
                class_weight="balanced",
                dual="auto",
                max_iter=10000,
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=threads,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=threads,
            random_state=seed,
        )
    raise ValueError(name)


def high_scores(estimator, matrix: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(matrix)[:, 1], dtype=np.float32)
    return expit(np.asarray(estimator.decision_function(matrix), dtype=np.float64)).astype(
        np.float32
    )


def probability_tensor(scores: np.ndarray) -> np.ndarray:
    return np.stack((1.0 - scores, scores), axis=-1).astype(np.float32)


def benchmark(estimators: list, matrix: np.ndarray, repetitions: int) -> dict[str, float]:
    sample = matrix[:30]
    for _ in range(5):
        for estimator in estimators:
            high_scores(estimator, sample)
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        for estimator in estimators:
            high_scores(estimator, sample)
        durations.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(durations)
    return {
        "batch30_median_ms": float(np.median(values)),
        "batch30_p95_ms": float(np.percentile(values, 95)),
        "throughput_sequences_per_second": float(30000.0 / np.median(values)),
    }


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_order = list(joblib.load(args.feature_order))

    raw_train, train_y, _ = load_split(args.data_dir / "train.npz")
    raw_validation, validation_y, _ = load_split(args.data_dir / "validation.npz")
    raw_test, test_y, test_ids = load_split(args.data_dir / "test.npz")
    train_columns = aggregate_sequences(raw_train, metadata["feature_names"])
    existing = set(feature_order)
    feature_order.extend(name for name in train_columns if name not in existing)

    def matrix(raw: np.ndarray, columns: dict[str, np.ndarray] | None = None) -> np.ndarray:
        columns = columns or aggregate_sequences(raw, metadata["feature_names"])
        output = np.column_stack([columns[name] for name in feature_order]).astype(np.float32)
        if not np.all(np.isfinite(output)):
            raise ValueError("Classical feature matrix contains NaN or infinity")
        return output

    train_x = matrix(raw_train, train_columns)
    validation_x = matrix(raw_validation)
    test_x = matrix(raw_test)
    print(
        f"features={train_x.shape[1]} train={len(train_x)} "
        f"validation={len(validation_x)} test={len(test_x)}",
        flush=True,
    )

    for model_name in args.models:
        started = time.perf_counter()
        estimators = []
        validation_scores = np.empty((len(validation_x), len(LABEL_NAMES)), dtype=np.float32)
        test_scores = np.empty((len(test_x), len(LABEL_NAMES)), dtype=np.float32)
        task_seconds: dict[str, float] = {}
        for task, label_name in enumerate(LABEL_NAMES):
            task_started = time.perf_counter()
            estimator = make_estimator(
                model_name, args.seed + task, args.trees, args.threads
            )
            estimator.fit(train_x, train_y[:, task])
            validation_scores[:, task] = high_scores(estimator, validation_x)
            test_scores[:, task] = high_scores(estimator, test_x)
            estimators.append(estimator)
            task_seconds[label_name] = time.perf_counter() - task_started
            print(
                f"{model_name}: fitted {label_name} in "
                f"{task_seconds[label_name]:.2f}s",
                flush=True,
            )

        validation_probabilities = probability_tensor(validation_scores)
        test_probabilities = probability_tensor(test_scores)
        thresholds = tune_binary_thresholds(validation_y, validation_probabilities)
        validation_predictions = (
            validation_scores >= thresholds[None, :]
        ).astype(np.int64)
        test_predictions = (test_scores >= thresholds[None, :]).astype(np.int64)
        validation_metrics = calculate_metrics(
            validation_y, validation_predictions, levels=2
        )
        test_metrics = calculate_metrics(test_y, test_predictions, levels=2)
        inference = benchmark(estimators, test_x, args.benchmark_repetitions)
        report = {
            "model": model_name,
            "model_family": type(estimators[0]).__name__,
            "seed": args.seed,
            "target_definition": "Low=levels 0/1; High=levels 2/3",
            "training_balance": "Class-balanced estimator loss per label",
            "threshold_selection": "Per-label validation macro-F1",
            "raw_feature_count": int(raw_train.shape[-1]),
            "aggregate_feature_count": int(train_x.shape[-1]),
            "decision_thresholds": thresholds.tolist(),
            "task_training_seconds": task_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "inference": inference,
            "parameters": {
                "trees": args.trees if model_name in {"random_forest", "extra_trees"} else None,
                "threads": args.threads,
            },
            "validation": validation_metrics,
            "test": test_metrics,
        }
        output_dir = args.output_root / f"model-classical-{model_name}-seed{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (output_dir / "feature_order.json").write_text(
            json.dumps(feature_order, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            clip_ids=test_ids,
            targets=test_y,
            predictions=test_predictions,
            probabilities=test_probabilities,
            decision_thresholds=thresholds,
        )
        print(
            json.dumps(
                {
                    "model": model_name,
                    "validation": {
                        "macro_f1": validation_metrics["mean_macro_f1"],
                        "balanced_accuracy": validation_metrics["mean_balanced_accuracy"],
                    },
                    "test": {
                        "macro_f1": test_metrics["mean_macro_f1"],
                        "balanced_accuracy": test_metrics["mean_balanced_accuracy"],
                        "kappa": test_metrics["mean_quadratic_weighted_kappa"],
                    },
                    "inference": inference,
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
