"""Benchmark conventional four-level classifiers on combined DAiSEE features.

Each affective target gets an independent class-balanced classifier for the
original labels 0, 1, 2, and 3. Hyperparameters are fixed before test scoring;
the validation split is reported but is not used to tune on the test set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from ml.evaluate_xgboost_baseline import aggregate_sequences
    from ml.train_tcn_corn import LABEL_NAMES, calculate_metrics
except ModuleNotFoundError:
    from evaluate_xgboost_baseline import aggregate_sequences
    from train_tcn_corn import LABEL_NAMES, calculate_metrics


MODEL_NAMES = ("logistic_regression", "linear_svm", "random_forest", "extra_trees")
LEVELS = 4


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
        return data["X"], data["y"].astype(np.int64), data["clip_ids"]


def make_estimator(name: str, seed: int, trees: int, threads: int):
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
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
                tol=1e-3,
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


def class_probabilities(estimator, matrix: np.ndarray) -> np.ndarray:
    """Return an N x 4 probability-like matrix aligned to levels 0..3."""
    if hasattr(estimator, "predict_proba"):
        observed = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    else:
        observed = softmax(
            np.asarray(estimator.decision_function(matrix), dtype=np.float64), axis=1
        )
    aligned = np.zeros((len(matrix), LEVELS), dtype=np.float32)
    for source_column, label in enumerate(estimator.classes_):
        aligned[:, int(label)] = observed[:, source_column]
    return aligned


def benchmark(estimators: list, matrix: np.ndarray, repetitions: int) -> dict[str, float]:
    sample = matrix[:30]
    for _ in range(5):
        for estimator in estimators:
            estimator.predict(sample)
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        for estimator in estimators:
            estimator.predict(sample)
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
        validation_probabilities = np.empty(
            (len(validation_x), len(LABEL_NAMES), LEVELS), dtype=np.float32
        )
        test_probabilities = np.empty(
            (len(test_x), len(LABEL_NAMES), LEVELS), dtype=np.float32
        )
        validation_predictions = np.empty_like(validation_y)
        test_predictions = np.empty_like(test_y)
        task_seconds: dict[str, float] = {}
        for task, label_name in enumerate(LABEL_NAMES):
            task_started = time.perf_counter()
            estimator = make_estimator(
                model_name, args.seed + task, args.trees, args.threads
            )
            estimator.fit(train_x, train_y[:, task])
            validation_predictions[:, task] = estimator.predict(validation_x)
            test_predictions[:, task] = estimator.predict(test_x)
            validation_probabilities[:, task] = class_probabilities(
                estimator, validation_x
            )
            test_probabilities[:, task] = class_probabilities(estimator, test_x)
            estimators.append(estimator)
            task_seconds[label_name] = time.perf_counter() - task_started
            print(
                f"{model_name}: fitted {label_name} in "
                f"{task_seconds[label_name]:.2f}s",
                flush=True,
            )

        validation_metrics = calculate_metrics(
            validation_y, validation_predictions, levels=LEVELS
        )
        test_metrics = calculate_metrics(test_y, test_predictions, levels=LEVELS)
        inference = benchmark(estimators, test_x, args.benchmark_repetitions)
        final_estimator = estimators[0]
        if hasattr(final_estimator, "steps"):
            family = type(final_estimator.steps[-1][1]).__name__
        else:
            family = type(final_estimator).__name__
        report = {
            "model": model_name,
            "model_family": family,
            "seed": args.seed,
            "target_definition": "Original DAiSEE levels 0, 1, 2, and 3",
            "training_balance": "Class-balanced estimator loss per label",
            "model_selection": "Fixed hyperparameters; validation reported separately",
            "raw_feature_count": int(raw_train.shape[-1]),
            "aggregate_feature_count": int(train_x.shape[-1]),
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
        output_dir = args.output_root / f"model-classical-ordinal-{model_name}-seed{args.seed}"
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
        )
        print(
            json.dumps(
                {
                    "model": model_name,
                    "validation": {
                        "macro_f1": validation_metrics["mean_macro_f1"],
                        "balanced_accuracy": validation_metrics["mean_balanced_accuracy"],
                        "kappa": validation_metrics["mean_quadratic_weighted_kappa"],
                    },
                    "test": {
                        "macro_f1": test_metrics["mean_macro_f1"],
                        "balanced_accuracy": test_metrics["mean_balanced_accuracy"],
                        "mae": test_metrics["mean_mae"],
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
