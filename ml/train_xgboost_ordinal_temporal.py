"""Train four-level XGBoost variants with richer temporal clip descriptors."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import f_classif

try:
    from ml.temporal_features import temporal_aggregate_sequences
    from ml.train_tcn_corn import LABEL_NAMES, calculate_metrics
except ModuleNotFoundError:
    from temporal_features import temporal_aggregate_sequences
    from train_tcn_corn import LABEL_NAMES, calculate_metrics


VARIANTS = ("baseline_combined", "temporal_all", "temporal_select_512")
LEVELS = 4
BASE_STATISTIC_NAMES = ("mean", "std", "min", "max", "median", "skew", "kurt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/artifacts/data-behavior"))
    parser.add_argument("--feature-order", type=Path, default=Path("feature_order.pkl"))
    parser.add_argument("--output-root", type=Path, default=Path("ml/artifacts"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--early-stopping", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benchmark-repetitions", type=int, default=100)
    return parser.parse_args()


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["X"], data["y"].astype(np.int64), data["clip_ids"]


def balanced_weights(targets: np.ndarray) -> np.ndarray:
    counts = np.bincount(targets, minlength=LEVELS).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"A training class is missing: {counts.tolist()}")
    class_weights = len(targets) / (LEVELS * counts)
    return class_weights[targets].astype(np.float32)


def select_features(matrix: np.ndarray, targets: np.ndarray, count: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, _ = f_classif(matrix, targets)
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float64).max)
    selected = np.argpartition(scores, -count)[-count:]
    return selected[np.argsort(scores[selected])[::-1]]


def benchmark(
    models: list[xgb.Booster],
    matrix: np.ndarray,
    selections: list[np.ndarray],
    feature_orders: list[list[str]],
    repetitions: int,
) -> dict[str, float]:
    batches = [
        xgb.DMatrix(matrix[:30, selection], feature_names=feature_order)
        for selection, feature_order in zip(selections, feature_orders)
    ]
    for _ in range(5):
        for model, batch in zip(models, batches):
            model.predict(batch, iteration_range=(0, model.best_iteration + 1))
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        for model, batch in zip(models, batches):
            model.predict(batch, iteration_range=(0, model.best_iteration + 1))
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
    original_order = list(joblib.load(args.feature_order))

    raw_train, train_y, _ = load_split(args.data_dir / "train.npz")
    raw_validation, validation_y, _ = load_split(args.data_dir / "validation.npz")
    raw_test, test_y, test_ids = load_split(args.data_dir / "test.npz")

    extraction_started = time.perf_counter()
    train_columns = temporal_aggregate_sequences(raw_train, metadata["feature_names"])
    validation_columns = temporal_aggregate_sequences(
        raw_validation, metadata["feature_names"]
    )
    test_columns = temporal_aggregate_sequences(raw_test, metadata["feature_names"])
    base_order = list(original_order)
    base_known = set(base_order)
    base_order.extend(
        name
        for feature in metadata["feature_names"]
        for statistic in BASE_STATISTIC_NAMES
        if (name := f"{feature}_{statistic}") not in base_known
    )
    temporal_order = list(base_order)
    temporal_known = set(temporal_order)
    temporal_order.extend(name for name in train_columns if name not in temporal_known)

    def build_matrix(columns: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
        output = np.column_stack([columns[name] for name in order]).astype(np.float32)
        if not np.all(np.isfinite(output)):
            raise ValueError("Temporal feature matrix contains NaN or infinity")
        return output

    baseline_train = build_matrix(train_columns, base_order)
    baseline_validation = build_matrix(validation_columns, base_order)
    baseline_test = build_matrix(test_columns, base_order)
    temporal_train = build_matrix(train_columns, temporal_order)
    temporal_validation = build_matrix(validation_columns, temporal_order)
    temporal_test = build_matrix(test_columns, temporal_order)
    extraction_seconds = time.perf_counter() - extraction_started
    print(
        f"features: baseline={baseline_train.shape[1]} temporal={temporal_train.shape[1]} "
        f"train={len(train_y)} validation={len(validation_y)} test={len(test_y)} "
        f"extraction={extraction_seconds:.2f}s",
        flush=True,
    )

    parameters = {
        "objective": "multi:softprob",
        "num_class": LEVELS,
        "eval_metric": "mlogloss",
        "eta": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.70,
        "lambda": 5.0,
        "alpha": 0.2,
        "tree_method": "hist",
        "seed": args.seed,
        "nthread": args.threads,
    }

    for variant in args.variants:
        started = time.perf_counter()
        if variant == "baseline_combined":
            train_x, validation_x, test_x = (
                baseline_train,
                baseline_validation,
                baseline_test,
            )
            candidate_order = base_order
        else:
            train_x, validation_x, test_x = (
                temporal_train,
                temporal_validation,
                temporal_test,
            )
            candidate_order = temporal_order

        output_dir = args.output_root / f"model-temporal-xgboost-{variant}-seed{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_probabilities = np.empty(
            (len(validation_y), len(LABEL_NAMES), LEVELS), dtype=np.float32
        )
        test_probabilities = np.empty(
            (len(test_y), len(LABEL_NAMES), LEVELS), dtype=np.float32
        )
        models: list[xgb.Booster] = []
        selections: list[np.ndarray] = []
        feature_orders: list[list[str]] = []
        best_iterations: dict[str, int] = {}
        selected_features: dict[str, list[str]] = {}
        task_training_seconds: dict[str, float] = {}

        for task, label_name in enumerate(LABEL_NAMES):
            task_started = time.perf_counter()
            if variant == "temporal_select_512":
                selection = select_features(train_x, train_y[:, task], 512)
            else:
                selection = np.arange(train_x.shape[1])
            order = [candidate_order[index] for index in selection]
            selected_features[label_name] = order
            train_matrix = xgb.DMatrix(
                train_x[:, selection],
                label=train_y[:, task],
                weight=balanced_weights(train_y[:, task]),
                feature_names=order,
            )
            validation_matrix = xgb.DMatrix(
                validation_x[:, selection],
                label=validation_y[:, task],
                feature_names=order,
            )
            test_matrix = xgb.DMatrix(test_x[:, selection], feature_names=order)
            model = xgb.train(
                {**parameters, "seed": args.seed + task},
                train_matrix,
                num_boost_round=args.rounds,
                evals=[(validation_matrix, "validation")],
                early_stopping_rounds=args.early_stopping,
                verbose_eval=False,
            )
            best_iterations[label_name] = int(model.best_iteration)
            model.save_model(output_dir / f"model_ordinal_{label_name}.ubj")
            validation_probabilities[:, task] = model.predict(
                validation_matrix, iteration_range=(0, model.best_iteration + 1)
            )
            test_probabilities[:, task] = model.predict(
                test_matrix, iteration_range=(0, model.best_iteration + 1)
            )
            models.append(model)
            selections.append(selection)
            feature_orders.append(order)
            task_training_seconds[label_name] = time.perf_counter() - task_started
            print(
                f"{variant}: {label_name} features={len(order)} "
                f"iteration={model.best_iteration} "
                f"seconds={task_training_seconds[label_name]:.2f}",
                flush=True,
            )

        validation_predictions = validation_probabilities.argmax(axis=-1)
        test_predictions = test_probabilities.argmax(axis=-1)
        validation_metrics = calculate_metrics(
            validation_y, validation_predictions, levels=LEVELS
        )
        test_metrics = calculate_metrics(test_y, test_predictions, levels=LEVELS)
        inference = benchmark(
            models,
            test_x,
            selections,
            feature_orders,
            args.benchmark_repetitions,
        )
        report = {
            "model": "Balanced four-level temporal XGBoost",
            "variant": variant,
            "seed": args.seed,
            "target_definition": "Original DAiSEE levels 0, 1, 2, and 3",
            "training_balance": "Each observed level contributes equal training loss per label",
            "model_selection": "Early stopping on validation multiclass log loss",
            "feature_selection": (
                "Top 512 per task by training-only ANOVA F score"
                if variant == "temporal_select_512"
                else "None"
            ),
            "raw_feature_count": int(raw_train.shape[-1]),
            "aggregate_feature_count": int(len(feature_orders[0])),
            "candidate_feature_count": int(train_x.shape[-1]),
            "feature_extraction_seconds": extraction_seconds,
            "task_training_seconds": task_training_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "best_iterations": best_iterations,
            "inference": inference,
            "parameters": parameters,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (output_dir / "selected_features.json").write_text(
            json.dumps(selected_features, indent=2), encoding="utf-8"
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
                    "variant": variant,
                    "validation": {
                        "macro_f1": validation_metrics["mean_macro_f1"],
                        "balanced_accuracy": validation_metrics["mean_balanced_accuracy"],
                        "accuracy": float(
                            np.mean(
                                [
                                    value["accuracy"]
                                    for value in validation_metrics["per_task"].values()
                                ]
                            )
                        ),
                        "kappa": validation_metrics["mean_quadratic_weighted_kappa"],
                    },
                    "test": {
                        "macro_f1": test_metrics["mean_macro_f1"],
                        "balanced_accuracy": test_metrics["mean_balanced_accuracy"],
                        "accuracy": float(
                            np.mean(
                                [value["accuracy"] for value in test_metrics["per_task"].values()]
                            )
                        ),
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
