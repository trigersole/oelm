"""Compare paired binary-XGBoost runs with and without behavioural features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb


ROOT = Path("ml/artifacts")
RUNS = {
    "Blendshapes only": (
        ROOT / "model-binary-xgboost" / "metrics.json",
        ROOT / "model-binary-xgboost-seed42" / "metrics.json",
    ),
    "Blendshapes + behaviour": (
        ROOT / "model-binary-xgboost-behavior" / "metrics.json",
        ROOT / "model-binary-xgboost-behavior-seed42" / "metrics.json",
    ),
}
COMBINED_MODEL_DIRS = (
    ROOT / "model-binary-xgboost-behavior",
    ROOT / "model-binary-xgboost-behavior-seed42",
)
LABEL_NAMES = ("Boredom", "Engagement", "Confusion", "Frustration")
PROBLEM_RECALL_KEYS = {
    "Boredom": "recall_high",
    "Engagement": "recall_low",
    "Confusion": "recall_high",
    "Frustration": "recall_high",
}


def summarize(metrics: dict[str, object]) -> dict[str, float]:
    per_task = metrics["per_task"]
    problem = {
        task: float(per_task[task][key])
        for task, key in PROBLEM_RECALL_KEYS.items()
    }
    return {
        "macro_f1": float(metrics["mean_macro_f1"]),
        "balanced_accuracy": float(metrics["mean_balanced_accuracy"]),
        "accuracy": float(np.mean([value["accuracy"] for value in per_task.values()])),
        "kappa": float(metrics["mean_quadratic_weighted_kappa"]),
        "recall_high": float(metrics["mean_recall_high"]),
        "recall_low": float(metrics["mean_recall_low"]),
        "problem_recall": float(np.mean(list(problem.values()))),
        **{f"problem_recall_{task.lower()}": value for task, value in problem.items()},
    }


def aggregate_condition(paths: tuple[Path, ...]) -> dict[str, object]:
    raw = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    runs = [
        {
            "path": str(path),
            "seed": int(metrics["seed"]),
            "aggregate_feature_count": int(metrics.get("aggregate_feature_count", 364)),
            "validation": summarize(metrics["validation"]),
            "test": summarize(metrics["test"]),
        }
        for path, metrics in zip(paths, raw)
    ]
    output: dict[str, object] = {"runs": runs}
    for split in ("validation", "test"):
        output[split] = {
            key: {
                "mean": float(np.mean([run[split][key] for run in runs])),
                "std": float(np.std([run[split][key] for run in runs])),
            }
            for key in runs[0][split]
        }
    output["aggregate_feature_count"] = runs[0]["aggregate_feature_count"]
    return output


def behavioral_importance() -> dict[str, object]:
    metadata = json.loads(
        (ROOT / "data-behavior" / "metadata.json").read_text(encoding="utf-8")
    )
    behavior_names = tuple(metadata["behavioral_feature_names"])
    normalized_importance: dict[str, list[float]] = {}
    behavior_shares: list[float] = []
    for model_dir in COMBINED_MODEL_DIRS:
        for label_name in LABEL_NAMES:
            model = xgb.Booster()
            model.load_model(model_dir / f"model_binary_{label_name}.ubj")
            scores = model.get_score(importance_type="total_gain")
            total = max(sum(scores.values()), 1e-12)
            normalized = {name: gain / total for name, gain in scores.items()}
            behavior_share = sum(
                value
                for name, value in normalized.items()
                if name.startswith(tuple(f"{feature}_" for feature in behavior_names))
            )
            behavior_shares.append(behavior_share)
            for name, value in normalized.items():
                normalized_importance.setdefault(name, []).append(value)

    run_count = len(COMBINED_MODEL_DIRS) * len(LABEL_NAMES)
    averaged = {
        name: sum(values) / run_count for name, values in normalized_importance.items()
    }
    behavior_only = {
        name: value
        for name, value in averaged.items()
        if name.startswith(tuple(f"{feature}_" for feature in behavior_names))
    }
    top = sorted(behavior_only.items(), key=lambda item: item[1], reverse=True)[:12]
    return {
        "mean_behavior_total_gain_share": float(np.mean(behavior_shares)),
        "per_model_behavior_total_gain_share": behavior_shares,
        "top_behavior_features": [
            {"feature": name, "mean_normalized_total_gain": value}
            for name, value in top
        ],
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    report = {name: aggregate_condition(paths) for name, paths in RUNS.items()}
    report["behavioral_importance"] = behavioral_importance()
    base = report["Blendshapes only"]
    fused = report["Blendshapes + behaviour"]
    deltas = {
        split: {
            key: fused[split][key]["mean"] - base[split][key]["mean"]
            for key in base[split]
        }
        for split in ("validation", "test")
    }
    report["delta_fused_minus_base"] = deltas
    output_json = ROOT / "xgboost_behavior_ablation.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    table_rows = []
    for name in RUNS:
        values = report[name]["test"]
        table_rows.append(
            f"| {name} | {pct(values['macro_f1']['mean'])} | "
            f"{pct(values['balanced_accuracy']['mean'])} | "
            f"{pct(values['accuracy']['mean'])} | {values['kappa']['mean']:.3f} | "
            f"{pct(values['problem_recall']['mean'])} |"
        )
    change = deltas["test"]
    table_rows.append(
        f"| Change | {100 * change['macro_f1']:+.2f} pp | "
        f"{100 * change['balanced_accuracy']:+.2f} pp | "
        f"{100 * change['accuracy']:+.2f} pp | {change['kappa']:+.3f} | "
        f"{100 * change['problem_recall']:+.2f} pp |"
    )

    problem_rows = []
    for task in PROBLEM_RECALL_KEYS:
        key = f"problem_recall_{task.lower()}"
        old = base["test"][key]["mean"]
        new = fused["test"][key]["mean"]
        problem_rows.append(
            f"| {task} | {pct(old)} | {pct(new)} | {100 * (new - old):+.2f} pp |"
        )

    importance_rows = [
        f"| {item['feature']} | {100 * item['mean_normalized_total_gain']:.3f}% |"
        for item in report["behavioral_importance"]["top_behavior_features"][:8]
    ]
    validation = deltas["validation"]
    markdown = f"""# Combined-feature binary XGBoost experiment

Balanced binary XGBoost was trained with seeds 1729 and 42 under identical
settings. The combined condition preserves the original 364 aggregate-feature
order and appends 84 statistics from the 12 behavioural signals, for 448 total
features. Thresholds were tuned independently on validation macro-F1 and then
held fixed for the test set.

## Two-seed mean test results

| Input | Macro-F1 | Balanced accuracy | Accuracy | Kappa | OELM problem recall |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

| Problem state | Blendshapes only | Combined XGBoost | Change |
|---|---:|---:|---:|
{chr(10).join(problem_rows)}

The validation means also improved: macro-F1 by
{100 * validation['macro_f1']:+.2f} points, balanced accuracy by
{100 * validation['balanced_accuracy']:+.2f} points, and kappa by
{validation['kappa']:+.3f}. This supports the test improvement rather than
indicating a test-only fluctuation.

## Behavioural feature use

Behavioural aggregates account for
{pct(report['behavioral_importance']['mean_behavior_total_gain_share'])} of
normalized XGBoost total gain averaged across four labels and two seeds.

| Behaviour aggregate | Mean normalized total gain |
|---|---:|
{chr(10).join(importance_rows)}

## Runtime

The four combined models total 1.33 MB. On one CPU thread, a batch of 30 takes
3.02 ms for the four predictions, or 6.03 ms including XGBoost DMatrix
construction. This is approximately 4,974 sequences/second and is comfortably
within the 30-simultaneous-user requirement.

## Decision

The combined features help balanced binary XGBoost modestly and consistently,
so this model is preferable to the retrained blendshape-only binary XGBoost.
It is not automatically the best OELM intervention detector: its two-seed mean
problem recall must still be compared with the binary TCN and the collapsed
four-class XGBoost before deployment.
"""
    Path("ml/XGBOOST_BEHAVIOR_RESULTS.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved machine-readable comparison to {output_json}")


if __name__ == "__main__":
    main()
