"""Summarize paired binary-TCN runs with and without behavioural features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("ml/artifacts")
RUNS = {
    "Blendshapes only": (
        ROOT / "model-binary-tcn" / "metrics.json",
        ROOT / "model-binary-tcn-seed42" / "metrics.json",
    ),
    "Blendshapes + behaviour": (
        ROOT / "model-binary-tcn-behavior" / "metrics.json",
        ROOT / "model-binary-tcn-behavior-seed42" / "metrics.json",
    ),
}
PROBLEM_RECALL_KEYS = {
    "Boredom": "recall_high",
    "Engagement": "recall_low",
    "Confusion": "recall_high",
    "Frustration": "recall_high",
}


def summarize_split(metrics: dict[str, object]) -> dict[str, float]:
    per_task = metrics["per_task"]
    problem_recalls = {
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
        "problem_recall": float(np.mean(list(problem_recalls.values()))),
        **{f"problem_recall_{task.lower()}": value for task, value in problem_recalls.items()},
    }


def aggregate_runs(paths: tuple[Path, ...]) -> dict[str, object]:
    raw = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    runs = []
    for path, metrics in zip(paths, raw):
        runs.append({
            "path": str(path),
            "seed": int(metrics["seed"]),
            "parameter_count": int(metrics["parameter_count"]),
            "validation": summarize_split(metrics["best_validation"]),
            "test": summarize_split(metrics["test"]),
        })
    output: dict[str, object] = {"runs": runs}
    for split in ("validation", "test"):
        keys = runs[0][split].keys()
        output[split] = {
            key: {
                "mean": float(np.mean([run[split][key] for run in runs])),
                "std": float(np.std([run[split][key] for run in runs], ddof=0)),
            }
            for key in keys
        }
    output["parameter_count"] = int(runs[0]["parameter_count"])
    return output


def percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    report = {name: aggregate_runs(paths) for name, paths in RUNS.items()}
    base = report["Blendshapes only"]
    fused = report["Blendshapes + behaviour"]
    deltas = {}
    for split in ("validation", "test"):
        deltas[split] = {
            key: fused[split][key]["mean"] - base[split][key]["mean"]
            for key in base[split]
        }
    report["delta_fused_minus_base"] = deltas
    output_json = ROOT / "behavior_ablation.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    for name in RUNS:
        values = report[name]["test"]
        rows.append(
            f"| {name} | {percentage(values['macro_f1']['mean'])} | "
            f"{percentage(values['balanced_accuracy']['mean'])} | "
            f"{percentage(values['accuracy']['mean'])} | "
            f"{values['kappa']['mean']:.3f} | "
            f"{percentage(values['problem_recall']['mean'])} |"
        )
    delta = deltas["test"]
    rows.append(
        f"| Change | {100 * delta['macro_f1']:+.2f} pp | "
        f"{100 * delta['balanced_accuracy']:+.2f} pp | "
        f"{100 * delta['accuracy']:+.2f} pp | {delta['kappa']:+.3f} | "
        f"{100 * delta['problem_recall']:+.2f} pp |"
    )

    problem_rows = []
    for task in PROBLEM_RECALL_KEYS:
        key = f"problem_recall_{task.lower()}"
        base_value = base["test"][key]["mean"]
        fused_value = fused["test"][key]["mean"]
        problem_rows.append(
            f"| {task} | {percentage(base_value)} | {percentage(fused_value)} | "
            f"{100 * (fused_value - base_value):+.2f} pp |"
        )

    validation_delta = deltas["validation"]
    markdown = f"""# Behavioural-feature ablation

The binary TCN was trained twice per condition with seeds 1729 and 42. Both
conditions used the same participant-independent DAiSEE splits, 5 FPS timeline,
binary labels, class weights, augmentation, validation threshold tuning and
early stopping. Only the 12 behavioural inputs differ.

## Two-seed mean test results

| Input | Macro-F1 | Balanced accuracy | Accuracy | Kappa | OELM problem recall |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

OELM problem recall treats High boredom, Low engagement, High confusion and
High frustration as the intervention-positive states.

| Problem state | Blendshapes only | With behaviour | Change |
|---|---:|---:|---:|
{chr(10).join(problem_rows)}

## Validation check

The two-seed validation mean changed by
{100 * validation_delta['macro_f1']:+.2f} percentage points for macro-F1 and
{100 * validation_delta['balanced_accuracy']:+.2f} points for balanced
accuracy. Therefore the small test gains in ordinary accuracy, macro-F1 and
kappa are not supported by better validation performance.

## Extraction and runtime

- Train: 4,975/4,976 raw videos found; 99.953% frame detection.
- Validation: 1,536/1,536 videos found; 99.967% frame detection.
- Test: 1,577/1,577 videos found; 99.932% frame detection.
- Parameters: 178,373 blendshape-only versus 180,677 fused (+1.29%).
- Fused CPU inference, one thread: 3.227 ms median for one sequence and
  23.411 ms for a batch of 30 (1,281 sequences/second).

## Decision

Do not replace the blendshape-only binary TCN with the current early-fusion
model. Behavioural fusion makes predictions more majority-oriented: ordinary
accuracy rises, but balanced accuracy and the recall of intervention-relevant
states fall. The largest losses are Low engagement and High confusion.

The extracted features remain useful for a follow-up experiment using
clip-relative head/gaze normalization or a separate behavioural branch with
late fusion. Those changes should be selected on validation problem recall,
not test accuracy.
"""
    Path("ml/BEHAVIOR_RESULTS.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved machine-readable comparison to {output_json}")


if __name__ == "__main__":
    main()
