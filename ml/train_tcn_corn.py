"""Train and evaluate the OELM multi-task TCN-CORN model."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from ml.model import (
        LSTMModelConfig,
        ModelConfig,
        MultiTaskLSTMCORN,
        MultiTaskTCNCORN,
        corn_loss,
        corn_probabilities,
    )
except ModuleNotFoundError:
    from model import (
        LSTMModelConfig,
        ModelConfig,
        MultiTaskLSTMCORN,
        MultiTaskTCNCORN,
        corn_loss,
        corn_probabilities,
    )


LABEL_NAMES = ("Boredom", "Engagement", "Confusion", "Frustration")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_temporal_inputs(raw: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(raw)
    delta[:, 1:] = raw[:, 1:] - raw[:, :-1]
    return np.concatenate([raw, delta], axis=-1).astype(np.float32)


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        raw: np.ndarray,
        targets: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = False,
    ):
        inputs = make_temporal_inputs(raw)
        self.inputs = ((inputs - mean) / std).astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.inputs[index].copy()
        if self.augment:
            if np.random.random() < 0.50:
                sequence += np.random.normal(0.0, 0.015, sequence.shape).astype(np.float32)
            if np.random.random() < 0.35:
                frame_count = np.random.randint(1, 4)
                frame_indices = np.random.choice(len(sequence), frame_count, replace=False)
                sequence[frame_indices] = 0.0
            if np.random.random() < 0.25:
                sequence = np.roll(sequence, np.random.randint(-2, 3), axis=0).copy()
        return torch.from_numpy(sequence), torch.from_numpy(self.targets[index])


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["X"], data["y"], data["clip_ids"]


def compute_normalizer(train_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inputs = make_temporal_inputs(train_raw)
    mean = inputs.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = inputs.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-5)
    return mean, std


def compute_class_weights(
    targets: np.ndarray, levels: int = 4, exact_balance: bool = False
) -> np.ndarray:
    """Compute per-task weights from training labels only."""
    weights = np.ones((targets.shape[1], levels), dtype=np.float32)
    for task in range(targets.shape[1]):
        counts = np.bincount(targets[:, task], minlength=levels).astype(np.float64)
        nonzero = counts > 0
        if exact_balance:
            weights[task, nonzero] = len(targets) / (levels * counts[nonzero])
        else:
            weights[task, nonzero] = 1.0 / np.sqrt(counts[nonzero])
            weights[task, nonzero] /= weights[task, nonzero].mean()
            weights[task] = np.clip(weights[task], 0.25, 4.0)
    return weights


def macro_f1(targets: np.ndarray, predictions: np.ndarray, levels: int = 4) -> float:
    scores = []
    for level in range(levels):
        tp = np.sum((targets == level) & (predictions == level))
        fp = np.sum((targets != level) & (predictions == level))
        fn = np.sum((targets == level) & (predictions != level))
        denominator = 2 * tp + fp + fn
        scores.append(float(2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def balanced_accuracy(targets: np.ndarray, predictions: np.ndarray, levels: int = 4) -> float:
    recalls = []
    for level in range(levels):
        mask = targets == level
        if np.any(mask):
            recalls.append(float(np.mean(predictions[mask] == level)))
    return float(np.mean(recalls)) if recalls else 0.0


def quadratic_weighted_kappa(targets: np.ndarray, predictions: np.ndarray, levels: int = 4) -> float:
    observed = np.zeros((levels, levels), dtype=np.float64)
    for truth, prediction in zip(targets, predictions):
        observed[int(truth), int(prediction)] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(observed.sum(), 1)
    indices = np.arange(levels)
    weights = ((indices[:, None] - indices[None, :]) ** 2) / ((levels - 1) ** 2)
    denominator = np.sum(weights * expected)
    return float(1.0 - np.sum(weights * observed) / denominator) if denominator else 0.0


def calculate_metrics(
    targets: np.ndarray, predictions: np.ndarray, levels: int = 4
) -> dict[str, object]:
    per_task: dict[str, dict[str, float]] = {}
    for task, name in enumerate(LABEL_NAMES):
        truth, predicted = targets[:, task], predictions[:, task]
        task_metrics = {
            "macro_f1": macro_f1(truth, predicted, levels),
            "balanced_accuracy": balanced_accuracy(truth, predicted, levels),
            "mae": float(np.mean(np.abs(truth - predicted))),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(truth, predicted, levels),
            "accuracy": float(np.mean(truth == predicted)),
        }
        if levels == 2:
            tp = int(np.sum((truth == 1) & (predicted == 1)))
            tn = int(np.sum((truth == 0) & (predicted == 0)))
            fp = int(np.sum((truth == 0) & (predicted == 1)))
            fn = int(np.sum((truth == 1) & (predicted == 0)))
            task_metrics.update({
                "precision_high": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall_high": float(tp / (tp + fn)) if tp + fn else 0.0,
                "recall_low": float(tn / (tn + fp)) if tn + fp else 0.0,
                "support_low": int(np.sum(truth == 0)),
                "support_high": int(np.sum(truth == 1)),
            })
        per_task[name] = task_metrics
    output = {
        "per_task": per_task,
        "mean_macro_f1": float(np.mean([value["macro_f1"] for value in per_task.values()])),
        "mean_balanced_accuracy": float(np.mean([value["balanced_accuracy"] for value in per_task.values()])),
        "mean_mae": float(np.mean([value["mae"] for value in per_task.values()])),
        "mean_quadratic_weighted_kappa": float(np.mean([
            value["quadratic_weighted_kappa"] for value in per_task.values()
        ])),
    }
    if levels == 2:
        output.update({
            "mean_precision_high": float(np.mean([
                value["precision_high"] for value in per_task.values()
            ])),
            "mean_recall_high": float(np.mean([
                value["recall_high"] for value in per_task.values()
            ])),
            "mean_recall_low": float(np.mean([
                value["recall_low"] for value in per_task.values()
            ])),
        })
    return output


def tune_binary_thresholds(
    targets: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Choose per-task thresholds on validation macro-F1 only."""
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    candidates = np.linspace(0.05, 0.95, 91)
    for task in range(targets.shape[1]):
        scores = np.asarray([
            macro_f1(targets[:, task], probabilities[:, task, 1] >= threshold, levels=2)
            for threshold in candidates
        ])
        best_score = scores.max()
        best = candidates[np.isclose(scores, best_score)]
        thresholds[task] = best[np.argmin(np.abs(best - 0.5))]
    return thresholds


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    levels: int = 4,
    thresholds: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_targets, all_predictions, all_probabilities = [], [], []
    for inputs, targets in loader:
        logits = model(inputs.to(device))
        probabilities = corn_probabilities(logits)
        if levels == 2 and thresholds is not None:
            threshold_tensor = torch.as_tensor(thresholds, device=probabilities.device)
            predictions = (probabilities[..., 1] >= threshold_tensor).long()
        else:
            predictions = probabilities.argmax(dim=-1)
        all_targets.append(targets.numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
    target_array = np.concatenate(all_targets)
    prediction_array = np.concatenate(all_predictions)
    probability_array = np.concatenate(all_probabilities)
    return (
        calculate_metrics(target_array, prediction_array, levels),
        target_array,
        prediction_array,
        probability_array,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/artifacts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/model"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--architecture", choices=("tcn", "lstm"), default="tcn")
    parser.add_argument("--target-mode", choices=("ordinal4", "binary"), default="ordinal4")
    parser.add_argument("--lstm-hidden-size", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-clips", type=int, default=None, help="Smoke-test limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_ids = load_split(args.data_dir / "train.npz")
    validation_x, validation_y, validation_ids = load_split(args.data_dir / "validation.npz")
    test_x, test_y, test_ids = load_split(args.data_dir / "test.npz")
    if train_x.ndim != 3:
        raise ValueError(f"Expected train X [clips, frames, features], got {train_x.shape}")
    expected_shape = train_x.shape[1:]
    for split_name, values in (("validation", validation_x), ("test", test_x)):
        if values.shape[1:] != expected_shape:
            raise ValueError(
                f"{split_name} shape {values.shape[1:]} differs from train {expected_shape}"
            )
    raw_features = int(train_x.shape[-1])
    levels = 2 if args.target_mode == "binary" else 4
    if args.target_mode == "binary":
        train_y = (train_y >= 2).astype(np.int64)
        validation_y = (validation_y >= 2).astype(np.int64)
        test_y = (test_y >= 2).astype(np.int64)
    if args.max_train_clips:
        limit = min(args.max_train_clips, len(train_x))
        train_x, train_y, train_ids = train_x[:limit], train_y[:limit], train_ids[:limit]

    mean, std = compute_normalizer(train_x)
    class_weights_np = compute_class_weights(
        train_y, levels=levels, exact_balance=args.target_mode == "binary"
    )
    class_weights = torch.from_numpy(class_weights_np).to(device)
    np.savez(output_dir / "normalization.npz", mean=mean, std=std)

    train_dataset = SequenceDataset(train_x, train_y, mean, std, augment=True)
    validation_dataset = SequenceDataset(validation_x, validation_y, mean, std)
    test_dataset = SequenceDataset(test_x, test_y, mean, std)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    if args.architecture == "tcn":
        config = ModelConfig(
            blendshapes=raw_features,
            channels=args.channels,
            dropout=args.dropout,
            ordinal_levels=levels,
        )
        model = MultiTaskTCNCORN(config).to(device)
    else:
        config = LSTMModelConfig(
            blendshapes=raw_features,
            hidden_size=args.lstm_hidden_size,
            layers=args.lstm_layers,
            dropout=args.dropout,
            ordinal_levels=levels,
        )
        model = MultiTaskLSTMCORN(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_score = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = corn_loss(logits, targets, class_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(inputs)
            seen += len(inputs)
        scheduler.step()
        metrics, validation_targets, _, validation_probabilities = evaluate(
            model, validation_loader, device, levels=levels
        )
        decision_thresholds = None
        if levels == 2:
            decision_thresholds = tune_binary_thresholds(
                validation_targets, validation_probabilities
            )
            validation_predictions = (
                validation_probabilities[..., 1] >= decision_thresholds[None, :]
            ).astype(np.int64)
            metrics = calculate_metrics(
                validation_targets, validation_predictions, levels=2
            )
        score = float(metrics["mean_macro_f1"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch:03d} loss={epoch_record['train_loss']:.4f} "
            f"val_macro_f1={score:.4f} val_mae={metrics['mean_mae']:.4f}"
        )
        if score > best_score + 1e-5:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": args.architecture,
                    "target_mode": args.target_mode,
                    "model_state": model.state_dict(),
                    "model_config": config.to_dict(),
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "label_names": LABEL_NAMES,
                    "class_weights": class_weights_np,
                    "seed": args.seed,
                    "validation_metrics": metrics,
                    "decision_thresholds": decision_thresholds,
                },
                output_dir / "best_checkpoint.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    decision_thresholds = checkpoint.get("decision_thresholds")
    validation_metrics, _, _, _ = evaluate(
        model,
        validation_loader,
        device,
        levels=levels,
        thresholds=decision_thresholds,
    )
    test_metrics, test_targets, test_predictions, test_probabilities = evaluate(
        model, test_loader, device, levels=levels, thresholds=decision_thresholds
    )

    model.eval()
    example = torch.zeros(
        1, int(train_x.shape[1]), raw_features * 2, device=device
    )
    traced = torch.jit.trace(model, example)
    if args.target_mode == "binary":
        model_filename = f"model_{args.architecture}_binary.ts"
    else:
        model_filename = "model_tcn_corn.ts" if args.architecture == "tcn" else "model_lstm_corn.ts"
    traced.save(str(output_dir / model_filename))
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        clip_ids=test_ids,
        targets=test_targets,
        predictions=test_predictions,
        probabilities=test_probabilities,
        decision_thresholds=decision_thresholds,
    )
    summary = {
        "seed": args.seed,
        "architecture": args.architecture,
        "target_mode": args.target_mode,
        "target_definition": "Low=levels 0/1; High=levels 2/3" if levels == 2 else "Four ordinal levels 0/1/2/3",
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "train_clips": len(train_dataset),
        "validation_clips": len(validation_dataset),
        "test_clips": len(test_dataset),
        "model_config": config.to_dict(),
        "class_weights": class_weights_np.tolist(),
        "decision_thresholds": decision_thresholds.tolist() if decision_thresholds is not None else None,
        "best_validation": validation_metrics,
        "test": test_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"best_validation": validation_metrics, "test": test_metrics}, indent=2))
    print(f"Saved model and metrics to {output_dir}")


if __name__ == "__main__":
    main()
