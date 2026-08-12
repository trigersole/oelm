"""Small multi-task temporal convolutional model with CORN ordinal heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    blendshapes: int = 52
    channels: int = 96
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.20
    tasks: int = 4
    ordinal_levels: int = 4

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["dilations"] = list(self.dilations)
        return output


@dataclass(frozen=True)
class LSTMModelConfig:
    blendshapes: int = 52
    hidden_size: int = 64
    layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.20
    tasks: int = 4
    ordinal_levels: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class MultiTaskTCNCORN(nn.Module):
    """Input shape: [batch, time, 2 * features] (values + first differences)."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        input_features = config.blendshapes * 2
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_features, config.channels, kernel_size=1),
            nn.GroupNorm(8, config.channels),
            nn.GELU(),
        )
        self.temporal_blocks = nn.Sequential(
            *[
                ResidualTemporalBlock(
                    config.channels, config.kernel_size, dilation, config.dropout
                )
                for dilation in config.dilations
            ]
        )
        self.attention = nn.Conv1d(config.channels, 1, kernel_size=1)
        thresholds = config.ordinal_levels - 1
        self.heads = nn.ModuleList(
            [nn.Linear(config.channels, thresholds) for _ in range(config.tasks)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs.transpose(1, 2)
        features = self.temporal_blocks(self.input_projection(features))
        weights = torch.softmax(self.attention(features), dim=-1)
        pooled = torch.sum(features * weights, dim=-1)
        return torch.stack([head(pooled) for head in self.heads], dim=1)


class MultiTaskLSTMCORN(nn.Module):
    """BiLSTM challenger using the same values-plus-differences inputs."""

    def __init__(self, config: LSTMModelConfig = LSTMModelConfig()):
        super().__init__()
        self.config = config
        input_features = config.blendshapes * 2
        self.input_norm = nn.LayerNorm(input_features)
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=config.hidden_size,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        output_features = config.hidden_size * (2 if config.bidirectional else 1)
        self.attention = nn.Linear(output_features, 1)
        thresholds = config.ordinal_levels - 1
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(output_features),
                    nn.Dropout(config.dropout),
                    nn.Linear(output_features, thresholds),
                )
                for _ in range(config.tasks)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(self.input_norm(inputs))
        weights = torch.softmax(self.attention(sequence), dim=1)
        pooled = torch.sum(sequence * weights, dim=1)
        return torch.stack([head(pooled) for head in self.heads], dim=1)


def corn_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """CORN conditional ordinal loss for logits [B, tasks, levels-1]."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("Expected logits [B,T,K-1] and targets [B,T]")
    losses: list[torch.Tensor] = []
    thresholds = logits.shape[-1]
    for task in range(logits.shape[1]):
        task_target = targets[:, task]
        sample_weights = (
            class_weights[task, task_target]
            if class_weights is not None
            else torch.ones_like(task_target, dtype=logits.dtype)
        )
        numerator = logits.new_zeros(())
        denominator = logits.new_zeros(())
        for threshold in range(thresholds):
            active = torch.ones_like(task_target, dtype=torch.bool)
            if threshold > 0:
                active = task_target > (threshold - 1)
            if not torch.any(active):
                continue
            binary_target = (task_target[active] > threshold).to(logits.dtype)
            binary_loss = F.binary_cross_entropy_with_logits(
                logits[active, task, threshold], binary_target, reduction="none"
            )
            weights = sample_weights[active]
            numerator = numerator + torch.sum(binary_loss * weights)
            denominator = denominator + torch.sum(weights)
        losses.append(numerator / denominator.clamp_min(1e-8))
    return torch.stack(losses).mean()


def corn_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert conditional CORN logits to valid four-class probabilities."""
    conditional = torch.sigmoid(logits)
    survival = torch.cumprod(conditional, dim=-1)
    probabilities = [1.0 - survival[..., 0]]
    for level in range(1, survival.shape[-1]):
        probabilities.append(survival[..., level - 1] - survival[..., level])
    probabilities.append(survival[..., -1])
    return torch.stack(probabilities, dim=-1).clamp_min(0.0)
