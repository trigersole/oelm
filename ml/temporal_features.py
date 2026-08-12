"""Leakage-free clip-level temporal descriptors for fixed-length sequences."""

from __future__ import annotations

import numpy as np

from ml.evaluate_xgboost_baseline import aggregate_sequences


TEMPORAL_STATISTIC_NAMES = (
    "q10",
    "q25",
    "q75",
    "q90",
    "iqr",
    "range",
    "slope",
    "trend_r2",
    "first_last_delta",
    "diff_mean",
    "diff_std",
    "diff_abs_mean",
    "diff_abs_max",
    "accel_abs_mean",
    "autocorr_lag1",
    "autocorr_lag5",
    "motion_ratio",
    "spectral_entropy",
    "high_frequency_ratio",
)


def _autocorrelation(raw: np.ndarray, centered: np.ndarray, lag: int) -> np.ndarray:
    variance = np.mean(centered * centered, axis=1)
    covariance = np.mean(centered[:, :-lag] * centered[:, lag:], axis=1)
    return np.divide(
        covariance,
        variance,
        out=np.zeros_like(covariance),
        where=variance > 1e-12,
    )


def temporal_aggregate_sequences(
    raw: np.ndarray, feature_names: list[str]
) -> dict[str, np.ndarray]:
    """Return the original seven statistics plus motion/shape descriptors.

    The operation uses only frames within each clip. It does not read labels or
    combine information across train, validation, and test participants.
    """
    if raw.ndim != 3 or raw.shape[-1] != len(feature_names):
        raise ValueError("Expected raw shape [clips, frames, features]")
    if raw.shape[1] < 6:
        raise ValueError("At least six frames are required for lag-5 descriptors")

    values = np.asarray(raw, dtype=np.float32)
    columns = aggregate_sequences(values, feature_names)
    mean = values.mean(axis=1)
    centered = values - mean[:, None, :]
    variance_sum = np.sum(centered * centered, axis=1)

    quantiles = np.quantile(values, [0.10, 0.25, 0.75, 0.90], axis=1)
    diff = np.diff(values, axis=1)
    acceleration = np.diff(diff, axis=1)
    abs_diff = np.abs(diff)

    time_axis = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float32)
    time_energy = float(np.sum(time_axis * time_axis))
    trend_numerator = np.sum(centered * time_axis[None, :, None], axis=1)
    slope = trend_numerator / time_energy
    trend_r2 = np.divide(
        trend_numerator * trend_numerator,
        variance_sum * time_energy,
        out=np.zeros_like(trend_numerator),
        where=variance_sum > 1e-12,
    )

    scale = values.std(axis=1) + 1e-6
    motion_ratio = np.mean(abs_diff > (0.25 * scale[:, None, :]), axis=1)

    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    spectrum = spectrum[:, 1:, :]
    total_power = spectrum.sum(axis=1)
    normalized_power = np.divide(
        spectrum,
        total_power[:, None, :],
        out=np.zeros_like(spectrum),
        where=total_power[:, None, :] > 1e-12,
    )
    spectral_entropy = -np.sum(
        normalized_power * np.log(normalized_power + 1e-12), axis=1
    ) / np.log(max(spectrum.shape[1], 2))
    high_start = max(1, spectrum.shape[1] // 2)
    high_frequency_ratio = np.divide(
        spectrum[:, high_start:, :].sum(axis=1),
        total_power,
        out=np.zeros_like(total_power),
        where=total_power > 1e-12,
    )

    temporal_values = {
        "q10": quantiles[0],
        "q25": quantiles[1],
        "q75": quantiles[2],
        "q90": quantiles[3],
        "iqr": quantiles[2] - quantiles[1],
        "range": values.max(axis=1) - values.min(axis=1),
        "slope": slope,
        "trend_r2": trend_r2,
        "first_last_delta": values[:, -1] - values[:, 0],
        "diff_mean": diff.mean(axis=1),
        "diff_std": diff.std(axis=1),
        "diff_abs_mean": abs_diff.mean(axis=1),
        "diff_abs_max": abs_diff.max(axis=1),
        "accel_abs_mean": np.abs(acceleration).mean(axis=1),
        "autocorr_lag1": _autocorrelation(values, centered, 1),
        "autocorr_lag5": _autocorrelation(values, centered, 5),
        "motion_ratio": motion_ratio,
        "spectral_entropy": spectral_entropy,
        "high_frequency_ratio": high_frequency_ratio,
    }
    columns.update(
        {
            f"{feature}_{statistic}": temporal_values[statistic][:, feature_index]
            for feature_index, feature in enumerate(feature_names)
            for statistic in TEMPORAL_STATISTIC_NAMES
        }
    )
    return columns

