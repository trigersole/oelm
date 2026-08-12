"""Small unit tests that do not require access to the DAiSEE CSV files."""

from __future__ import annotations

import unittest

import numpy as np

from ml.prepare_sequences import normalize_clip_id, resample_clip
from ml.extract_behavioral_features import interpolate_missing, rotation_to_euler
from ml.temporal_features import TEMPORAL_STATISTIC_NAMES, temporal_aggregate_sequences

try:
    import torch
    from ml.model import (
        LSTMModelConfig,
        ModelConfig,
        MultiTaskLSTMCORN,
        MultiTaskTCNCORN,
        corn_loss,
        corn_probabilities,
    )
    from ml.train_tcn_corn import calculate_metrics, tune_binary_thresholds
except (ImportError, OSError):
    torch = None


class PrepareSequencesTests(unittest.TestCase):
    def test_normalizes_avi_clip_id(self) -> None:
        self.assertEqual(normalize_clip_id(" 12345.avi "), "12345")
        self.assertEqual(normalize_clip_id("12345"), "12345")

    def test_resamples_missing_middle_frame(self) -> None:
        timestamps = np.asarray([0.0, 400.0])
        values = np.asarray([[0.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        result, observed = resample_clip(timestamps, values, frames=3, interval_ms=200.0)
        np.testing.assert_allclose(result[1], [0.5, 2.0])
        np.testing.assert_array_equal(observed, [1, 0, 1])

    def test_interpolates_behavior_when_face_is_temporarily_missing(self) -> None:
        values = np.zeros((3, 12), dtype=np.float32)
        values[0, 0], values[0, -1] = 1.0, 1.0
        values[2, 0], values[2, -1] = 3.0, 1.0
        result = interpolate_missing(values)
        self.assertAlmostEqual(float(result[1, 0]), 2.0)
        np.testing.assert_array_equal(result[:, -1], [1.0, 0.0, 1.0])

    def test_identity_rotation_has_zero_euler_angles(self) -> None:
        angles = rotation_to_euler(np.eye(4))
        np.testing.assert_allclose(angles, [0.0, 0.0, 0.0], atol=1e-7)

    def test_temporal_aggregates_capture_trend_without_nonfinite_values(self) -> None:
        frames = np.arange(10, dtype=np.float32)
        raw = np.stack([frames, np.ones_like(frames)], axis=1)[None, ...]
        columns = temporal_aggregate_sequences(raw, ["trend", "constant"])
        self.assertGreater(float(columns["trend_slope"][0]), 0.0)
        self.assertAlmostEqual(float(columns["constant_slope"][0]), 0.0)
        self.assertEqual(len(columns), 2 * (7 + len(TEMPORAL_STATISTIC_NAMES)))
        self.assertTrue(all(np.all(np.isfinite(value)) for value in columns.values()))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class OrdinalModelTests(unittest.TestCase):
    def test_temporal_models_have_matching_output_shape(self) -> None:
        inputs = torch.randn(2, 50, 104)
        self.assertEqual(tuple(MultiTaskTCNCORN()(inputs).shape), (2, 4, 3))
        self.assertEqual(tuple(MultiTaskLSTMCORN()(inputs).shape), (2, 4, 3))

    def test_corn_probabilities_are_valid(self) -> None:
        logits = torch.randn(8, 4, 3)
        probabilities = corn_probabilities(logits)
        self.assertEqual(tuple(probabilities.shape), (8, 4, 4))
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(8, 4))

    def test_corn_loss_is_finite(self) -> None:
        logits = torch.randn(8, 4, 3)
        targets = torch.randint(0, 4, (8, 4))
        self.assertTrue(torch.isfinite(corn_loss(logits, targets)))

    def test_binary_models_and_threshold_tuning(self) -> None:
        inputs = torch.randn(2, 50, 104)
        tcn_logits = MultiTaskTCNCORN(ModelConfig(ordinal_levels=2))(inputs)
        lstm_logits = MultiTaskLSTMCORN(LSTMModelConfig(ordinal_levels=2))(inputs)
        self.assertEqual(tuple(tcn_logits.shape), (2, 4, 1))
        self.assertEqual(tuple(lstm_logits.shape), (2, 4, 1))
        probabilities = corn_probabilities(tcn_logits).detach().numpy()
        targets = np.asarray([[0, 0, 1, 1], [1, 1, 0, 0]])
        thresholds = tune_binary_thresholds(targets, probabilities)
        self.assertEqual(tuple(thresholds.shape), (4,))
        predictions = (probabilities[..., 1] >= thresholds).astype(np.int64)
        metrics = calculate_metrics(targets, predictions, levels=2)
        self.assertIn("mean_recall_high", metrics)

    def test_temporal_models_accept_fused_feature_count(self) -> None:
        inputs = torch.randn(2, 50, 128)
        tcn = MultiTaskTCNCORN(ModelConfig(blendshapes=64))
        lstm = MultiTaskLSTMCORN(LSTMModelConfig(blendshapes=64))
        self.assertEqual(tuple(tcn(inputs).shape), (2, 4, 3))
        self.assertEqual(tuple(lstm(inputs).shape), (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
