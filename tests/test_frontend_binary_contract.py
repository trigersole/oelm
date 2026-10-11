from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class FrontendBinaryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT_DIR / "src" / "app.js").read_text(encoding="utf-8")

    def test_frontend_declares_only_low_and_high(self) -> None:
        self.assertIn("const LEVEL_LABELS = ['Low','High'];", self.source)
        self.assertIn("const LEVEL_COUNT = LEVEL_LABELS.length;", self.source)
        self.assertNotIn("['Very Low','Low','High','Very High']", self.source)

    def test_frontend_records_binary_model_version(self) -> None:
        self.assertIn("const MODEL_VERSION  = 'xgb260-static-seed1729-v1';", self.source)

    def test_only_binary_predictions_reach_supabase(self) -> None:
        self.assertIn("function isBinaryPrediction(prediction)", self.source)
        self.assertIn("if (!isBinaryPrediction(prediction)) throw new Error", self.source)
        self.assertIn("/predict-binary`;", self.source)
        self.assertNotIn("${API_BASE_URL}/api/predict", self.source)
        self.assertIn("table: 'emotion_predictions'", self.source)
        self.assertIn("params: { on_conflict: 'request_id' }", self.source)
        self.assertIn("model_version: prediction.model_version || MODEL_VERSION", self.source)

    def test_frontend_uses_xgb260_static_aggregation(self) -> None:
        aggregation = (
            ROOT_DIR / "src" / "data" / "featureAggregation.js"
        ).read_text(encoding="utf-8")
        order = (ROOT_DIR / "src" / "data" / "featureOrder.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("aggregateStaticWindow(frames, blendshapeNames)", aggregation)
        self.assertIn("['mean', 'std', 'min', 'max', 'median']", aggregation)
        self.assertIn("out[cat.categoryName] = Number(cat.score);", self.source)
        self.assertNotIn("cat.score.toFixed", self.source)
        self.assertNotIn("_skew", order)
        self.assertNotIn("_kurt", order)
        self.assertEqual(order.count('"'), 520)

    def test_prediction_delivery_is_timeout_bounded_and_idempotent(self) -> None:
        required_fragments = [
            "const API_REQUEST_TIMEOUT_MS = 5_000;",
            "async function fetchWithTimeout",
            "async function fetchWithRetry",
            "const requestId = createRequestId();",
            "request_id: requestId",
            "delivery_status: 'pending'",
            "const RELIABILITY_DB_NAME = 'oelm-reliability';",
            "window.addEventListener('online'",
            "table: 'prediction_delivery_events'",
            "['participant_login', 'list_cohorts', 'overview'].includes(action)",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_small_chart_segments_are_left_unlabelled(self) -> None:
        self.assertNotIn("barLabelAnnotations", self.source)
        self.assertIn("return visibleMask?.[bi] && pct > 0 ? `${pct}%` : ''", self.source)
        self.assertIn("textangle: 0", self.source)
        self.assertIn("constraintext: 'inside'", self.source)
        self.assertIn("uniformtext: { minsize: 10, mode: 'hide' }", self.source)
        self.assertIn("annotations: lockAnnotations", self.source)
        self.assertNotIn("smallLabelAnnotations", self.source)
        self.assertIn("textfont: { color: '#0b1220', size: 11 }", self.source)

    def test_binary_level_count_drives_charts_and_overrides(self) -> None:
        required_fragments = [
            "Array(LEVEL_COUNT).fill(0)",
            "r.level >= LEVEL_COUNT",
            "i < LEVEL_COUNT",
            "row.level_percentages.length === LEVEL_COUNT",
            "Math.random() * LEVEL_COUNT",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_raw_blendshapes_are_batched_as_value_arrays(self) -> None:
        self.assertIn("const RAW_BLENDSHAPE_BATCH_SECONDS = 30;", self.source)
        self.assertIn("BLENDSHAPE_NAMES.map(name => Number(features[name] ?? 0))", self.source)
        self.assertIn("queueRawBlendshapeBatch(false)", self.source)
        self.assertIn("queueRawBlendshapeBatch(true)", self.source)
        self.assertIn("participantRequest('save_blendshape_batch'", self.source)
        self.assertIn("const res = await fetchWithRetry(ADMIN_API_URL", self.source)
        self.assertNotIn("CompressionStream", self.source)

    def test_participant_relogin_restores_binary_history(self) -> None:
        self.assertIn("adminRequest('participant_login'", self.source)
        self.assertIn("restorePredictionHistory(loginResult.predictions)", self.source)
        self.assertIn("Relogin detected. Resuming your previous session", self.source)
        self.assertIn("activeSegmentRef.current = resumePending", self.source)


if __name__ == "__main__":
    unittest.main()
