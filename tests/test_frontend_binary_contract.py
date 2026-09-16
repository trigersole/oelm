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
        self.assertIn("const MODEL_VERSION  = 'xgb-binary-v1';", self.source)

    def test_only_binary_predictions_reach_supabase(self) -> None:
        self.assertIn("function isBinaryPrediction(prediction)", self.source)
        self.assertIn("if (!isBinaryPrediction(prediction)) throw new Error", self.source)
        self.assertIn("/predict-binary`;", self.source)
        self.assertNotIn("${API_BASE_URL}/api/predict", self.source)
        self.assertIn("await sbRequest('emotion_predictions', 'POST', payload)", self.source)

    def test_small_chart_segments_are_left_unlabelled(self) -> None:
        self.assertIn("const barLabelAnnotations = labels.flatMap", self.source)
        self.assertIn("normalizedWidths[bi] < 0.3", self.source)
        self.assertIn("if (pct < 25) return []", self.source)
        self.assertIn("annotations: [...lockAnnotations, ...barLabelAnnotations]", self.source)
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


if __name__ == "__main__":
    unittest.main()
