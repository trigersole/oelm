from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class PredictionReliabilitySchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sql = (ROOT_DIR / "docs" / "supabase-prediction-reliability.sql").read_text(
            encoding="utf-8"
        )

    def test_prediction_request_ids_are_unique(self) -> None:
        self.assertIn("add column if not exists request_id text", self.sql)
        self.assertIn("emotion_predictions_request_id_key", self.sql)
        self.assertIn("on public.emotion_predictions (request_id)", self.sql)

    def test_delivery_audit_table_accepts_only_known_statuses(self) -> None:
        self.assertIn("create table if not exists public.prediction_delivery_events", self.sql)
        self.assertIn("delivery_status in ('missed', 'failed', 'recovered')", self.sql)
        self.assertIn("event_source in ('hugging_face', 'supabase')", self.sql)

    def test_migration_is_transactional_and_additive(self) -> None:
        self.assertIn("begin;", self.sql)
        self.assertIn("commit;", self.sql)
        self.assertNotIn("drop table", self.sql.lower())


if __name__ == "__main__":
    unittest.main()
