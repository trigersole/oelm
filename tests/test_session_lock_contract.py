from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class SessionLockContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frontend = (ROOT_DIR / "src/app.js").read_text(encoding="utf-8")
        cls.edge = (ROOT_DIR / "supabase/functions/oelm-admin/index.ts").read_text(
            encoding="utf-8"
        )
        cls.sql = (ROOT_DIR / "docs/supabase-session-locking.sql").read_text(
            encoding="utf-8"
        )

    def test_lock_is_enforced_in_database_for_all_session_write_tables(self) -> None:
        self.assertIn("create or replace function public.lock_oelm_cohort", self.sql)
        self.assertIn("create or replace function public.reject_locked_oelm_session_write", self.sql)
        for table in (
            "emotion_predictions",
            "logs",
            "manual_overrides",
            "pause_reflection_aoi",
            "prediction_delivery_events",
            "raw_blendshape_batches",
        ):
            self.assertIn(f"public.{table}", self.sql)
        self.assertIn("set active = false", self.sql)
        self.assertIn("status = 'locked'", self.sql)
        self.assertIn("SESSION_LOCKED", self.sql)
        self.assertNotIn("pg_advisory_xact_lock", self.sql)
        self.assertNotIn("perform 1", self.sql.lower())

    def test_edge_function_checks_session_and_exposes_admin_lock(self) -> None:
        self.assertIn('input?.action === "session_status"', self.edge)
        self.assertIn('input?.action === "lock_cohort"', self.edge)
        self.assertIn('db.rpc("lock_oelm_cohort"', self.edge)
        self.assertIn('code: "SESSION_LOCKED"', self.edge)
        self.assertIn("}, 423, origin)", self.edge)

    def test_frontend_stops_collection_and_drops_terminal_retries(self) -> None:
        self.assertIn("SESSION_STATUS_POLL_MS = 120_000", self.frontend)
        self.assertIn("const handleSessionLocked = useCallback", self.frontend)
        self.assertIn("rawBlendshapeBatchRef.current = []", self.frontend)
        self.assertIn("streamRef.current?.getTracks().forEach", self.frontend)
        self.assertIn("await outboxDelete(item.outbox_id)", self.frontend)
        self.assertIn("handleLockCohort(g.id)", self.frontend)
        self.assertIn("'End Class'", self.frontend)


if __name__ == "__main__":
    unittest.main()
