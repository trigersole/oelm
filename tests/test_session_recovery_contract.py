from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class SessionRecoveryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.edge = (ROOT_DIR / "supabase/functions/oelm-admin/index.ts").read_text(encoding="utf-8")
        cls.migration = (
            ROOT_DIR / "docs/supabase-raw-blendshape-session-recovery.sql"
        ).read_text(encoding="utf-8")

    def test_login_is_scoped_to_participant_and_cohort(self) -> None:
        self.assertIn(
            'const ACTIVE_MODEL_VERSION = "xgb260-static-seed1729-v1";',
            self.edge,
        )
        self.assertIn('input?.action === "participant_login"', self.edge)
        self.assertIn('.eq("cohort_id", cohortId)', self.edge)
        self.assertIn('.eq("participant_id", participantId)', self.edge)
        self.assertIn('role: "oelm-participant"', self.edge)
        self.assertGreaterEqual(
            self.edge.count('.eq("model_version", ACTIVE_MODEL_VERSION)'),
            2,
        )

    def test_raw_batches_require_participant_token(self) -> None:
        self.assertIn('input?.action === "save_blendshape_batch"', self.edge)
        self.assertIn('participant?.role !== "oelm-participant"', self.edge)
        self.assertIn('expectedFeatureCount = 52', self.edge)
        self.assertIn('values.length > 300', self.edge)

    def test_raw_table_is_not_exposed_to_browser_roles(self) -> None:
        self.assertIn("values jsonb not null", self.migration)
        self.assertIn(
            "revoke all on table public.raw_blendshape_batches from anon, authenticated",
            self.migration,
        )


if __name__ == "__main__":
    unittest.main()
