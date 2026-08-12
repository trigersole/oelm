# Load testing OELM for 50 simultaneous students

The browser creates two different traffic patterns:

- Startup burst: each opened browser immediately reads the cohort list from Supabase and sends a warm-up prediction to Hugging Face.
- Steady learning: each active student sends one prediction every 10 seconds and then inserts the prediction into Supabase.

For 50 students, steady traffic is approximately 5 Hugging Face requests/second and 5 Supabase prediction inserts/second. The startup burst can be 50 requests arriving almost together, so test both the burst and a longer steady phase.

The current prediction payload is about 11.2 KB because `emotion_predictions.raw_data` contains all 364 aggregate features. At 50 students and one prediction every 10 seconds, a one-hour class produces about 18,000 rows and sends roughly 200 MB of prediction JSON to Supabase before database compression and index overhead. During a longer test, also watch database size growth; request latency alone is not enough.

## 1. Hugging Face test (safe, no database writes)

Run from the project root in PowerShell:

```powershell
python tools/oelm_load_test.py `
  --users 50 `
  --duration 300 `
  --interval 10 `
  --ramp-up 0 `
  --output load-test-results/hf-50-users.json
```

`--ramp-up 0` is the worst-case start in which all 50 students arrive together. Run a second, more typical scenario with `--ramp-up 30`.

The defaults mark the test failed when the error rate exceeds 1% or Hugging Face p95 latency exceeds 2 seconds. The app has a 10-second prediction interval, but keeping p95 below 2 seconds leaves useful headroom.

## 2. Supabase read burst (no writes)

Set the values only in the current PowerShell process. Do not commit a service-role key; the browser's anon key is sufficient and should remain protected by Row Level Security.

```powershell
$env:SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
$env:SUPABASE_ANON_KEY = "YOUR_ANON_KEY"

python tools/oelm_load_test.py `
  --users 50 `
  --duration 60 `
  --interval 10 `
  --ramp-up 0 `
  --supabase-mode read `
  --output load-test-results/supabase-read-50-users.json
```

This sends the same cohort-list query that each browser makes, but does not insert data.

## 3. Full end-to-end write test

Create or choose a dedicated test cohort first. The command below creates identifiable session, log, and prediction rows in the configured Supabase project. It never deletes them automatically.

```powershell
python tools/oelm_load_test.py `
  --users 50 `
  --duration 300 `
  --interval 10 `
  --ramp-up 0 `
  --supabase-mode write `
  --cohort-id YOUR_TEST_COHORT `
  --confirm-production-writes `
  --output load-test-results/full-50-users.json
```

The command prints a run ID. Rows are identified by:

- `sessions.label = 'loadtest:<run-id>'`
- participant IDs beginning with `loadtest-<run-id>-`
- `logs.event_name = 'load_test_session_login'`
- `emotion_predictions.model_version = 'xgb-v1-load-test'`

To clean up, first replace the run ID below and run the `select` to verify the exact sessions. Only then run the transaction.

```sql
select session_id, participant_id, label
from public.sessions
where label = 'loadtest:REPLACE_WITH_RUN_ID';

begin;

create temporary table load_test_sessions on commit drop as
select session_id
from public.sessions
where label = 'loadtest:REPLACE_WITH_RUN_ID';

delete from public.manual_overrides
where session_id in (select session_id from load_test_sessions);

delete from public.emotion_predictions
where session_id in (select session_id from load_test_sessions);

delete from public.logs
where session_id in (select session_id from load_test_sessions);

delete from public.sessions
where session_id in (select session_id from load_test_sessions);

commit;
```

## What to watch during the run

In the Supabase dashboard, open Reports and watch the API Gateway/PostgREST response errors and response speed, plus database CPU, memory, IOPS, and connection counts. Inspect API logs for `429`, `5xx`, or `PGRST003` errors.

For Hugging Face, keep the Space page and runtime logs open. A passing test should have:

- less than 1% failed requests;
- no sustained `429`, `502`, `503`, or timeout responses;
- Hugging Face p95 below 2 seconds;
- Supabase p95 below 1 second;
- no steadily rising latency during the five-minute steady phase.

If the Hugging Face test fails on CPU Basic, move to CPU Upgrade before considering a GPU: OELM uses small XGBoost CPU models. Paid Space hardware can also use multiple replicas. If Supabase fails, first inspect slow queries and Row Level Security policies, confirm the existing indexes from the schema scripts are present, and then consider a larger compute size.
