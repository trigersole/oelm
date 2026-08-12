-- Cleanup for OELM load-test run 20260805-50students-5min-01 only.
-- The DELETEs are deliberately constrained by the dedicated cohort, run label,
-- participant prefix, event name, and load-test model version.

begin;

delete from public.manual_overrides
where cohort_id = 'oelm_50_5min_20260805_01'
  and participant_id like 'loadtest-20260805-50students-5min-01-%';

delete from public.emotion_predictions
where cohort_id = 'oelm_50_5min_20260805_01'
  and participant_id like 'loadtest-20260805-50students-5min-01-%'
  and model_version = 'xgb-v1-load-test';

delete from public.logs
where cohort_id = 'oelm_50_5min_20260805_01'
  and participant_id like 'loadtest-20260805-50students-5min-01-%'
  and event_name = 'load_test_session_login';

delete from public.sessions
where cohort_id = 'oelm_50_5min_20260805_01'
  and participant_id like 'loadtest-20260805-50students-5min-01-%'
  and label = 'loadtest:20260805-50students-5min-01';

delete from public.login_credentials
where cohort_id = 'oelm_50_5min_20260805_01'
  and task_description = 'Automated 50-student five-minute load test';

commit;

-- All values should be zero after the transaction.
select
  (select count(*) from public.sessions
    where cohort_id = 'oelm_50_5min_20260805_01') as sessions_remaining,
  (select count(*) from public.emotion_predictions
    where cohort_id = 'oelm_50_5min_20260805_01') as predictions_remaining,
  (select count(*) from public.logs
    where cohort_id = 'oelm_50_5min_20260805_01') as logs_remaining,
  (select count(*) from public.login_credentials
    where cohort_id = 'oelm_50_5min_20260805_01') as cohorts_remaining;
