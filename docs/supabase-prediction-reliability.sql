-- OELM browser request reliability and delivery audit
-- Run once in the Supabase SQL Editor before deploying the matching frontend.
-- This migration is additive and preserves all existing prediction rows.

begin;

alter table public.emotion_predictions
  add column if not exists request_id text,
  add column if not exists delivery_status text default 'saved',
  add column if not exists delivery_attempts integer default 1,
  add column if not exists delivered_at timestamptz default now();

update public.emotion_predictions
set
  request_id = coalesce(request_id, gen_random_uuid()::text),
  delivery_status = coalesce(delivery_status, 'saved'),
  delivery_attempts = coalesce(delivery_attempts, 1),
  delivered_at = coalesce(delivered_at, predicted_at, now())
where
  request_id is null
  or delivery_status is null
  or delivery_attempts is null
  or delivered_at is null;

create unique index if not exists emotion_predictions_request_id_key
  on public.emotion_predictions (request_id);

alter table public.emotion_predictions
  drop constraint if exists emotion_predictions_delivery_status_check;

alter table public.emotion_predictions
  add constraint emotion_predictions_delivery_status_check
  check (delivery_status in ('pending', 'saved', 'recovered'));

alter table public.emotion_predictions
  drop constraint if exists emotion_predictions_delivery_attempts_check;

alter table public.emotion_predictions
  add constraint emotion_predictions_delivery_attempts_check
  check (delivery_attempts >= 0);

create table if not exists public.prediction_delivery_events (
  request_id text primary key,
  session_id text,
  participant_id text,
  cohort_id text,
  pause_and_reflect_number integer,
  event_source text not null check (event_source in ('hugging_face', 'supabase')),
  delivery_status text not null check (delivery_status in ('missed', 'failed', 'recovered')),
  attempt_count integer not null default 1 check (attempt_count > 0),
  error_message text,
  first_failed_at timestamptz not null default now(),
  last_attempt_at timestamptz not null default now(),
  created_at timestamptz not null default now()
);

create index if not exists prediction_delivery_events_session_idx
  on public.prediction_delivery_events (session_id, last_attempt_at desc);

create index if not exists prediction_delivery_events_cohort_participant_idx
  on public.prediction_delivery_events (cohort_id, participant_id, last_attempt_at desc);

alter table public.prediction_delivery_events enable row level security;

grant select, insert, update on public.prediction_delivery_events to anon, authenticated;

drop policy if exists prediction_delivery_events_read on public.prediction_delivery_events;
create policy prediction_delivery_events_read
  on public.prediction_delivery_events
  for select
  to anon, authenticated
  using (true);

drop policy if exists prediction_delivery_events_insert on public.prediction_delivery_events;
create policy prediction_delivery_events_insert
  on public.prediction_delivery_events
  for insert
  to anon, authenticated
  with check (
    request_id <> ''
    and event_source in ('hugging_face', 'supabase')
    and delivery_status in ('missed', 'failed', 'recovered')
    and attempt_count > 0
  );

drop policy if exists prediction_delivery_events_update on public.prediction_delivery_events;
create policy prediction_delivery_events_update
  on public.prediction_delivery_events
  for update
  to anon, authenticated
  using (true)
  with check (
    request_id <> ''
    and event_source in ('hugging_face', 'supabase')
    and delivery_status in ('missed', 'failed', 'recovered')
    and attempt_count > 0
  );

commit;
