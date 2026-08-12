-- OELM Supabase schema alignment
-- Run this first in Supabase SQL Editor.
-- It is non-destructive: it adds columns that match your shared table names
-- and backfills them from the older column names.

-- Shared UI name -> Supabase column name
-- Cohort ID      -> cohort_id
-- Activity Type  -> activity_type
-- Task Description -> task_description
-- Access Code    -> access_code
-- Created On     -> created_at
-- Participant ID -> participant_id

-- 1) Cohort / access-code table
alter table public.login_credentials
  add column if not exists cohort_id text,
  add column if not exists activity_type text,
  add column if not exists task_description text,
  add column if not exists access_code text,
  add column if not exists created_at timestamptz,
  add column if not exists active boolean default true,
  add column if not exists group_id text,
  add column if not exists reflection_mode text,
  add column if not exists session_group text,
  add column if not exists password text,
  add column if not exists date timestamptz,
  add column if not exists type text,
  add column if not exists session_description text;

update public.login_credentials
set
  cohort_id = coalesce(cohort_id, group_id, session_group),
  activity_type = coalesce(activity_type, reflection_mode, type),
  task_description = coalesce(task_description, session_description),
  access_code = coalesce(access_code, password),
  created_at = coalesce(created_at, date)
where
  cohort_id is null
  or activity_type is null
  or task_description is null
  or access_code is null
  or created_at is null;

create unique index if not exists login_credentials_cohort_id_key
  on public.login_credentials (cohort_id);

create unique index if not exists login_credentials_access_code_key
  on public.login_credentials (access_code);

-- 2) Participant sessions table
alter table public.sessions
  add column if not exists cohort_id text,
  add column if not exists activity_type text,
  add column if not exists participant_id text,
  add column if not exists group_id text,
  add column if not exists reflection_mode text,
  add column if not exists session_group text,
  add column if not exists session_group_type text,
  add column if not exists user_id text;

update public.sessions
set
  cohort_id = coalesce(cohort_id, group_id, session_group),
  activity_type = coalesce(activity_type, reflection_mode, session_group_type),
  participant_id = coalesce(participant_id, user_id)
where
  cohort_id is null
  or activity_type is null
  or participant_id is null;

create index if not exists sessions_cohort_participant_idx
  on public.sessions (cohort_id, participant_id);

create index if not exists sessions_session_id_idx
  on public.sessions (session_id);

-- 3) Prediction table
alter table public.emotion_predictions
  add column if not exists cohort_id text,
  add column if not exists activity_type text,
  add column if not exists participant_id text,
  add column if not exists group_id text,
  add column if not exists reflection_mode text,
  add column if not exists user_id text;

update public.emotion_predictions ep
set
  cohort_id = coalesce(ep.cohort_id, ep.group_id, s.cohort_id, s.group_id, s.session_group),
  activity_type = coalesce(ep.activity_type, ep.reflection_mode, s.activity_type, s.reflection_mode, s.session_group_type),
  participant_id = coalesce(ep.participant_id, ep.user_id, s.participant_id, s.user_id)
from public.sessions s
where ep.session_id = s.session_id
  and (
    ep.cohort_id is null
    or ep.activity_type is null
    or ep.participant_id is null
  );

create index if not exists emotion_predictions_session_id_idx
  on public.emotion_predictions (session_id);

create index if not exists emotion_predictions_cohort_participant_idx
  on public.emotion_predictions (cohort_id, participant_id);

-- 4) Manual override table
alter table public.manual_overrides
  add column if not exists cohort_id text,
  add column if not exists participant_id text,
  add column if not exists status boolean default true,
  add column if not exists overridden_at timestamptz default now(),
  add column if not exists pause_and_reflect_number integer,
  add column if not exists group_id text,
  add column if not exists session_group text,
  add column if not exists user_id text;

update public.manual_overrides
set
  cohort_id = coalesce(cohort_id, group_id, session_group),
  participant_id = coalesce(participant_id, user_id)
where cohort_id is null or participant_id is null;

create index if not exists manual_overrides_cohort_participant_idx
  on public.manual_overrides (cohort_id, participant_id);

create index if not exists manual_overrides_active_lookup_idx
  on public.manual_overrides (
    session_id,
    participant_id,
    cohort_id,
    pause_and_reflect_number,
    bucket_label,
    label_col
  )
  where status = true;

do $$
declare
  item record;
begin
  for item in
    select conname
    from pg_constraint c
    join pg_class t on t.oid = c.conrelid
    join pg_namespace n on n.oid = t.relnamespace
    where n.nspname = 'public'
      and t.relname = 'manual_overrides'
      and c.contype = 'u'
      and exists (
        select 1
        from unnest(c.conkey) k
        join pg_attribute a on a.attrelid = c.conrelid and a.attnum = k
        where a.attname = 'session_id'
      )
      and exists (
        select 1
        from unnest(c.conkey) k
        join pg_attribute a on a.attrelid = c.conrelid and a.attnum = k
        where a.attname = 'bucket_label'
      )
      and exists (
        select 1
        from unnest(c.conkey) k
        join pg_attribute a on a.attrelid = c.conrelid and a.attnum = k
        where a.attname = 'label_col'
      )
  loop
    execute format('alter table public.manual_overrides drop constraint if exists %I', item.conname);
  end loop;

  for item in
    select idx.relname as index_name
    from pg_index i
    join pg_class idx on idx.oid = i.indexrelid
    join pg_class tbl on tbl.oid = i.indrelid
    join pg_namespace n on n.oid = tbl.relnamespace
    where n.nspname = 'public'
      and tbl.relname = 'manual_overrides'
      and i.indisunique
      and not i.indisprimary
      and idx.relname <> 'manual_overrides_one_active_per_bucket_idx'
      and pg_get_indexdef(i.indexrelid) ilike '%session_id%'
      and pg_get_indexdef(i.indexrelid) ilike '%bucket_label%'
      and pg_get_indexdef(i.indexrelid) ilike '%label_col%'
  loop
    execute format('drop index if exists public.%I', item.index_name);
  end loop;
end $$;

-- Keep manual edit history while allowing only the latest row to stay active
-- for the same edited bucket.
with ranked_manual_overrides as (
  select
    ctid,
    row_number() over (
      partition by
        coalesce(session_id::text, ''),
        coalesce(participant_id::text, ''),
        coalesce(cohort_id::text, ''),
        coalesce(pause_and_reflect_number, -1),
        coalesce(bucket_label::text, ''),
        coalesce(label_col::text, '')
      order by coalesce(overridden_at, 'epoch'::timestamptz) desc
    ) as rn
  from public.manual_overrides
  where status = true
)
update public.manual_overrides mo
set status = false
from ranked_manual_overrides r
where mo.ctid = r.ctid
  and r.rn > 1;

create unique index if not exists manual_overrides_one_active_per_bucket_idx
  on public.manual_overrides (
    coalesce(session_id::text, ''),
    coalesce(participant_id::text, ''),
    coalesce(cohort_id::text, ''),
    coalesce(pause_and_reflect_number, -1),
    coalesce(bucket_label::text, ''),
    coalesce(label_col::text, '')
  )
  where status = true;

-- 5) Logs table
alter table public.logs
  add column if not exists cohort_id text,
  add column if not exists participant_id text,
  add column if not exists created_at timestamptz,
  add column if not exists vlearn_url text,
  -- temporary migration-only column; final cleanup removes it
  add column if not exists learning_url text,
  add column if not exists group_id text,
  add column if not exists session_group text,
  add column if not exists user_id text,
  add column if not exists time_stamp timestamptz;

update public.logs
set
  cohort_id = coalesce(cohort_id, group_id, session_group),
  participant_id = coalesce(participant_id, user_id, event_data->>'participant_id', event_data->>'user_id'),
  created_at = coalesce(created_at, time_stamp),
  vlearn_url = coalesce(vlearn_url, event_data->>'vlearn_url', event_data->>'learning_url', event_data->>'link_url'),
  learning_url = coalesce(learning_url, event_data->>'learning_url', event_data->>'vlearn_url', event_data->>'link_url')
where
  cohort_id is null
  or participant_id is null
  or created_at is null
  or vlearn_url is null;

create index if not exists logs_cohort_participant_idx
  on public.logs (cohort_id, participant_id);

create index if not exists logs_event_created_idx
  on public.logs (event_name, created_at desc);

-- After this succeeds, verify the app. Then run
-- docs/supabase-final-cleanup.sql to remove redundant legacy columns.
