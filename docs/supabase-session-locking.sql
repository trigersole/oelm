-- OELM cohort/session locking
-- Run this entire file, from BEGIN through COMMIT.

begin;

alter table public.sessions
  add column if not exists status text not null default 'active',
  add column if not exists locked_at timestamptz,
  add column if not exists locked_by text;

alter table public.sessions
  drop constraint if exists sessions_status_check;

alter table public.sessions
  add constraint sessions_status_check
  check (status in ('active', 'locked'));

create index if not exists sessions_cohort_status_idx
  on public.sessions (cohort_id, status);

-- Clean up the earlier, more complex draft if it was ever partially applied.
drop trigger if exists reject_active_session_for_locked_cohort on public.sessions;
drop function if exists public.reject_active_session_for_locked_cohort();

create or replace function public.lock_oelm_cohort(
  p_cohort_id text,
  p_locked_by text default 'administrator'
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  locked_count integer;
begin
  if nullif(btrim(p_cohort_id), '') is null then
    raise exception 'Cohort ID is required.';
  end if;

  update public.login_credentials
  set active = false
  where cohort_id = p_cohort_id;

  if not found then
    raise exception 'Cohort not found.';
  end if;

  update public.sessions
  set
    status = 'locked',
    locked_at = coalesce(locked_at, now()),
    locked_by = coalesce(nullif(btrim(p_locked_by), ''), 'administrator')
  where cohort_id = p_cohort_id
    and status = 'active';

  get diagnostics locked_count = row_count;
  return locked_count;
end;
$$;

revoke all on function public.lock_oelm_cohort(text, text) from public, anon, authenticated;
grant execute on function public.lock_oelm_cohort(text, text) to service_role;

create or replace function public.reject_locked_oelm_session_write()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  if exists (
    select 1
    from public.sessions as session_row
    where session_row.session_id::text = new.session_id::text
      and session_row.status = 'locked'
  ) then
    raise exception 'SESSION_LOCKED: this class session no longer accepts data.';
  end if;

  return new;
end;
$$;

revoke all on function public.reject_locked_oelm_session_write() from public, anon, authenticated;

drop trigger if exists reject_locked_session_write_emotion_predictions on public.emotion_predictions;
create trigger reject_locked_session_write_emotion_predictions
before insert or update on public.emotion_predictions
for each row execute function public.reject_locked_oelm_session_write();

drop trigger if exists reject_locked_session_write_logs on public.logs;
create trigger reject_locked_session_write_logs
before insert or update on public.logs
for each row execute function public.reject_locked_oelm_session_write();

drop trigger if exists reject_locked_session_write_manual_overrides on public.manual_overrides;
create trigger reject_locked_session_write_manual_overrides
before insert or update on public.manual_overrides
for each row execute function public.reject_locked_oelm_session_write();

drop trigger if exists reject_locked_session_write_pause_reflection_aoi on public.pause_reflection_aoi;
create trigger reject_locked_session_write_pause_reflection_aoi
before insert or update on public.pause_reflection_aoi
for each row execute function public.reject_locked_oelm_session_write();

drop trigger if exists reject_locked_session_write_prediction_delivery_events on public.prediction_delivery_events;
create trigger reject_locked_session_write_prediction_delivery_events
before insert or update on public.prediction_delivery_events
for each row execute function public.reject_locked_oelm_session_write();

drop trigger if exists reject_locked_session_write_raw_blendshape_batches on public.raw_blendshape_batches;
create trigger reject_locked_session_write_raw_blendshape_batches
before insert or update on public.raw_blendshape_batches
for each row execute function public.reject_locked_oelm_session_write();

commit;
