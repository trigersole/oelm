-- Validate exactly one participant access code without exposing login_credentials.
-- Run in the Supabase SQL Editor after taking a database backup.

begin;

create or replace function public.validate_cohort_access_code(p_access_code text)
returns table (
  cohort_id text,
  activity_type text,
  task_description text,
  active boolean
)
language sql
stable
security definer
set search_path = ''
as $$
  select
    credentials.cohort_id::text,
    credentials.activity_type::text,
    credentials.task_description::text,
    coalesce(credentials.active, true)
  from public.login_credentials as credentials
  where credentials.access_code = btrim(p_access_code)
    and coalesce(credentials.active, true)
  limit 1;
$$;

revoke all on function public.validate_cohort_access_code(text) from public;
grant execute on function public.validate_cohort_access_code(text) to anon, authenticated;

-- Participants may call the function, but cannot list or change credentials.
revoke all on table public.login_credentials from anon, authenticated;
grant select, insert, update on table public.login_credentials to service_role;

commit;
