-- OELM manual override history migration
-- Run this in the Supabase SQL Editor before testing repeated edits.
--
-- Goal:
-- 1. Every manual edit is stored as a separate row.
-- 2. Only the latest row for the same session/cohort/participant/reflection/bucket/emotion is status=true.
-- 3. Older rows are kept for history with status=false.

begin;

alter table public.manual_overrides
  add column if not exists status boolean default true,
  add column if not exists overridden_at timestamptz default now(),
  add column if not exists participant_id text,
  add column if not exists cohort_id text,
  add column if not exists pause_and_reflect_number integer;

-- Remove old unique rules that forced one row per bucket and caused overwrites.
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

-- For any existing duplicate active rows, keep only the latest one active.
with ranked as (
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
from ranked r
where mo.ctid = r.ctid
  and r.rn > 1;

-- Enforce one active row for each edited bucket while still allowing history rows.
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

create index if not exists manual_overrides_history_lookup_idx
  on public.manual_overrides (
    session_id,
    participant_id,
    cohort_id,
    pause_and_reflect_number,
    bucket_label,
    label_col,
    overridden_at desc
  );

commit;
