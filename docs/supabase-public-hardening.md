# Supabase hardening before enabling research-data collection

The public OELM binary demo is intentionally deployed without Supabase values.
It performs inference but does not save participant sessions or research data.

Do not publish `src/config.local.js`, an administrator password, a Supabase
secret/service-role key, or the current unrestricted cohort lookup.

## Why the existing research client is not ready for anonymous internet traffic

- The browser currently reads all rows and access codes from
  `public.login_credentials` before matching an entered code locally.
- Instructor access is currently a browser-side ID/password comparison.
- Existing AOI setup grants `anon` broad read, insert, and update permissions and
  uses policies whose predicates are `true` for reads and updates.
- A public/publishable key is visible to every visitor by design. Database grants
  and Row Level Security therefore have to enforce every permission.

## Recommended production design

1. Take a Supabase backup and test these changes in a separate staging project.
2. Enable Supabase Auth. Use anonymous Auth users for participants and an
   email/password or SSO Auth account for instructors.
3. Add an `auth_user_id uuid` ownership column to participant-owned tables:
   `sessions`, `emotion_predictions`, `manual_overrides`,
   `pause_reflection_aoi`, and `logs`.
4. Populate `auth_user_id` from `auth.uid()` on insert and link child records to
   a session owned by the same authenticated user.
5. Revoke all `anon` access to `login_credentials`. Replace the browser's full
   table read with a narrowly scoped `security definer` RPC or Edge Function
   that validates one access code and returns only its active cohort metadata.
6. Enable RLS on every exposed table. Participant policies must permit only rows
   where `auth_user_id = auth.uid()`; do not use `using (true)` for research data.
7. Move cohort creation and administrator reporting behind authenticated,
   server-side instructor authorization. Never authorize an instructor by
   comparing a password in JavaScript.
8. Grant only the operations each role needs. Participants generally need
   insert/select/update on their own rows, not delete or cross-participant read.
9. Put only the Supabase URL and publishable key in the browser. Keep all secret
   and service-role keys in an Edge Function or other server environment.
10. Test allowed and denied operations for anonymous, participant, and
    instructor identities before enabling data collection on the public site.

## Audit the current project

Run these read-only queries in the Supabase SQL Editor:

```sql
select
  n.nspname as schema_name,
  c.relname as table_name,
  c.relrowsecurity as rls_enabled
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public'
  and c.relkind = 'r'
order by c.relname;

select
  schemaname,
  tablename,
  policyname,
  roles,
  cmd,
  qual,
  with_check
from pg_policies
where schemaname = 'public'
order by tablename, policyname;
```

Do not switch the public deployment out of demo mode until the frontend uses
Supabase Auth and the resulting policies have passed both allow and deny tests.
