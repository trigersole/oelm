# Public deployment setup

The repository contains no deployed Supabase values or administrator password.
Only a Supabase publishable key is delivered to the browser; administrator
credentials and the elevated Supabase key stay in Edge Function secrets.

## 1. Protect cohort access codes

Back up the Supabase project, then run
`docs/supabase-secure-access-code.sql` in the Supabase SQL Editor. The migration
creates `validate_cohort_access_code`, permits visitors to validate one code,
and removes direct browser access to `login_credentials`.

Then run `docs/supabase-raw-blendshape-session-recovery.sql`. It adds the
value-only raw blendshape batch table and the prediction timing fields required
to resume an existing participant session accurately.

Before deploying the request-reliability frontend, also run
`docs/supabase-prediction-reliability.sql`. It adds idempotent prediction
request IDs, backend-only delivery fields, and the
`prediction_delivery_events` audit table. If the frontend is deployed before
this migration, new predictions remain in the browser outbox until the schema
is available.

Finally, run `docs/supabase-session-locking.sql`. It adds the irreversible
instructor class lock, disables future participant logins, and installs
database triggers that reject late, in-flight, or queued writes from locked
sessions. Deploy the updated Edge Function only after this migration succeeds.

## 2. Configure the administrator Edge Function

Deploy `supabase/functions/oelm-admin/index.ts` as the `oelm-admin` function with
JWT verification disabled. Set these values in Supabase Dashboard under Edge
Function Secrets; do not put them in `src/config.local.js` or GitHub:

- `OELM_ADMIN_USER_ID`
- `OELM_ADMIN_PASSWORD`
- `OELM_ADMIN_SESSION_SECRET` (a random value of at least 32 bytes)
- `OELM_SUPABASE_SECRET_KEY` (the server-side secret key)
- `OELM_ALLOWED_ORIGINS` (comma-separated exact origins)
- `OELM_SUPABASE_PUBLISHABLE_KEY` (the browser-safe publishable key)
- `OELM_HF_SPACE_URL` (the public binary-model Space URL)

For local and production testing, an example allowed-origins value is:

```text
http://127.0.0.1:8080,https://oelm-binary.siddhartha-paliwal21.chatgpt.site
```

The function verifies the administrator ID and password server-side, returns a
one-hour signed administrator session, and performs cohort/reporting queries
with the server-only key.

## 3. Prepare frontend deployment files without committing values

After redeploying `oelm-admin` with its `public_config` action, run:

```powershell
.\scripts\prepare-public-dist.ps1
```

The committed frontend contains only the public Edge Function URL. At startup,
the browser requests its Supabase URL, publishable key, and Hugging Face URL
from `public_config`. Administrator credentials and the Supabase secret key are
never returned. A publishable key remains visible to website visitors by
design; never return a secret or service-role key from `public_config`.

## 4. Required checks before publication

1. Confirm an invalid cohort code is rejected.
2. Confirm a valid code creates a session and predictions.
3. In browser Network tools, confirm participant login calls
   `/rest/v1/rpc/validate_cohort_access_code` and never selects
   `/rest/v1/login_credentials`.
4. Confirm the administrator credentials open the administrator view.
5. Confirm cohort creation and reporting work through `oelm-admin`.
6. Confirm a direct browser request to `/rest/v1/login_credentials` is denied.
7. Confirm the prediction request uses `/predict-binary` and returns only Low
   and High labels.
8. Confirm `emotion_predictions.request_id` is populated and unique, then
   briefly test an offline/online transition and confirm the queued row is
   eventually marked `recovered`.

Participant isolation is a separate database change. Until owner-based RLS is
installed on sessions and child tables, do not assume one participant is unable
to query another participant's rows through the public Data API.
