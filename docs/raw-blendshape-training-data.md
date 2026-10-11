# Raw blendshape training batches

The browser sends one asynchronous batch approximately every 30 seconds while
emotion capture is active. No video, landmarks, or blendshape names are stored
in a batch. Each `raw_blendshape_batches.values` value is a JSON array shaped as:

```text
[frame][52 blendshape scores]
```

Scores use the fixed `BLENDSHAPE_NAMES` order in `src/app.js`. This keeps each
database row small without making the data difficult to retrieve. The row also
contains the session, participant, cohort, capture rate, frame count, and batch
start/end timestamps.

Only the Supabase Edge Function can write this table. Browser publishable keys
have no direct table permission.

## Retrieve batches for training

Use the Supabase SQL Editor or a trusted server with the service-role key:

```sql
select
  session_id,
  participant_id,
  cohort_id,
  started_at,
  ended_at,
  capture_fps,
  frame_count,
  values
from public.raw_blendshape_batches
order by participant_id, session_id, started_at;
```

In Python, each row can be expanded without decompression:

```python
for batch in batches:
    for frame_index, scores in enumerate(batch["values"]):
        elapsed_seconds = frame_index / batch["capture_fps"]
        # scores is a list of 52 floats in BLENDSHAPE_NAMES order
```

Apply `docs/supabase-raw-blendshape-session-recovery.sql` before deploying the
matching frontend and Edge Function.
