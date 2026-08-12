# Open Emotional Learner Model

Open Emotional Learner Model, or OELM, is a browser-based learning engagement tool. The frontend captures local face blendshape data with MediaPipe, sends aggregated features to a Hugging Face-hosted model API, and stores sessions, predictions, login credentials, cohorts, logs, and manual reflection edits in Supabase.

## Project Structure

```text
.
|-- index.html                 # Small browser entry point
|-- src/
|   |-- app.js                 # React app, UI flow, charts, capture, Supabase calls
|   |-- config.js              # Frontend URLs and public Supabase anon key
|   |-- styles.css             # App styling
|   `-- data/
|       `-- featureOrder.js    # Ordered model feature list sent to the API
|-- docs/
|   |-- supabase-schema-alignment.sql
|   |-- supabase-manual-overrides-history.sql
|   `-- supabase-final-cleanup.sql
|-- app.py                     # FastAPI inference server
|-- Models/                    # Local model files
|-- feature_order.pkl          # Backend feature order used by the API
|-- requirements.txt           # Python dependencies
|-- Dockerfile                 # Docker/Hugging Face Space container setup
|-- Procfile                   # Process command for deployment platforms
`-- railway.toml               # Railway deployment settings
```

## Frontend

The browser app is loaded from `index.html`, but the actual work is separated into files under `src/`.

- `src/config.js` contains safe committed defaults; local values go in the ignored `src/config.local.js` file.
- `src/app.js` contains the React application logic.
- `src/styles.css` contains all visual styling.
- `src/data/featureOrder.js` contains the exact feature order used when calling the model API.

Because the app uses JavaScript modules, run it through a local web server instead of opening `index.html` directly.

```bash
python -m http.server 8080
```

Then open:

```text
http://localhost:8080
```

## Configuration

Copy `src/config.example.js` to `src/config.local.js`, then add the local runtime values there:

```text
src/config.local.js
```

Important values:

- `HF_SPACE_URL`: public Hugging Face Space URL for the model API.
- `API_BASE_URL`: API base URL used by the frontend. Usually the same as `HF_SPACE_URL`.
- `SUPABASE_URL`: Supabase project URL.
- `SUPABASE_KEY`: Supabase anon/public key used by the browser.
- `ADMIN_USER_ID`: local instructor login identifier.
- `ADMIN_PASSWORD`: local instructor login password.

`src/config.local.js` is ignored by Git and must never be committed. Any value delivered to a browser is still visible to that browser, so protect Supabase data with strict Row Level Security. For a production instructor login, replace the client-side ID/password comparison with server-side authentication.

## Model API

The backend is a FastAPI service in `app.py`.

Main endpoints:

- `GET /health`: checks that the server is running and models are loaded.
- `POST /predict`: accepts aggregated blendshape features and returns predictions for Boredom, Engagement, Confusion, and Frustration.

Run locally:

```bash
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 7860
```

The frontend expects the API to expose `/predict` and allow browser requests through CORS.

## Supabase Tables Used By The Frontend

The frontend currently expects these tables to exist:

- `sessions`
- `emotion_predictions`
- `manual_overrides`
- `login_credentials`
- `logs`

Before production use, confirm the database schema matches the fields written by `src/app.js`.

For clearer naming, run `docs/supabase-schema-alignment.sql` first in the Supabase SQL Editor. It adds columns that match the dashboard language: `cohort_id`, `activity_type`, `task_description`, `access_code`, `created_at`, `participant_id`, and `vlearn_url`.

For manual edit history, run `docs/supabase-manual-overrides-history.sql`. It keeps every manual override row, marks older rows as `status = false`, and keeps only the latest matching edit as `status = true`. The instructor dashboard reads only `status = true` manual overrides.

After the app is verified against those aligned columns, run `docs/supabase-final-cleanup.sql` to remove older redundant columns such as `session_group`, `group_id`, `password`, `type`, `reflection_mode`, `session_group_type`, `session_description`, `user_id`, `date`, `time_stamp`, and `learning_url`. The cleanup also clears `logs.event_data` so the logs table keeps the event name in `event_name` and stores searchable values in normal columns.

## Development Notes

- Camera processing stays local in the browser; the app sends aggregated blendshape features to the model API, not webcam video.
- Keep `src/data/featureOrder.js` aligned with `feature_order.pkl` used by the backend.
- If the Hugging Face model URL changes, update `src/config.js` only.
- If Supabase policies or table names change, update the Supabase helper calls in `src/app.js`.

## Load Testing

Use `tools/oelm_load_test.py` to simulate the startup burst and 10-second prediction cadence of 50 simultaneous students. See `docs/load-testing.md` for safe Hugging Face-only, Supabase read-only, and guarded end-to-end write-test commands.
