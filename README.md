---
title: OELM Binary XGBoost API
emoji: 🧠
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# Open Emotional Learner Model — binary XGBoost

This branch is the two-level OELM variant. It combines the four original DAiSEE
intensity levels into a binary target for each affective state:

| Binary class | Original DAiSEE levels |
|---|---|
| `0 — Low` | Very Low + Low |
| `1 — High` | High + Very High |

The browser processes camera frames locally with MediaPipe, aggregates 52 face
blendshapes into 364 statistics every 10 seconds, and sends only those statistics
to the FastAPI model service. Four directly trained binary XGBoost heads predict
Boredom, Engagement, Confusion, and Frustration.

## Selected checkpoint

The deployed models use training seed `1729`. This seed was selected because it
had the stronger mean validation macro-F1 of the two audited runs. The held-out
test split was not used for checkpoint selection.

The per-label thresholds were also selected on validation macro-F1:

| Label | High threshold |
|---|---:|
| Boredom | 0.47 |
| Engagement | 0.46 |
| Confusion | 0.49 |
| Frustration | 0.49 |

The exact thresholds, feature order, model files, and preserved metrics are in
[`binary_models/`](binary_models/).

## Run and verify the model locally

From this branch or worktree in PowerShell:

```powershell
python -m venv .venv-api
.\.venv-api\Scripts\python -m pip install --upgrade pip
.\.venv-api\Scripts\python -m pip install -r requirements.txt
.\.venv-api\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 7860
```

In a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:7860/health
.\.venv-api\Scripts\python tools\smoke_test_binary_api.py
.\.venv-api\Scripts\python -m unittest tests.test_binary_api
```

The response for every affective target must contain only `label: 0` / `Low` or
`label: 1` / `High`, with two probabilities that sum to one.

### Test the complete browser application

Copy the safe configuration template to the ignored local file:

```powershell
Copy-Item src\config.example.js src\config.local.js
```

Set these two values in `src/config.local.js` while the local API is running:

```javascript
HF_SPACE_URL: 'http://127.0.0.1:7860',
API_BASE_URL: 'http://127.0.0.1:7860',
```

Add your local Supabase and login values to the same ignored file, then start the
frontend:

```powershell
python -m http.server 8080
```

Open `http://localhost:8080`. Predictions, timelines, split charts, manual
overrides, saved probabilities, and simulated predictions all use only Low/High.

## API contract

Endpoints:

- `GET /` — service links
- `GET /health` — model readiness and feature count
- `GET /model-info` — mapping, seed, and validation thresholds
- `POST /predict` — binary inference from `agg_features`
- `GET /docs` — interactive OpenAPI documentation

Example response fragment:

```json
{
  "Boredom": {
    "label": 0,
    "level": "Low",
    "probabilities": {"0": 0.72, "1": 0.28},
    "threshold": 0.47
  }
}
```

## Hugging Face deployment

This repository is ready for a Hugging Face **Docker Space**. The README metadata
sets `sdk: docker` and `app_port: 7860`; the Dockerfile packages only the API and
the binary deployment resources.

See [`docs/hugging-face-binary-deployment.md`](docs/hugging-face-binary-deployment.md)
for the complete deployment and verification workflow.

## Project layout

```text
app.py                         Binary FastAPI inference service
binary_models/                 Four UBJ models, thresholds, feature order, metrics
Dockerfile                     Hugging Face Docker Space image
requirements.txt               Lightweight API-only Python dependencies
index.html                     Browser entry point
src/app.js                     React interface and MediaPipe processing
src/config.example.js          Safe configuration template
docs/                          Supabase, deployment, and testing instructions
ml/train_xgboost_binary.py     Reproducible direct-binary training pipeline
tests/test_binary_api.py       Local model/API contract tests
tools/smoke_test_binary_api.py Running-server smoke test
```

## Supabase configuration

The frontend uses the following tables:

- `sessions`
- `emotion_predictions`
- `manual_overrides`
- `pause_reflection_aoi`
- `login_credentials`
- `logs`

Apply the SQL files in `docs/` as described by their headers. Keep all Supabase,
Hugging Face, ID, and password values in the ignored `src/config.local.js` or in
the hosting platform's secret/variable settings. Never commit them.

Any value delivered to browser JavaScript is visible to the browser. Supabase
Row Level Security must enforce data access, and production instructor login
should use server-side authentication rather than a client-side password check.
