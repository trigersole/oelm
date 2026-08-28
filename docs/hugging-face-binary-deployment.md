# Deploy the binary OELM API to Hugging Face Spaces

## 1. Verify locally first

Run the API and smoke test from the repository root:

```powershell
python -m venv .venv-api
.\.venv-api\Scripts\python -m pip install --upgrade pip
.\.venv-api\Scripts\python -m pip install -r requirements.txt
.\.venv-api\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 7860
```

In another terminal:

```powershell
.\.venv-api\Scripts\python tools\smoke_test_binary_api.py
```

Optional Docker verification:

```powershell
docker build -t oelm-binary .
docker run --rm -p 7860:7860 oelm-binary
```

Then rerun the smoke test.

## 2. Create the Space

1. Open `https://huggingface.co/new-space`.
2. Choose a Space name such as `oelm-binary-xgboost`.
3. Select **Docker** as the SDK.
4. Select **CPU Basic** hardware; XGBoost inference does not require a GPU.
5. Choose public or private visibility based on whether the model source may be public.

The YAML metadata at the top of the root README already declares `sdk: docker`
and `app_port: 7860`.

## 3. Push the feature branch to the Space

From the two-level worktree, add the Space repository as a separate remote:

```powershell
git remote add huggingface https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE_NAME
git push huggingface codex/two-level-model:main
```

Authenticate using a Hugging Face user access token with write permission when
Git prompts for credentials. Do not place the token in source files or the remote URL.

Every pushed commit triggers a Space rebuild and restart.

## 4. Verify the deployed API

Replace the example hostname with the one shown by the Space:

```powershell
$env:OELM_API_URL = 'https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space'
Invoke-RestMethod "$env:OELM_API_URL/health"
python tools\smoke_test_binary_api.py --url $env:OELM_API_URL
```

Expected health values include:

```json
{
  "status": "ok",
  "model_version": "xgb-binary-v1",
  "levels": ["Low", "High"],
  "feature_count": 364
}
```

Interactive API documentation is available at `/docs`, and the exact mapping and
thresholds are available at `/model-info`.

## 5. Point the frontend to the Space

In the ignored local `src/config.local.js` file:

```javascript
window.OELM_CONFIG = {
  HF_SPACE_URL: 'https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space',
  API_BASE_URL: 'https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space',
  SUPABASE_URL: 'YOUR_LOCAL_OR_HOSTED_VALUE',
  SUPABASE_KEY: 'YOUR_LOCAL_OR_HOSTED_VALUE',
  ADMIN_USER_ID: 'YOUR_LOCAL_OR_HOSTED_VALUE',
  ADMIN_PASSWORD: 'YOUR_LOCAL_OR_HOSTED_VALUE',
};
```

Do not commit this file. If the frontend has a fixed production hostname, set the
optional Space variable `OELM_ALLOWED_ORIGINS` to that origin. Multiple origins
can be supplied as a comma-separated list. The default `*` is useful for initial
testing but should be restricted for production.

## Included deployment resources

The Docker build intentionally includes only:

- `app.py`
- `requirements.txt`
- `binary_models/feature_order.json`
- `binary_models/model_config.json`
- `binary_models/model_binary_*.ubj`
- `binary_models/training_metrics.json`

Training datasets, local environments, credentials, Supabase configuration,
legacy four-level models, and ignored ML artifacts are not copied into the image.
