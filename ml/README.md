# OELM temporal ordinal training

This directory trains the proposed lightweight multi-task TCN-CORN model on
the frame-level DAiSEE blendshape CSV files. It does not replace the existing
XGBoost model or API.

## Data mapping

`video_name` in each split CSV is matched to `ClipID` in `AllLabels.csv` after
trimming whitespace and removing the `.avi` suffix. The preparation command
also verifies that no labeled clip occurs in more than one split.

The current files contain 143 test clips with no row in `AllLabels.csv`. They
are retained in the original CSV but excluded from supervised evaluation and
listed in `ml/artifacts/data/metadata.json`.

Clips with fewer than 50 extracted frames are linearly resampled onto the same
5 FPS timeline used by OELM. Most clips already contain all 50 frames.

## Environment

Create a separate training environment so the deployed XGBoost API remains
small:

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python -m pip install -r requirements-ml.txt
```

## Prepare fixed-length sequences

```powershell
.\.venv-ml\Scripts\python -m ml.prepare_sequences
```

The resulting compressed files are written to `ml/artifacts/data` and are
ignored by Git.

## Smoke test

```powershell
.\.venv-ml\Scripts\python -m unittest ml.test_pipeline
.\.venv-ml\Scripts\python -m ml.train_tcn_corn --epochs 2 --max-train-clips 256
```

## Full training

```powershell
.\.venv-ml\Scripts\python -m ml.train_tcn_corn --epochs 50 --patience 10
```

To train the directly comparable bidirectional LSTM challenger:

```powershell
.\.venv-ml\Scripts\python -m ml.train_tcn_corn --architecture lstm --epochs 50 --patience 10 --output-dir ml/artifacts/model-lstm
```

The LSTM run exports `model_lstm_corn.ts`; the TCN run exports
`model_tcn_corn.ts`.

## Binary Low/High experiments

Binary targets use `Low = levels 0/1` and `High = levels 2/3`. Training uses
class-balanced loss weights calculated from the training split. Validation and
test distributions are never resampled. Each decision threshold is selected
on validation macro-F1 and then held fixed for the test set.

```powershell
.\.venv-ml\Scripts\python -m ml.train_tcn_corn --architecture tcn --target-mode binary --output-dir ml/artifacts/model-binary-tcn
.\.venv-ml\Scripts\python -m ml.train_tcn_corn --architecture lstm --target-mode binary --output-dir ml/artifacts/model-binary-lstm
.\.venv-ml\Scripts\python -m ml.evaluate_binary_xgboost
.\.venv-ml\Scripts\python -m ml.train_xgboost_binary
```

See `ALL_MODEL_RESULTS.md` or `results/all_model_results_two_run.xlsx` for the
standardized two-seed Low/High metrics and balanced-model recommendation.

## Behavioural-feature ablation

The raw DAiSEE videos can add 12 label-independent signals to each existing
5 FPS sequence: head pitch/yaw/roll, binocular gaze position, eye aspect ratio,
mouth opening, face position/area, and face-detection continuity. Extraction
uses exactly the clip IDs already present in `ml/artifacts/data`, so it does not
change the evaluated population.

Create an extraction-only environment and download the same official Face
Landmarker task referenced by `src/app.js`:

```powershell
python -m venv .venv-behavior
.\.venv-behavior\Scripts\python -m pip install -r requirements-behavior.txt
curl.exe -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -o "ml\artifacts\face_landmarker.task"
```

Then extract, fuse, and train the otherwise unchanged binary TCN:

```powershell
.\.venv-behavior\Scripts\python -m ml.extract_behavioral_features --workers 4
.\.venv-ml\Scripts\python -m ml.fuse_behavioral_features
.\.venv-ml\Scripts\python -m ml.train_tcn_corn --architecture tcn --target-mode binary --data-dir ml/artifacts/data-behavior --output-dir ml/artifacts/model-binary-tcn-behavior
```

To train balanced binary XGBoost on the original 364 aggregate features plus
84 aggregates from the 12 behavioural channels:

```powershell
.\.venv-ml\Scripts\python -m ml.train_xgboost_binary --data-dir ml/artifacts/data-behavior --include-extra-features --output-dir ml/artifacts/model-binary-xgboost-behavior
```

Run the paired comparison and the 30-sequence CPU benchmark with:

```powershell
.\.venv-ml\Scripts\python -m ml.compare_xgboost_behavior
.\.venv-ml\Scripts\python -m ml.benchmark_xgboost_inference
```

## Classical binary baselines

The classical trainer uses the same 448 combined clip-level aggregates and
the same participant-independent split as combined-feature XGBoost. Run all
four baselines for the primary seed, then repeat the stochastic tree models:

```powershell
.\.venv-ml\Scripts\python -m ml.train_classical_binary --seed 1729
.\.venv-ml\Scripts\python -m ml.train_classical_binary --seed 42 --models random_forest extra_trees
```

This evaluates logistic regression, linear SVM, Random Forest, and Extra Trees
with validation-only threshold selection and a 30-sequence CPU benchmark. The
results are included in the consolidated workbook and CSV below.

Train the same classical algorithms on the original four DAiSEE levels with:

```powershell
.\.venv-ml\Scripts\python -m ml.train_classical_ordinal --seed 1729
.\.venv-ml\Scripts\python -m ml.train_classical_ordinal --seed 42 --models random_forest extra_trees
```

The four-level trainer uses direct multiclass predictions without collapsing
levels. It reports macro-F1, balanced accuracy, accuracy, MAE, and quadratic
weighted kappa for every affective target.

## Selected temporal XGBoost

The temporal experiment expands each 50-frame sequence with quantiles, range,
trend, frame-to-frame velocity, acceleration, lag correlation, motion density,
spectral entropy, and high-frequency motion. It compares the 448-feature
combined baseline, all 1,664 candidates, and 512 training-selected features per
affective target:

```powershell
.\.venv-ml\Scripts\python -m ml.train_xgboost_ordinal_temporal --seed 1729
.\.venv-ml\Scripts\python -m ml.train_xgboost_ordinal_temporal --seed 42 --variants temporal_select_512
```

Feature selection uses training-only ANOVA F scores. Validation data controls
early stopping, and test labels are used only for the final saved metrics. The
selected model files and exact feature lists are stored under
`ml/artifacts/model-temporal-xgboost-temporal_select_512-seed<seed>`.

The feature extractor never reads labels. Thresholds and model selection still
use validation data only; the test set remains untouched until final scoring.

The trainer automatically uses CUDA when available; otherwise it uses CPU. It
saves the best validation checkpoint, a deployable TorchScript model, complete
metrics, and per-clip test predictions under `ml/artifacts/model`.

Model selection is based on mean macro-F1 across boredom, engagement,
confusion, and frustration. The report also includes balanced accuracy, MAE,
quadratic weighted kappa, and ordinary accuracy for each label.

## Compare with the production XGBoost baseline

Install the two baseline-only packages and evaluate both architectures on the
same labeled test sequences:

```powershell
.\.venv-ml\Scripts\python -m pip install xgboost==2.1.1 joblib>=1.3
.\.venv-ml\Scripts\python -m ml.evaluate_xgboost_baseline
```

Do not replace the production model solely because TCN-CORN is newer. Promote
it only when its validation/test metrics improve the measures important to the
study. The first seed-1729 run is effectively tied with XGBoost on mean
macro-F1, while TCN-CORN has lower ordinal MAE.

## CPU inference benchmark

```powershell
.\.venv-ml\Scripts\python -m ml.benchmark_inference
```

This measures both a single prediction and a synchronized batch of 30 users.

See `ALL_MODEL_RESULTS.md` for the audited combined recommendation. The
verified two-run workbook is `ml/results/all_model_results_two_run.xlsx`, with
a flat run-level CSV at `ml/results/all_model_results_two_run.csv`. Every
included model group contains seeds 42 and 1729.
