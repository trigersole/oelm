# All OELM model results

This is the authoritative standardized report. Every included OELM model is averaged across exactly two independent training runs (seeds 42 and 1729). Ordinal and binary results are separate tasks and are not directly comparable.

## Four-level ordinal models

| Model | Runs | Macro-F1 | Balanced accuracy | Accuracy | MAE | Kappa | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---|
| TCN-CORN | 2 | 29.22% | 30.05% | 55.14% | 0.565 | 0.157 | Best temporal ordinal challenger |
| BiLSTM-CORN | 2 | 29.32% | 29.92% | 55.19% | 0.556 | 0.149 | Recurrent temporal benchmark |
| Logistic Regression | 2 | 26.21% | 30.99% | 40.87% | 0.828 | 0.114 | Fast baseline; weak exact-level discrimination |
| Linear SVM | 2 | 28.58% | 32.16% | 50.03% | 0.693 | 0.158 | Best classical 4-level class balance |
| Random Forest | 2 | 24.07% | 26.77% | 62.07% | 0.471 | 0.059 | Majority-dominated; low MAE but weak macro-F1 |
| Extra Trees | 2 | 24.56% | 27.10% | 62.52% | 0.461 | 0.073 | Best tree MAE; weak minority-level recall |
| Retrained combined XGBoost | 2 | 28.74% | 28.98% | 57.84% | 0.518 | 0.134 | Controlled retraining baseline for temporal ablation |
| Full temporal XGBoost | 2 | 28.56% | 28.86% | 58.89% | 0.495 | 0.131 | Best temporal XGBoost raw accuracy/MAE; costly and weaker macro-F1 |
| Selected temporal XGBoost | 2 | 29.35% | 29.54% | 58.21% | 0.508 | 0.140 | Compact temporal trade-off; strongest temporal XGBoost macro-F1 |

## DAiSEE paper Table 3 accuracy comparison

The paper values below were supplied by the user and report Top-1 accuracy averaged over three runs. The comparison is indicative, not strictly equivalent, because input modality, preprocessing, run aggregation, and evaluated population may differ.

| Origin / model | Boredom | Engagement | Confusion | Frustration | Mean |
|---|---:|---:|---:|---:|---:|
| Paper: InceptionNet Frame Level | 36.50% | 47.10% | 70.30% | 78.30% | 58.05% |
| Paper: InceptionNet Video Level | 32.30% | 46.40% | 66.30% | 77.30% | 55.58% |
| Paper: C3D Training | 47.20% | 48.60% | 67.90% | 78.30% | 60.50% |
| Paper: C3D Fine-Tuning | 45.20% | 56.10% | 66.30% | 79.10% | 61.68% |
| Paper: LRCN | 53.70% | 57.90% | 72.30% | 73.50% | 64.35% |
| OELM: Selected temporal XGBoost | 41.25% | 52.28% | 65.16% | 74.16% | 58.21% |
| OELM: Extra Trees | 47.81% | 54.15% | 69.72% | 78.38% | 62.52% |

## Binary Low/High models

| Model | Runs | Macro-F1 | Balanced accuracy | Accuracy | Kappa | Problem recall | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---|
| Direct binary XGBoost | 2 | 54.79% | 55.49% | 82.83% | 0.098 | 22.21% | Superseded by combined-feature XGBoost |
| Binary TCN | 2 | 54.86% | 57.46% | 79.26% | 0.116 | 31.15% | Recommended intervention-detection gate |
| Binary BiLSTM | 2 | 55.54% | 56.58% | 82.51% | 0.122 | 26.01% | Compromise model; not primary recommendation |
| Binary TCN + behaviour | 2 | 55.39% | 57.00% | 81.63% | 0.118 | 27.38% | Do not promote; problem recall decreased |
| Combined-feature XGBoost | 2 | 55.96% | 57.11% | 83.08% | 0.122 | 25.53% | Best XGBoost; confirmation/calibration model |
| Logistic Regression | 2 | 55.73% | 56.42% | 84.07% | 0.120 | 21.95% | Fast, interpretable classical baseline |
| Linear SVM | 2 | 54.88% | 55.93% | 83.51% | 0.108 | 21.67% | Fast baseline; weaker than logistic regression |
| Random Forest | 2 | 56.31% | 58.57% | 81.88% | 0.133 | 29.86% | Best classical intervention-oriented trade-off |
| Extra Trees | 2 | 56.72% | 57.82% | 84.08% | 0.138 | 25.03% | Best classical overall macro-F1 and kappa |

## Final recommendation

- Use selected temporal XGBoost as the compact four-label temporal model: 29.35% macro-F1, 29.54% balanced accuracy, 58.21% accuracy, and 0.508 MAE across two seeds.
- Full temporal XGBoost has slightly stronger raw accuracy/MAE (58.89%, 0.495) but lower macro-F1 (28.56%) and uses all 1,664 features, so it is the accuracy-first rather than deployment-first option.
- Selected temporal XGBoost's 58.21% mean accuracy is slightly above the paper's frame-level InceptionNet mean (58.05%), but below C3D training (60.50%), C3D fine-tuning (61.68%), and LRCN (64.35%).
- Keep Linear SVM as the four-label class-balance benchmark.
- Pair the selected temporal severity model with the binary TCN intervention gate because the temporal severity gain does not improve rare-level balance.
- Do not select four-label Random Forest or Extra Trees from their high accuracy/low MAE alone; their macro-F1 shows majority-level dominance.
- Use combined-feature binary XGBoost as the best XGBoost confirmation/calibration model.
- Random Forest is the strongest classical one-model trade-off; Extra Trees has the best classical macro-F1 and kappa.
- Logistic regression is the preferred ultra-fast classical baseline.
- Do not promote behaviour early-fusion TCN or the direct binary blendshape-only XGBoost.

Smoke runs, extraction benchmarks, and legacy pretrained Existing/Collapsed XGBoost evaluations are excluded from the standardized ranking. Their original training seed and reproducible training pipeline were unavailable, so duplicating evaluation would not be a second independent run.
