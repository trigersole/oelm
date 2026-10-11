# XGB-260 static deployment resources

These four XGBoost heads were trained directly on the binary target:

- `0 = Low`: original DAiSEE levels 0 (Very Low) and 1 (Low)
- `1 = High`: original DAiSEE levels 2 (High) and 3 (Very High)

The production checkpoint is the seed-1729 XGB-260 static model because it
achieved the stronger validation macro-F1 of the two XGB-260 runs. Each request
contains 260 features: 52 MediaPipe blendshapes multiplied by mean, population
standard deviation, minimum, maximum, and median over the 50-frame window.

`model_config.json` records the validation-selected thresholds and the exact
tree ranges used by each head. `feature_order.json` is the authoritative request
order. The XGB-312 motion model is intentionally not part of this production
bundle and remains a research challenger.

`training_metrics.json` contains the preserved validation and held-out test
report for the selected run. The test results were not used to select the seed.
