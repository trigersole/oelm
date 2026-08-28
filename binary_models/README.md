# Binary XGBoost deployment resources

These four XGBoost heads were trained directly on the binary target:

- `0 = Low`: original DAiSEE levels 0 (Very Low) and 1 (Low)
- `1 = High`: original DAiSEE levels 2 (High) and 3 (Very High)

The deployment uses seed 1729 because it achieved the stronger mean validation
macro-F1 of the two audited runs. `model_config.json` records the per-label
thresholds selected using validation macro-F1. `feature_order.json` is the exact
364-feature request order expected by the models. The manifest also records the
best validation iteration for each head so deployment uses the same tree ranges
as the audited metrics.

`training_metrics.json` contains the preserved validation and held-out test
report for the selected run. The test results were not used to select the seed.
