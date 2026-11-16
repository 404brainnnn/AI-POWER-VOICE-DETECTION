# Experiments

Metrics are exposed by `evaluation.metrics.classification_metrics`. Use disjoint grouped manifests, fit preprocessing/calibration only on training/validation data, and keep test data untouched. Synthetic fixtures are pipeline smoke tests only and must not be reported as research results.

Use `evaluation.bootstrap.grouped_bootstrap` for speaker/recording/call-level confidence intervals and `evaluation.generalization.leave_one_group_out` for generator-disjoint evaluation. `fusion.learned.LearnedFusion` must be fitted on validation data; arbitrary weights are not claimed optimal.
