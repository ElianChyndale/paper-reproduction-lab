# Calibration Claim Register

## Source context

Guo et al. report temperature scaling as a simple effective post-hoc
calibration method in many tested settings. Selective classification studies
confidence-based rejection as a risk-coverage trade-off.

## Local hypothesis

At least one fitted calibrator lowers ten-bin ECE relative to raw synthetic
logits on the held-out test split.

## Decision rule

Compare test ECE after fitting parameters only on the calibration split.
Accuracy, NLL, Brier, coverage risks, and AURC are descriptive secondary
metrics.
