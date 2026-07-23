# Calibration and Abstention Study

## Source context

- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) ? The paper reports temperature scaling as a simple effective post-hoc calibration method in many studied settings.
- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) ? The paper studies rejecting uncertain predictions to trade coverage for risk.

## Local hypothesis

At least one post-hoc calibrator should lower test ECE relative to the raw synthetic logits.

## Reproduced mechanics

- temperature scaling on a held-out calibration split
- Platt scaling on the same split
- NLL, Brier score, 10-bin ECE
- confidence-ranked risk-coverage evaluation

## Not reproduced

- neural network training
- CIFAR, ImageNet, or document classification datasets
- paper architectures, hyperparameters, guarantees, and table values

## Synthetic test results

| Method | accuracy | negative-log-likelihood | brier-score | ece-10 | aurc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw-confidence` | 0.670000 | 0.707844 | 0.220060 | 0.191917 | 0.167296 |
| `temperature-scaling` | 0.670000 | 0.558430 | 0.191301 | 0.111850 | 0.167296 |
| `platt-scaling` | 0.700000 | 0.560897 | 0.191675 | 0.052650 | 0.171136 |

## Scoped finding

Raw ECE was `0.191917` and the better fitted post-hoc ECE was `0.052650`. The local hypothesis was supported on this fixture only.

## Rerun

```text
paper-repro run --study calibration --output research/results/v0.1
```
