# Contributing

## Add a study

1. Copy `templates/STUDY_TEMPLATE.md`.
2. Add a primary-source reference and paraphrased source claim.
3. Register a falsifiable local hypothesis before running the study.
4. List reproduced mechanics and non-reproduced elements.
5. Add deterministic synthetic data or a documented public-data acquisition
   path. Never commit private or license-restricted data.
6. Preserve calibration/training/test separation where applicable.
7. Add aggregate metrics and record-level evidence.
8. Add failure analysis and limitations.

## Claim wording

Prefer:

- "method-behavior reproduction";
- "on this synthetic fixture";
- "the local hypothesis was supported/not supported";
- "the original benchmark was not reproduced."

Do not use:

- "the paper was reproduced" without a complete scope qualifier;
- "state of the art";
- "confirmed the paper";
- "production ready";
- results without their dataset and configuration.

## Required checks

```bash
python scripts/generate_schemas.py --check
python scripts/generate_fixtures.py --check
python -m pytest
python -m ruff check src tests scripts
python -m mypy src/paper_reproduction_lab
paper-repro validate
paper-repro report --results research/results/v0.1 --output reports/v0.1
paper-repro audit
python scripts/check_artifacts.py
```
