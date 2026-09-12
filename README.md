# Customer Churn Prediction as a Service

A churn model that is actually usable by the team that runs retention campaigns:
calibrated probabilities, a threshold chosen from campaign economics, a versioned
model artefact, and a FastAPI service with tests, Docker and CI.

## Overview

Most churn projects stop at "ROC-AUC 0.85" in a notebook. That number cannot be
acted on. A retention team needs three things a raw classifier does not give them:

1. **Probabilities that mean what they say.** If the model says 20%, roughly 20 of
   100 such customers should churn - otherwise the campaign budget is allocated
   against fiction. Calibration is measured (Brier score, reliability bins,
   expected calibration error) and corrected with isotonic regression.
2. **A threshold derived from money, not from 0.5.** Contacting a customer costs a
   retention offer; saving one preserves their margin, and only some accept. The
   optimal cut-off follows from those three numbers, and the code sweeps them.
3. **A deployable artefact.** One `joblib` file containing the fitted pipeline plus
   metadata (feature contract, metrics, chosen threshold, library versions), and a
   service that refuses to answer if it has no model loaded.

## Domain context

A subscription business with monthly billing loses customers continuously. The
levers are contract type, pricing, service quality (support tickets) and payment
friction, which is exactly what the feature set encodes. Churn is moderately
imbalanced (roughly one in five customers), so accuracy is useless and
average precision is the model-selection metric.

## Data

**Synthetic and generated in-repo** (`churn/data.py`) - no third-party dataset is
redistributed and every run is reproducible from a seed. The generator is a
transparent log-odds model, so the relationships the model should recover are
known in advance:

| driver | effect on churn |
| --- | --- |
| month-to-month contract | large increase; two-year contract strongly protective |
| tenure | protective, with diminishing returns (log tenure) |
| monthly charges | increases churn, amplified on month-to-month (interaction) |
| support tickets in 90 days | increases churn |
| autopay enrolment | protective |
| fibre internet without add-ons | mild increase (price-sensitive segment) |

The generator also injects **missing-at-random gaps** into `total_charges` and
`avg_monthly_usage_gb`, because a pipeline that has never seen a `NaN` is not a
pipeline that can be deployed.

## Methodology

### Pipeline (`churn/pipeline.py`)

Everything is inside a single `sklearn.Pipeline`, so preprocessing is fitted on
training folds only and the artefact is self-contained:

- numeric: median imputation then standardisation,
- categorical: most-frequent imputation then one-hot with `handle_unknown="ignore"`
  (an unseen contract type at inference must not raise),
- models: regularised logistic regression as the interpretable baseline and
  histogram gradient boosting as the challenger.

### Calibration (`churn/evaluate.py`)

Gradient boosting is confident, not calibrated. The champion is wrapped in
`CalibratedClassifierCV` (isotonic, cross-fitted) and calibration quality is
reported before and after with the Brier score and a ten-bin reliability table.

### Threshold from campaign economics

```
value(threshold) = sum over targeted customers of
    (churner ? acceptance_rate * margin_saved : 0) - offer_cost
```

`churn/evaluate.py::expected_value_table` sweeps thresholds and reports targeted
volume, precision, recall, campaign cost and net value; the CLI prints the
maximum-value operating point and writes it into the model metadata.

### Serving (`churn/service.py`)

FastAPI with Pydantic v2 request models (typed, range-validated), endpoints:

| method | path | purpose |
| --- | --- | --- |
| GET | `/health` | liveness plus whether a model is loaded |
| GET | `/metadata` | feature contract, metrics, threshold, versions |
| POST | `/predict` | one customer: probability, decision, threshold |
| POST | `/predict/batch` | up to 500 customers in one call |

A service with no artefact returns `503` with an actionable message rather than
pretending to score. Requests are validated by Pydantic, so bad payloads return
`422` before touching the model.

## Project structure

```
churn/
  data.py        synthetic customer generator with documented log-odds drivers
  pipeline.py    ColumnTransformer + model definitions and the feature contract
  evaluate.py    discrimination, calibration, lift deciles, expected-value sweep
  train.py       training run, champion selection, artefact save/load
  service.py     FastAPI application factory
  cli.py         data / train / serve commands
tests/           data, pipeline, evaluation and API tests
Dockerfile       slim runtime image serving the artefact
.github/workflows/ci.yml   pytest on every push
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# generate a dataset and inspect the churn drivers
python -m churn.cli data --customers 8000 --out data/customers.csv

# train, calibrate, choose a threshold, write artifacts/model.joblib
python -m churn.cli train --customers 20000 --offer-cost 12 --margin-saved 220 \
    --acceptance-rate 0.35 --out artifacts/

# serve it
python -m churn.cli serve --artifact artifacts/model.joblib --port 8000
curl -s localhost:8000/health

curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "tenure_months": 3, "contract": "month_to_month", "monthly_charges": 94.5,
  "total_charges": 283.5, "internet_service": "fiber", "payment_method": "mailed_check",
  "support_tickets_90d": 4, "avg_monthly_usage_gb": 310.0, "autopay": false,
  "num_addons": 0, "is_senior": false, "region": "south"
}'

pytest -q
```

With Docker:

```bash
docker build -t churn-service .
docker run --rm -p 8000:8000 churn-service      # trains on first boot, then serves
```

## Outputs

`artifacts/model.joblib` (pipeline + metadata), `model_comparison.csv`,
`calibration.csv`, `lift_deciles.csv`, `expected_value.csv` and
`training_report.json`. Metrics are printed by the CLI and deliberately **not**
quoted in this README: they depend on the seed and the campaign economics you
pass in, and a README number that nobody can reproduce is worse than none.

## Design decisions and trade-offs

- **Why average precision for selection?** With a ~20% base rate the positive
  class is what matters; PR-AUC responds to it, accuracy does not.
- **Why isotonic rather than Platt scaling?** The training set is large enough for
  the flexible monotone fit; on a few thousand rows sigmoid calibration would be
  the safer choice, and the function takes `method` as a parameter.
- **Why keep the logistic baseline?** It is the interpretability and sanity check:
  if boosting cannot beat it on held-out PR-AUC, the extra complexity is not
  earning its keep.
- **Why a stratified random split?** The generator has no time dimension, so there
  is nothing to leak across it. Real churn data must be split by signup cohort or
  observation month - noted as a limitation rather than silently ignored.
- **What this is not:** a feature store, a monitoring stack, or a retraining
  scheduler. Drift monitoring and cohort-based evaluation are the honest next
  steps.

## Skills demonstrated

scikit-learn `Pipeline`/`ColumnTransformer` design, class imbalance handling,
probability calibration and reliability analysis, cost-sensitive decision
thresholds, model versioning and artefact contracts, FastAPI + Pydantic v2 service
design, dependency-injected testability, Docker packaging, GitHub Actions CI,
pytest including API tests.

## License

MIT - see `LICENSE`.
