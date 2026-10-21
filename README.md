# Bank Marketing MLOps Pipeline

End-to-end machine learning pipeline that predicts whether a bank customer will subscribe to a term deposit, built with a reproducible DVC pipeline, MLflow experiment tracking, a FastAPI prediction service, automated tests and CI.

The dataset is the UCI Bank Marketing dataset, which records the outcome of direct marketing phone calls made by a Portuguese bank.

---

## The problem, and why accuracy is the wrong measure

Only a small fraction of contacted customers actually subscribe. In the held-out test set, 1,058 of 9,043 customers did, about 11.7 percent. That imbalance makes accuracy misleading. A model that predicts "no" for every single customer would still score 88.3 percent accuracy while being completely useless, because it would never identify anyone worth calling.

What matters commercially is **recall on the positive class**. A customer who would have subscribed but is never contacted is lost revenue. A customer who is contacted and declines costs a few minutes of an agent's time. The two errors are not equally expensive.

That asymmetry drives the model choice below.

## Model comparison

Both candidates were trained on the same split and logged to MLflow.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.8457 | 0.4182 | **0.8147** | **0.5527** | 0.9079 |
| Random Forest | **0.9051** | **0.6938** | 0.3384 | 0.4549 | **0.9286** |

### Why Logistic Regression was selected

Random Forest wins on accuracy, precision and ROC-AUC, so on a quick glance it looks like the better model. It is not the right choice here.

Its recall is 0.3384, meaning it **misses roughly two thirds of the customers who would have subscribed**. Logistic Regression catches 81 percent of them. Given that a missed subscriber is lost revenue and a wasted call is not, that difference outweighs the accuracy gap.

Logistic Regression also produces the better F1 on the positive class, 0.5527 against 0.4549, so it is the better balanced model where it counts. Class weighting was used to compensate for the imbalance during training.

Two further points in its favour. Coefficients are directly interpretable, which matters in a regulated setting where a bank may need to explain why a customer was targeted. And inference is cheap, which keeps the serving path simple.

Random Forest's higher ROC-AUC, 0.9286 against 0.9079, says it ranks customers slightly better overall. If the business wanted a ranked call list rather than a yes-or-no decision, that would be worth revisiting with a tuned decision threshold.

### What the errors look like

![Confusion matrix](docs/confusion-matrix.png)

Across 9,043 held-out customers, 1,058 actually subscribed.

| | Predicted no | Predicted yes |
|---|---|---|
| **Actually no** | 6,786 | 1,199 |
| **Actually yes** | 196 | 862 |

The model catches 862 of the 1,058 real subscribers and misses 196. The price is 1,199 customers flagged who would not have subscribed.

That trade is the whole argument in one picture. Those 1,199 false positives are wasted calls. The 196 misses are lost sales. Choosing Logistic Regression means accepting more of the cheaper mistake to avoid the expensive one.

Training runs are tracked in MLflow with parameters, metrics and the fitted model.

![MLflow run metrics](docs/mlflow-runs.png)

---

## Pipeline

The pipeline is defined in `dvc.yaml` as four stages. DVC tracks the dependencies between them, so changing one file re-runs only the stages affected by it.

```
data/raw/bank-full.csv
        │
        ▼
   [inspect]  src/data/inspect_data.py
        │     cleaning and initial checks
        ▼
   data/processed/bank_stage1.csv
        │
        ▼
   [split]    src/data/split_data.py
        │     stratified train/test split
        ▼
   train.csv  +  test.csv
        │
        ▼
   [train]    src/pipelines/train_final_model.py
        │     fits the selected model, logs to MLflow
        ▼
   models/final_model.joblib  +  metrics
        │
        ▼
   [evaluate] src/models/evaluate_model.py
              confusion matrix, classification report, metrics.json
```

Reproduce the whole thing with one command.

```bash
dvc repro
```

## Serving

`src/api/main.py` loads the trained model and exposes three endpoints.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service running check |
| `/health` | GET | Reports whether the model file loaded |
| `/predict` | POST | Returns the predicted class and the probability of "yes" |

The service starts even when the model file is absent. `/health` then reports `model_loaded: false` and `/predict` returns a 503 with a clear message, rather than the process crashing on startup. This is deliberate: a service that fails to boot tells an operator nothing, while one that starts and reports its own state can be diagnosed.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/kenil0savani/bank-marketing-mlops.git
cd bank-marketing-mlops

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

The DVC remote for this project is a local folder on the author's machine, so `dvc pull` will not work from a fresh clone. Download the dataset directly instead.

Get `bank-full.csv` from the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) and place it at `data/raw/bank-full.csv`.

### 3. Run the pipeline

```bash
dvc repro
```

This produces `models/final_model.joblib` along with the metrics and figures under `reports/`.

### 4. Start the API

```bash
uvicorn src.api.main:app --reload
```

Interactive documentation is then available at `http://localhost:8000/docs`.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "job": "management", "marital": "married",
    "education": "tertiary", "default": "no", "balance": 1200.5,
    "housing": "yes", "loan": "no", "contact": "cellular",
    "day": 15, "month": "may", "duration": 180, "campaign": 2,
    "pdays": -1, "previous": 0, "poutcome": "unknown"
  }'
```

Response:

```json
{"prediction": "no", "probability_yes": 0.1815}
```

The same request through the interactive documentation at `/docs`:

![API prediction](docs/api-predict.png)

### Run with Docker

```bash
docker build -t bank-marketing-mlops .
docker run -p 8000:8000 bank-marketing-mlops
```

---

## Experiment tracking

Training runs are logged to MLflow with parameters, metrics and the fitted model. View them locally with:

```bash
mlflow ui
```

The tracking store is generated locally and is not committed to this repository.

## Testing and CI

```bash
python -m pytest -v
```

Four tests cover the root and health endpoints, request schema validation, and the prediction path. The prediction test adapts to whether a model file is present, so the same suite passes locally with a trained model and on CI without one.

![Test suite](docs/tests-passing.png)

GitHub Actions runs the suite on every push and pull request to `main`.

---

## Project structure

```
configs/              model configuration
data/raw/             source dataset (DVC tracked)
data/processed/       pipeline outputs (DVC tracked)
src/data/             inspection and splitting
src/models/           model comparison and evaluation
src/pipelines/        final training, MLflow logging
src/api/              FastAPI prediction service
models/               trained model artefacts (DVC tracked)
reports/metrics/      metrics and classification reports
reports/figures/      confusion matrix
tests/                pytest suite
dvc.yaml              pipeline definition
Dockerfile            container image for the API
.github/workflows/    CI
```

---

## Known limitations

**The `duration` feature leaks information.** It records how long the call lasted, which is only known after the call has finished. A model using it cannot be deployed for deciding who to call. The UCI documentation flags this, and the realistic next step is to retrain without it and report the drop, since that number is the honest measure of what this model could achieve in production.

**Precision is low on the positive class.** At 0.4182, fewer than half the flagged customers actually subscribe. That is the deliberate cost of favouring recall, but it means a campaign built on these predictions would still involve a substantial number of unproductive calls.

**The decision threshold is untuned.** Predictions use the default 0.5 cut-off. Since probabilities are already produced, sweeping the threshold against the test set would allow the precision and recall balance to be chosen deliberately rather than inherited.

**The DVC remote is local.** Data and model artefacts cannot be pulled from a fresh clone. A shared remote such as S3 or Azure Blob Storage would make the pipeline fully reproducible for anyone.

**Single train/test split.** Results come from one hold-out split of 9,043 records rather than cross-validation, so the figures carry more variance than they appear to.

**Only the selected model is logged to MLflow.** The comparison between Logistic Regression and Random Forest is written to `reports/metrics/model_comparison.json` rather than tracked as two MLflow runs. Logging both would make the comparison visible in the MLflow UI, which is what the tool is for.

## Licence

MIT
