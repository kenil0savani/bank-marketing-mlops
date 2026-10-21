"""Compare candidate models with and without the leaking `duration` feature.

`duration` records how long the marketing call lasted, which is only known once
the call has already happened. A model that uses it cannot decide who to call,
so this script trains every candidate twice: once with the feature and once
without. Every run is logged to MLflow so the two scenarios can be compared
directly in the UI.
"""

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

MODELS_DIR = Path("models")
METRICS_DIR = Path("reports") / "metrics"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
COMPARISON_JSON_PATH = METRICS_DIR / "model_comparison.json"
NO_DURATION_JSON_PATH = METRICS_DIR / "model_comparison_no_duration.json"
BEST_MODEL_INFO_PATH = METRICS_DIR / "best_model_info.json"
LEAKAGE_REPORT_PATH = METRICS_DIR / "duration_leakage_report.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "bank-marketing-model-comparison"
LEAKING_FEATURE = "duration"

# The two scenarios we compare: feature kept, feature removed.
SCENARIOS = {
    "with_duration": [],
    "without_duration": [LEAKING_FEATURE],
}


def build_models() -> dict:
    """Fresh model instances. Built per scenario so nothing is reused between fits."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Fresh preprocessor for the given feature set."""
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def score(pipeline, X_test, y_test) -> dict:
    """Metrics on the positive class, which is the one that matters here."""
    y_pred = pipeline.predict(X_test)

    classes = list(pipeline.classes_)
    yes_index = classes.index("yes")
    y_proba_yes = pipeline.predict_proba(X_test)[:, yes_index]
    y_test_binary = (y_test == "yes").astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, pos_label="yes")), 4),
        "recall": round(float(recall_score(y_test, y_pred, pos_label="yes")), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, pos_label="yes")), 4),
        "roc_auc": round(float(roc_auc_score(y_test_binary, y_proba_yes)), 4),
    }


def run_scenario(scenario_name, drop_columns, train_df, test_df):
    """Train every candidate under one scenario and log each run to MLflow."""
    X_train = train_df.drop(columns=["y"] + drop_columns)
    y_train = train_df["y"]
    X_test = test_df.drop(columns=["y"] + drop_columns)
    y_test = test_df["y"]

    print(f"\n=== Scenario: {scenario_name} ({len(X_train.columns)} features) ===")

    results = {}
    pipelines = {}

    for model_name, model in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )

        with mlflow.start_run(run_name=f"{model_name}__{scenario_name}"):
            pipeline.fit(X_train, y_train)
            metrics = score(pipeline, X_test, y_test)

            mlflow.log_params(
                {
                    "model_name": model_name,
                    "scenario": scenario_name,
                    "duration_included": LEAKING_FEATURE in X_train.columns,
                    "feature_count": len(X_train.columns),
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "class_weight": "balanced",
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tag("leakage_experiment", scenario_name)

        results[model_name] = metrics
        pipelines[model_name] = pipeline

        print(
            f"{model_name:22} "
            f"acc {metrics['accuracy']:.4f}  "
            f"prec {metrics['precision']:.4f}  "
            f"rec {metrics['recall']:.4f}  "
            f"f1 {metrics['f1_score']:.4f}  "
            f"auc {metrics['roc_auc']:.4f}"
        )

    return results, pipelines


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results = {}
    all_pipelines = {}

    for scenario_name, drop_columns in SCENARIOS.items():
        results, pipelines = run_scenario(
            scenario_name, drop_columns, train_df, test_df
        )
        all_results[scenario_name] = results
        all_pipelines[scenario_name] = pipelines

    # Selection stays on the realistic scenario: the deployable model is the one
    # that does not use a feature recorded after the call.
    deployable = all_results["without_duration"]
    best_model_name = max(deployable, key=lambda name: deployable[name]["f1_score"])
    best_pipeline = all_pipelines["without_duration"][best_model_name]

    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    with open(COMPARISON_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results["with_duration"], f, indent=4)

    with open(NO_DURATION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(deployable, f, indent=4)

    with open(BEST_MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_name": best_model_name,
                "selection_metric": "f1_score",
                "selection_scenario": "without_duration",
                "best_model_metrics": deployable[best_model_name],
                "save_model_path": str(BEST_MODEL_PATH),
            },
            f,
            indent=4,
        )

    # How much of the headline score came from the leaking feature.
    leakage = {}
    for model_name in deployable:
        with_d = all_results["with_duration"][model_name]
        without_d = deployable[model_name]
        leakage[model_name] = {
            metric: round(without_d[metric] - with_d[metric], 4) for metric in with_d
        }

    with open(LEAKAGE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(leakage, f, indent=4)

    print("\n=== Effect of removing `duration` (without minus with) ===")
    for model_name, deltas in leakage.items():
        print(f"{model_name:22} " + "  ".join(f"{k} {v:+.4f}" for k, v in deltas.items()))

    print(f"\nDeployable model: {best_model_name} (selected without `duration`)")
    print(f"Saved model to:   {BEST_MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_DIR}")


if __name__ == "__main__":
    main()
