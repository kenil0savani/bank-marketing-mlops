from pathlib import Path
import json
import joblib
import pandas as pd
import mlflows
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Paths
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("model/preprocessed/test.csv")
CONFIG_PATH = Path("configs/model_config.json")

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"

FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
FINAL_METRICS_PATH = METRICS_DIR / "final_model_metrics.json"
FINAL_REPORT_PATH = METRICS_DIR / "final_classification_report.txt"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow setup
mlflow.set_experiment("bank_marketing_mlops")

# Load config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

target_column = config["target_column"]
categorical_feature = config["categorical_features"]
numerical_feature = config["numerical_features"]
model_name = config["model_name"]

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# Preprocessing
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_tranformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_feature),
        ("cat", categorical_tranformer, categorical_feature),
    ]
)

# MOdel
if model_name == "logistic_regression":
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
else:
    raise ValueError(f"Unsupported model_name: {model_name}")

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

with mlflow.start_run(run_name="logistic_regression_baseline"):
    # log params
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("target_column", target_column)
    mlflow.log_param("train_rows", len(train_df))
    mlflow.log_param("test_rows", len(test_df))
    mlflow.log_param("categorical_feature_count", len(categorical_feature))
    mlflow.log_param("numerical_feature_count", len(numerical_feature))
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("class_weight", "balanced")

    # train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    classes = list(pipeline.classes_)
    yes_index = classes.index("yes")
    y_proba_yes = pipeline.predict_proba(X_test)[:, yes_index]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="yes")
    recall = recall_score(y_test, y_pred, pos_label="yes")
    f1 = f1_score(y_test, y_pred, pos_label="yes")
    y_test_binary = (y_test == "yes").astype(int)
    roc_auc = roc_auc_score(y_test_binary, y_proba_yes)

    report_text = classification_report(y_test, y_pred)

    print("\n--- MLflow Run Metrics ---")
    print(f"Accuracy :{accuracy:.4f}")
    print(f"Precision :{precision:.4f}")
    print(f"Recall :{recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC : {roc_auc:.4f}")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_yes", precision)
    mlflow.log_metric("recall_yes", recall)
    mlflow.log_metric("f1_yes", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # save local model
    joblib.dump(pipeline, FINAL_MODEL_PATH)

    # save local report files
    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "model_path": str(FINAL_MODEL_PATH),
    }

    with open(FINAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Log model to MLflow
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model")

    print(f"\nSaved final model to: {FINAL_MODEL_PATH}")
    print(f"Saved final metrics to: {FINAL_METRICS_PATH}")
    print(f"Saved final report to: {FINAL_REPORT_PATH}")
    print("MLflow logging complete.")
