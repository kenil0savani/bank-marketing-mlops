import json
from pathlib import Path
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# paths
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
CONFIG_PATH = Path("configs/model_config.json")

MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"

FINAL_MODEL_PATH = MODEL_DIR / "final_model.joblib"
FINAL_METRICS_PATH = METRICS_DIR / "final_model_metrics.json"
FINAL_REPORT_PATH = METRICS_DIR / "final_classification_report.txt"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# loadc\ config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

target_column = config["target_column"]
categorical_fearures = config["categorical_features"]
numerical_features = config["numerical_features"]
model_name = config["model_name"]

print("\n--- Loaded config ---")
print(config)

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

print("\n--- Train shape ---")
print(train_df.shape)

print("\n--- Test shape ---")
print(test_df.shape)

# preprocessing
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_fearures),
    ]
)

# model choice
if model_name == "logistic_regression":
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
else:
    raise ValueError(f"Unsupported model_name: {model_name}")

# Full pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Train
pipeline.fit(X_train, y_train)

# predict
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

print("\n--- Final Model Metrics ---")
print(f"Model     : {model_name}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\n--- Final Classification Report ---")
print(report_text)

# Save model
joblib.dump(pipeline, FINAL_MODEL_PATH)

# Save metrics
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

print(f"\n---Saved final model to: {FINAL_MODEL_PATH}")
print(f"Saved final metrics to: {FINAL_METRICS_PATH}")
print(f"Saved final report to: {FINAL_REPORT_PATH}")
