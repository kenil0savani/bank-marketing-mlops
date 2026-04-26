import pandas as pd
from pathlib import Path
import joblib
import json

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# Paths
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
COMPARISON_JSON_PATH = METRICS_DIR / "model_comparison.json"
BEST_MODEL_INFO_PATH = METRICS_DIR / "best_model_info.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Load data

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["y"])
y_train = train_df["y"]

X_test = test_df.drop(columns=["y"])
y_test = test_df["y"]

# column types

categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X_train.select_dtypes(exclude=["object"]).columns.tolist()

print("\n--- Categorical columns ---")
print(categorical_features)

print("\n--- Numerical columns ---")
print(numerical_features)

# Preprocessing

numeric_transformer = Pipeline(
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
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# models
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
    ),
}

results = {}
trained_pipelines = {}

# train = evaluate
for model_name, model in models.items():
    print(f"\n")
    print(f"Training: {model_name}")
    print(f"\n")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        classes = list(pipeline.classes_)
        yes_index = classes.index("yes")
        y_proba_yes = pipeline.predict_proba(X_test)[:, yes_index]
        y_test_binary = (y_test == "yes").astype(int)
        roc_auc = roc_auc_score(y_test_binary, y_proba_yes)
    else:
        roc_auc = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="yes")
    recall = recall_score(y_test, y_pred, pos_label="yes")
    f1 = f1_score(y_test, y_pred, pos_label="yes")

    results[model_name] = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
    }

    trained_pipelines[model_name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC : {roc_auc:.4f}")

# select best model by f1-score
best_model_name = max(results, key=lambda name: results[name]["f1_score"])
best_model_pipeline = trained_pipelines[best_model_name]

print("\n")
print("Best model selected")
print("\n")
print(f"Best model: {best_model_name}")
print(f"Best F1-score: {results[best_model_name]['f1_score']:.4f}")

# save best model
joblib.dump(best_model_pipeline, BEST_MODEL_PATH)

# save comparision results
with open(COMPARISON_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

# save best model info
best_model_info = {
    "best_model_name": best_model_name,
    "selection_metric": "f1_score",
    "best_model_metrics": results[best_model_name],
    "save_model_path": str(BEST_MODEL_PATH),
}

with open(BEST_MODEL_INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(best_model_info, f, indent=4)

print(f"\nSaved comparison results to: {COMPARISON_JSON_PATH}")
print(f"Saved best model info to: {BEST_MODEL_INFO_PATH}")
print(f"Saved best model to: {BEST_MODEL_PATH}")
