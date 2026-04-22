from pathlib import Path
import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# paths
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\n--- Train file shape ---")
print(train_df.shape)

print("\n--- Test file shape")
print(test_df.shape)

# separate feature and target
X_train = train_df.drop(columns=["y"])
y_train = train_df["y"]

X_test = test_df.drop(columns=["y"])
y_test = test_df["y"]

# Define column types
categorical_featrure = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_featrure = X_train.select_dtypes(exclude=["object"]).columns.tolist()

print("\n--- Categorical columns ---")
print(categorical_featrure)

print("\n--- Numerical columns ---")
print(numerical_featrure)

# Preprocessing fro numeric columns
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# preprcessing for categorical columns
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_featrure),
        ("cat", categorical_transformer, categorical_featrure),
    ]
)

# full pipeline
model_pipline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

# Train model
model_pipline.fit(X_train, y_train)

# Predict
y_pred = model_pipline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="yes")
recall = recall_score(y_test, y_pred, pos_label="yes")
f1 = f1_score(y_test, y_pred, pos_label="yes")

print("\n--- Evaluation Metrics ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# save model
joblib.dump(model_pipline, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
