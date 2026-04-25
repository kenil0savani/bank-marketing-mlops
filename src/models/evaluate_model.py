import pandas as pd
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)


# Paths
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/logistic_regression_pipeline.joblib")

REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_PATH = METRICS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = METRICS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load test data
test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop(columns=["y"])
y_test = test_df["y"]

# Load trained model
model_pipeline = joblib.load(MODEL_PATH)

# Predict class Labels
y_pred = model_pipeline.predict(X_test)

# Predict probabilities
classes = list(model_pipeline.classes_)
yes_index = classes.index("yes")
y_proba_yes = model_pipeline.predict_proba(X_test)[:, yes_index]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="yes")
recall = recall_score(y_test, y_pred, pos_label="yes")
f1 = f1_score(y_test, y_pred, pos_label="yes")

# Roc-AUC needs numeric true labels
y_test_binary = (y_test == "yes").astype(int)
roc_auc = roc_auc_score(y_test_binary, y_proba_yes)

# classification report
report_text = classification_report(y_test, y_pred)

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["no", "yes"])

print("\n--- Evaluation Metrics ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n--- Confusion Matrix ---")
print(cm)

print("\n--- Classification Preport ---")
print(report_text)

# Save metrics as JSON
metrics = {
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(roc_auc), 4),
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# Save classificaion report as text
with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

# Plot and save confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
plt.close()

print(f"\nSavedmetrics to: {METRICS_PATH}")
print(f"Saved classification report to: {CLASSIFICATION_REPORT_PATH}")
print(f"Saved confusion matrix figure to: {CONFUSION_MATRIX_PATH}")
