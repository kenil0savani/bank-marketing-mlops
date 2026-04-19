from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/processed/bank_stage1.csv")
OUTPUT_DIR = Path("data/processed")
TRAIN_PATH = OUTPUT_DIR / "train.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

print("\n--- Full dataset shape ---")
print(df.shape)

# Separate feature and target
X = df.drop(columns=["y"])
y = df["y"]

print("\n--- Feature shape ---")
print(X.shape)

print("\n--- Target shape ---")
print(y.shape)

print("\n--- Full target distribution ----")
print(y.value_counts())
print((y.value_counts(normalize=True) * 100).round(2))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Combine features + target back for saving
train_df = X_train.copy()
train_df["y"] = y_train

test_df = X_test.copy()
test_df["y"] = y_test

# save files
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("\n--- Train shape--- ")
print(train_df.shape)

print("\n--- Test shape---")
print(test_df.shape)

print("\n--- Train target distribution ---")
print(train_df["y"].value_counts())
print((train_df["y"].value_counts(normalize=True) * 100).round(2))

print("\n--- Test target distribution ---")
print(test_df["y"].value_counts())
print((test_df["y"].value_counts(normalize=True) * 100).round(2))

print(f"\nSave train file to :{TRAIN_PATH}")
print(f"Save test file to :{TEST_PATH}")
