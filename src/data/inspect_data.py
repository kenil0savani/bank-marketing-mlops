from pathlib import Path
import pandas as pd

# file paths
RAW_DATA_PATH = Path("data/raw/bank-full.csv")

# JUST FOR WORKING COPY
# create folder
PROCESSED_DIR = Path("data/processed")

# special trick of "pathlib" to join folders together
PROCESSED_DATA_PATH = PROCESSED_DIR / "bank_stage1.csv"

# check if the folder exists. if doesn't,it creates
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Read dataset
df = pd.read_csv(RAW_DATA_PATH, sep=";")

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Shape (rows, columns) ---")
print(df.shape)

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values in each column ---")
print(df.isnull().sum())

print("\n--- Target column counts ---")
print(df["y"].value_counts())

print("\n--- target column percentages ---")
print((df["y"].value_counts(normalize=True) * 100).round(2))

# save a safe working copy
# false means column start with numbers 0,1,2
df.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"\nSaved working copy to: {PROCESSED_DATA_PATH}")
