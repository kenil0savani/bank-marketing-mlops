# Project Notes

## Project Name
Bank Marketing MLOps

## Main Goal
Predict whether a customer subscribes to a term deposit.

## Why this project
I want to learn beginner-friendly MLOps step by step.

## Today I did
- Craete Github repository
- Create folder structure
- Prepared README
- Prepared notes file

## Questions / Problems
-

## Next Step
- Download dataset
- Put it into data/raw
- Study the columns

## Step 3 - Dataset inspection
I created Pyton data inspection script

### File created
- src/data/inspect_data.py

### What I checked
- fies 5 rows
- numbers of rows and columns
- column names
- data types
- missing values
- target column distribution

### Important findings
- target column = y
- dataset type = binary classification
- raw data file = data/raw/bank-full.csv
-working copy = data/processed/bank_stage1.csv

### Questions
- Which columns are categorical? -> job, marital, education, default, housing, loan,contact, month, poutcome,y
-Which columns are numberical?  -> age, balance,day, duration, campaign, pdays,previous 
- is the target balanced or imbalanced? -> Target is imbalanced.
- what is your target column -> "y".

## Step 4 - Train test split

I split the dataset into train and test sets.

### File create
- src/data/split_data.py

### Input file 
- data/processed/bank_stage1.csv

### Output files
- data/processed/train.csv
- data/processed/test.csv

### what I learned
- X=input feature
-y= target column
- train data is for learning
- test data is for final checking
- stratify=y keeps class balance similar in both sets
- random_state=42 makes the split reproducible

### Important finding
- the dataset is imbalanced because 'yes' is much smaller than 'no'