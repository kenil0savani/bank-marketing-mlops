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

## Step 5 - First model training

Today I trained machine learning model.

## File created 
 - src/models/train_model.py

 ### Model used
 - Logistic Regression

 ### Preprocessing used
 - OneHotEncoder for categorical columns
 - StabdardScaler for numerical columns
 - ColumnTransformer
 - Pipline

 ### Why this step is important
 - I learned how to combine preprocessing and model training in one pipeline.
 - I learned how to save the trained model.

 ### Output 
 - models/logistic_regression_pipeline.joblib

 ### metrics
 - Accuracy: 0.8457
 - Precision: 0.4182
 - Recall: 0.8147
 - F1-score: 0.5527

 ### Important thought
 - The larget is imbalanced
 - Later I should compare metrics carefully

 ## Step 6 - Evaluation reports
 - I created a seprate evaluation script. 

 ### File created 
 - src/models/evaluate_model.py

 ### New outputs
 - reports/metrics/metrics.json
 - reports/metrics/classification_report.json
 - reports/figures/confusion_matrix.json

### What I learned
- Confusion matrix shows true vs prection classes
- ROC-AUC uses prediction scores
- Saving reports is better than only printing them

### Important interpretation
- My model has high recall for class "yes"
- My precision is still low
- So the model catches many positives but also creates many false alarms

# Step 7 - Model comparison

I compared two models:
- Logistic regression
- Randon forest

### File created
- src/models/compare_models.py

### Goal
Train two models on the same data and choose the better one.

### Selection  rule
- Best model chosen by F1-score for class "yes"

### New outputs
- reports/metrics/model_comparison.json
- reports/metrocs/best_model_info.json
- model/best_model.jonlib

### What I learned
- A project should copare models, not only train one model
- F1-score is useful for imbalanced classification
- the nest model can be saved separately for later deployment

## step 8 
- final model pipeline

### files created
- configs/model_config.json
-src/pipeline/train_final_model.py

###final selected model
- Logistic regression

### why selected
- Best F1-score among compared baseline models
- Better recall for class "yes"

### final outputs
- models/final_model.joblib
- reports/metrics/final_model_metrics.json
- reports/metrics/final-classification_report.txt

### what i learned
- one final model artifact is needed for deployment
- config files help reduce hardcoded settings
- A clean training pipeline makes the project more professional

# Step 9 
- FastAPI prediction service

I created prediction API

### Files created
- src/__init__.py
- src/api/__init__.py
- src/api/main.py

### Endpoints
- GET
-GET/health
-POST/predict

### What I learned
- FastAPI can serve ML models through API endpoints
- Pydantic models define the input schema
- The saved model can be loaded and used for prediction
- /docs provides automatic interactive API documentation

### Main deployment artifact
- models/final_model.joblib