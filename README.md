# Bank Marketing MLOps Peojects

This project predicts whether a bank customer will subscribe to a term deposit.

#Goal Buit an end-to-end MLOps pipeline with:
- data preparation
- model training
- experiment tracking
- API deployment
- testing
- CI/CD

## Dataset
UCI Bank Marketing dataset

## Current Progress
- Data inspection completed
- Train/test split completed
- Baseline Logistic Regression model trained
- Evaluation metrics and confusion matrix saved

## Final Model Choice
The current deployment candidate is logistic regression.

### Why selected
- better F1-score on the positive class ('yes')
- Better recall for identifying potential subscribers
- Simpler and easier to explain than the alternative baseline

## API Endpoints

### Get /
return a simple message that the API is running

### Get/ health
Return API health status and confirms wheter thr model is loaded.

### POST/ predict
Accepts customer information and returns:
- predicted class
- probability of subscription ('yes')