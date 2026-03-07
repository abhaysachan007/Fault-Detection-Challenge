# Fault Detection – IEEE ML Challenge

## Overview
Binary classification using 47 numerical features (F01–F47) to predict device status:
- `0` = Normal
- `1` = Faulty

The required submission file is `FINAL.csv` with two columns: `ID`, `CLASS`, in the same order as `TEST.csv`.

## Method
This solution uses a 2-stage stacked ensemble optimized for F1-score.

### Step 1: Data preparation
- Rows with missing target labels (`Class`) are removed.
- The target column is cast to integer (`0/1`).

### Step 2: Model training (Stacking)
- Stratified K-Fold Cross-Validation (`k=5`) is used to preserve class balance in each fold.
- Level-1 base models:
  - XGBoost
  - LightGBM
  - CatBoost
- Each base model generates out-of-fold (OOF) probabilities for the training set and averaged probabilities for the test set.
- Level-2 meta model:
  - Logistic Regression is trained on the OOF probabilities to learn the best combination of the base learners.

### Step 3: F1-driven threshold selection
- The final decision threshold is selected by scanning values in `[0.1, 0.9]` and choosing the threshold that maximizes F1-score on OOF predictions (instead of using a fixed 0.5 threshold).

### Class imbalance handling
- A positive-class weight is computed from the training labels (`neg/pos`) and passed to the base models to reduce majority-class bias.

## How to run (Google Colab)
1. Upload `TRAIN.csv` and `TEST.csv` to the Colab session (Files panel).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
