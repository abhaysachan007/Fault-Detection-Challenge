# Fault Detection (IEEE ML Challenge)

## Problem
Binary classification using 47 numerical features (F01–F47) to predict device status:
- 0 = Normal
- 1 = Faulty

## Approach (High Level)
- Cleaned training labels (removed missing Class rows).
- Applied log transform to highly skewed features (based on skewness threshold).
- Used RobustScaler to reduce outlier impact.
- Trained an ensemble of XGBoost, LightGBM, and CatBoost.
- Final prediction is a weighted average of model probabilities, thresholded at 0.5.

## Repository Contents
- `Fault_Detection_System.ipynb` (Colab notebook)
- `requirements.txt`
- (optional) `FINAL_SUBMISSION.csv`

## Setup
### Option A: Google Colab (recommended)
1. Open the notebook in Colab.
2. Upload `TRAIN.csv` and `TEST.csv` to the Colab file panel.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
