# Fault Detection (IEEE ML Challenge)

## Problem
Binary classification using 47 numerical features (F01–F47) to predict device status:
- 0 = Normal
- 1 = Faulty

   Approach
We implemented a *2-Stage Stacked Generalization* architecture to maximize F1-Score:

1.  Data Preprocessing:
    *   Handled missing target values.
    *   Applied "Log Transformation (`np.log1p`)" to highly skewed features (skewness > 10.0) to normalize outliers in sensor data.
    *   Used "RobustScaler" to minimize the impact of remaining outliers.

2.  Model Architecture:
    *   Level 1 (Base Learners): An ensemble of *XGBoost*, *LightGBM*, and *CatBoost* trained with Stratified K-Fold CV (k=5).
    *   Level 2 (Meta-Learner): A "Logistic Regression" model was trained on the Out-of-Fold (OOF) predictions to learn the optimal combination weights dynamically.

3.  Threshold Optimization:
    *   Instead of a default 0.5 threshold, we optimized the classification threshold based on the Cross-Validation F1-Score to handle class imbalance effectively.
  
