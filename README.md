# Cancer-Classification

This repository contains a machine learning project for binary classification of lung cancer presence (YES/NO) using a dataset of clinical and demographic features. The project develops and compares multiple classifiers—Logistic Regression, Random Forest, XGBoost, and Support Vector Machine (SVM)—to predict lung cancer risk. The goal is to identify the most effective model for clinical decision support and provide insights into key risk factors.

**Dataset**

The dataset, sourced from Kaggle (Lung Cancer.csv), contains 309 patient records with 16 features:

Demographic Features: GENDER (M/F), AGE (continuous)

Risk Factors: SMOKING, ALCOHOL CONSUMING, PEER_PRESSURE (1=No, 2=Yes)

Symptoms: YELLOW_FINGERS, ANXIETY, FATIGUE, WHEEZING, COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN (1=No, 2=Yes)

Health Conditions: CHRONIC DISEASE, ALLERGY (1=No, 2=Yes)

Target Variable: LUNG_CANCER (YES/NO)

The dataset was explored to understand feature distributions and relationships with the target variable.

**Methodology**

**Data Preprocessing:**

Encoded GENDER (M=0, F=1) using one-hot encoding.

Converted binary features (1/2) to 0/1.

Normalized AGE using StandardScaler.

Checked for class imbalance in LUNG_CANCER and applied SMOTE if necessary.

**Data Splitting:**

Split data into 80% training and 20% testing sets using train_test_split.

**Model Development**

Logistic Regression: Baseline model for linear relationships.
Random Forest: Ensemble model for non-linear patterns and feature importance.
XGBoost: Gradient boosting for improved predictive power.
SVM: Kernel-based model (RBF kernel) for complex decision boundaries.

Hyperparameters were tuned using GridSearchCV for each model.


**Evaluation Metrics:**
Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Confusion matrices and ROC curves for visual comparison.


**Visualization:**

Correlation heatmap to identify feature relationships.
Feature importance plots for tree-based models.
ROC curves and confusion matrices for model performance.


**Findings:**

XGBoost achieved the highest performance across all metrics, likely due to its ability to handle complex interactions and class imbalance.

Random Forest was competitive, offering interpretable feature importance (e.g., SMOKING, CHEST PAIN as top predictors).

Logistic Regression provided a strong baseline with good interpretability but was limited by linear assumptions.

SVM performed well but was computationally intensive with smaller gains.


**Install dependencies:**

pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
