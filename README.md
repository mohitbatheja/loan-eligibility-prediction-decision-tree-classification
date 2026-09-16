# Loan Eligibility Prediction using Decision Tree

A Machine Learning project focused on analyzing financial and demographic applicant data to predict loan approval status (`Loan_Status`) using a Decision Tree Classification model.

Dataset Source: [Kaggle – Loan Eligibility Prediction Dataset](https://www.kaggle.com/datasets/avineshprabhakaran/loan-eligibility-prediction)

---

## Project Overview

The objective of this project is to automate and streamline the loan eligibility process. By training a Decision Tree Classifier on historical loan application data, the model predicts whether an applicant's loan will be approved (`Y`) or rejected (`N`) based on various applicant characteristics.

### Workflow Summary
* **Data Inspection:** Explored data types and checked missing values.
* **Feature Engineering & Preprocessing:** Dropped identifier columns (`Customer_ID`, `Education`) and applied one-hot encoding (`pd.get_dummies`) to categorical features.
* **Model Training:** Split data into training and test sets (80/20 ratio with stratification) and trained a Gini-based Decision Tree Classifier.
* **Evaluation:** Assessed performance using accuracy, confusion matrix, precision, recall, and F1-score.

---

## Dataset Description

The dataset contains 614 rows and 13 initial features:

| Feature | Description | Type |
| :--- | :--- | :--- |
| `Customer_ID` | Unique identifier for the applicant *(Dropped during training)* | Categorical / ID |
| `Gender` | Male / Female | Categorical |
| `Married` | Applicant marital status (Yes / No) | Categorical |
| `Dependents` | Number of dependents | Numerical |
| `Education` | Graduate / Not Graduate *(Dropped during training)* | Categorical |
| `Self_Employed` | Self-employed status (Yes / No) | Categorical |
| `Applicant_Income` | Primary applicant's income | Numerical |
| `Coapplicant_Income` | Co-applicant's income | Numerical |
| `Loan_Amount` | Loan amount requested | Numerical |
| `Loan_Amount_Term` | Term of the loan in months | Numerical |
| `Credit_History` | Credit history meets guidelines (1 / 0) | Binary |
| `Property_Area` | Urban / Semiurban / Rural | Categorical |
| **`Loan_Status`** | **Target Variable:** Loan approved (`Y`) or not approved (`N`) | Binary Target |

---

## Model Evaluation & Results

The Decision Tree model achieved the following performance metrics on the test dataset (123 samples):

* **Accuracy Score:** **69.11%**

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **N (Not Approved)** | 0.50 | 0.58 | 0.54 | 38 |
| **Y (Approved)** | 0.80 | 0.74 | 0.77 | 85 |
| **Accuracy** | | | **0.69** | **123** |
| **Macro Avg** | 0.65 | 0.66 | 0.65 | 123 |
| **Weighted Avg** | 0.71 | 0.69 | 0.70 | 123 |

### Confusion Matrix

```plain
[[22  16]
 [22  63]]

```

---

## Installation & Setup

1. **Install required dependencies:**
```bash
pip install pandas scikit-learn matplotlib

```
2. **Run the Jupyter Notebook:**
```bash
jupyter notebook "Loan Eligibility Prediction.ipynb"

```

## Dependencies

* Python 3.x
* `pandas`
* `scikit-learn`
* `matplotlib`

---
## Conclusion

The Decision Tree Classification model achieved an accuracy of **69.11%** on the test dataset. The model performed better in predicting **No** loan eligibility, with an F1-score of **0.77**, compared to **0.54** for the Yes class. The results show that the model can predict loan eligibility, but further improvement is needed, especially for the **Yes** class.
