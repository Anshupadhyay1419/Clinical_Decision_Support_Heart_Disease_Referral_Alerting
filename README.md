# Heart Disease Clinical Decision Support System

A machine learning-based Clinical Decision Support System for predicting the risk of heart disease using the UCI Cleveland Heart Disease dataset.

## Features

- Data preprocessing and missing-value handling
- Exploratory Data Analysis (EDA)
- One-hot encoding and feature scaling
- Logistic Regression model
- Model evaluation using Accuracy, Precision, Recall, F1-score and ROC-AUC
- Decision threshold analysis
- Sample patient risk prediction

## Project Structure

```text
Healthcare/
├── data/
├── outputs/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── threshold_analysis.py
│   └── patient_prediction.py
├── main.py
├── requirements.txt
└── README.md