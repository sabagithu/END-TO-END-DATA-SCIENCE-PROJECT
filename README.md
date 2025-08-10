# END-TO-END-DATA-SCIENCE-PROJECT

- **Company Name**: CodTech IT Solutions  
- **Intern Name**: *AYESHA SABA*  
- **Intern ID**: CT4MWP218  
- **Domain**: Data Science  
- **Duration**:  16 Weeks  
- **Mentor**: Neela Sanrosh

# ❤️ Heart Disease Prediction – End-to-End Data Science Project

## 📖 Project Overview
This project is part of my internship deliverables at **CodTech IT Solutions**, focused on building an end-to-end machine learning pipeline to predict the likelihood of heart disease based on patient health metrics.

It covers the full data science workflow: data collection, preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and deployment. The final model is served via a Flask API, allowing real-time predictions through HTTP requests.

---

## 🎯 Objectives
- Build an accurate heart disease prediction model.
- Deploy the model as a REST API for real-time usage.
- Demonstrate a complete data science workflow from raw data to production-ready application.

---

## 🛠 Tools & Technologies Used
- **Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Deployment**: Flask, Joblib, Ngrok  
- **Environment**: Jupyter Notebook / Conda (`heart_env`)

---

## 📂 Dataset
- **Source**: `heart.csv`  
- **Features**: Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, slope, number of major vessels, thalassemia  
- **Target**: Presence or absence of heart disease (binary classification)

---

## 📋 Project Workflow

### 🔹 Data Import & Inspection
- Loaded dataset into a Pandas DataFrame  
- Checked for missing values, data types, and basic statistics

### 🔹 Exploratory Data Analysis (EDA)
- Visualized distributions, correlations, and outliers  
- Used heatmaps, histograms, and boxplots for insights

### 🔹 Data Preprocessing & Feature Engineering
- Handled missing values  
- Encoded categorical variables  
- Standardized numerical features using `StandardScaler`

### 🔹 Model Building
- Tested multiple models: Logistic Regression, Random Forest, Gradient Boosting  
- Selected **Random Forest Classifier** for final deployment

### 🔹 Model Evaluation
- Evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC  
- Tuned hyperparameters for improved performance

### 🔹 Deployment
- Built a Flask API with endpoints:
  - `/` → Health check
  - `/predict` → Accepts POST requests with JSON input and returns prediction  
- Used **Ngrok** for temporary public URL hosting

---

## 📊 Final Results

- **Model Used**: Random Forest Classifier  
- **✅ Model Accuracy**: `0.8197`  

### 🧾 Confusion Matrix
[[24  8] 
 [ 3 26]]
- **Class 0**: 24 correctly predicted, 8 misclassified as Class 1  
- **Class 1**: 26 correctly predicted, 3 misclassified as Class 0  

### 📋 Classification Report

| Metric        | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|---------------|---------|---------|-----------|---------------|
| Precision     | 0.89    | 0.76    | 0.83      | 0.83          |
| Recall        | 0.75    | 0.90    | 0.82      | 0.82          |
| F1-Score      | 0.81    | 0.83    | 0.82      | 0.82          |
| Support       | 32      | 29      | 61        | 61            |

### 🔍 Insights
- The model performs slightly better at identifying Class 1 (higher recall).
- Precision is higher for Class 0, indicating fewer false positives.
- Overall, the model maintains a strong balance between precision and recall, with an F1-score above 0.80 for both classes.

---
## 📁 Repository Contents

- **`Task 3 end to end project.ipynb`**  
  Jupyter Notebook containing the full pipeline: data loading, preprocessing, model training, evaluation, and deployment setup.

- **`heart.csv`**  
  Dataset used for training and evaluation.

- **`random_forest_model.pkl`**  
  Serialized trained Random Forest model.

- **`scaler.pkl`**  
  Serialized `StandardScaler` object used for feature scaling.

- **`training_columns.pkl`**  
  List of feature columns used during training to ensure consistent input during inference.

- **`README.md`**  
  Project documentation and overview.

---

## 🚀 Potential Applications

- **Early diagnosis support for healthcare professionals**  
  Assists doctors in identifying potential heart disease cases based on patient data.

- **Risk assessment tools for preventive cardiology**  
  Helps evaluate individual risk levels to guide lifestyle changes and early interventions.

- **Integration into wearable health monitoring systems**  
  Can be embedded into smart devices to provide continuous health insights.

- **Real-time prediction services in clinical software**  
  Enables instant decision support within electronic health record (EHR) platforms.

---


## 🧪 Environment Setup

This project was developed using a Conda environment for reproducibility.

### 🔧 Environment Details
- **Environment Manager**: Conda  
- **Environment Name**: `heart_env`  
- **Python Version**: 3.x

### 📦 Installation

To recreate the environment:

```bash
conda create --name heart_env python=3.x
conda activate heart_env
pip install -r requirements.txt
