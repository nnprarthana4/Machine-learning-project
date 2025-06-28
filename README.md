# 🌾 Rainfall Prediction for Enhancing Crop Yield

## 📌 Project Overview

Agriculture is the backbone of India's economy, heavily dependent on rainfall patterns. Due to climate change, rainfall has become increasingly erratic, threatening crop productivity and food security. This project aims to use **machine learning** to predict rainfall and classify crop yield as **High** or **Low** based on agro-climatic factors.

## 🎯 Objectives

- Predict rainfall and its impact on crop yield.
- Classify yield into "High" or "Low" using ML techniques.
- Help farmers plan irrigation and crop schedules effectively.
- Compare ML models to find the most accurate classifier.
- Provide visual and interpretable outputs to guide decision-making.

---

## 📁 Dataset Description

- **Source**: [Kaggle](https://www.kaggle.com/)
- **File**: `crop_yield.csv`
- **Rows**: *[Specify after loading]*
- **Columns**: *[Specify after encoding]*

### 📊 Features

- `Crop_Year`
- `State`
- `Season`
- `Crop`
- `Area` (hectares)
- `Production` (tonnes)
- `Annual_Rainfall` (mm)
- `Pesticide` (kg/hectare)
- `Fertilizer` (kg/hectare)
- `Yield` (Production / Area)
- `Yield_Class` (Target: "High" or "Low")

---

## 🧪 Preprocessing Steps

- Removed rows with missing values.
- Applied One-Hot Encoding to categorical columns.
- Created binary target variable `Yield_Class` based on median yield.
- Dropped original `Yield` column after classification.
- Used 80:20 train-test split.
- Applied 10-fold cross-validation.

---

## 🤖 Machine Learning Models

Three models were evaluated:

| Model              | Test Accuracy | F1 Score | Precision | Recall |
|-------------------|---------------|----------|-----------|--------|
| **Random Forest**  | **94.87%**    | **94.81%** | **95.64%** | **93.99%** |
| Gradient Boosting | 92.13%        | 91.97%   | 92.21%    | 91.74% |
| Logistic Regression | 81.64%       | 78.22%   | 95.65%    | 66.16% |

> ✅ **Random Forest** outperformed other models and was selected for final deployment.

---

## 🛠️ Tools & Libraries Used

| Tool/Library       | Purpose                            |
|--------------------|------------------------------------|
| Python             | Programming language               |
| pandas, numpy      | Data loading & manipulation        |
| matplotlib, seaborn| Data visualization                 |
| scikit-learn       | ML model training & evaluation     |
| warnings           | Suppress unnecessary warnings      |

---

## 📈 Model Implementation Steps

1. Load and inspect data using `pandas`.
2. Clean data and handle missing values.
3. Encode categorical features using `OneHotEncoder`.
4. Transform target variable into binary class.
5. Train Random Forest Classifier with `n_estimators=100`.
6. Evaluate with 10-fold cross-validation.
7. Visualize results using confusion matrix and feature importance plots.
8. Deploy prediction functions for single and batch inputs.

---

## 🔍 Evaluation Metrics

- **Accuracy**: Overall correctness of model
- **Precision**: Correct "High Yield" predictions out of all predicted "High"
- **Recall**: Correct "High Yield" predictions out of all actual "High"
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed class-level error inspection

---

## 📌 Results

- Final Model: **Random Forest Classifier**
- Test Accuracy: **94.87%**
- F1 Score: **94.81%**
- Cross-validation Accuracy: **95.20%**
- Most Important Features: `Area`, `Crop`, `Season`, `State`, `Annual_Rainfall`

---

## 🧠 Functions

- `prediction(input_data)` – Predicts class for single input.
- `make_batch_predictions(input_dataframe)` – Predicts for multiple samples.

---

## 📊 Visual Outputs

- 📌 Confusion Matrix Heatmaps
- 📈 Accuracy Line Graphs (per fold)
- 🌾 Feature Importance Bar Charts

---

## 🧾 How to Run

Follow the steps below to set up and run the project:

```bash
# Step 1: Clone the repository
git clone https://github.com/nnprarthana4/Machine-learning-project.git

# Step 2: Navigate to the project directory
cd Machine-learning-project

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the main script
python ML.py
