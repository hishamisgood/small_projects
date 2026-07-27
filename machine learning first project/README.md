# 🌾 Predictive Modeling for Agriculture: Crop Classification

A machine learning project focused on evaluating soil metrics to identify the single best feature for predicting crop types using **Python**, **Pandas**, and **Scikit-Learn**.

---

## 📌 Project Overview
Soil composition is fundamental to crop success. This project analyzes soil measurement data to determine which specific soil property provides the strongest predictive power for classifying crop types. By evaluating individual features independently using logistic regression models, the pipeline programmatically identifies the optimal soil metric for agricultural planning.

---

## 🎯 Key Objectives
* **Feature Evaluation:** Train and test independent classification models on individual soil metrics.
* **Automated Scoring:** Programmatically map and compare `accuracy_score` metrics across all features.
* **Feature Selection:** Determine the single best-performing soil variable for predicting crop type.

---

## 📊 Dataset Overview
The project uses `soil_measures.csv`, which includes key chemical and physical soil measures:
* **N:** Nitrogen ratio in soil
* **P:** Phosphorus ratio in soil
* **K:** Potassium ratio in soil
* **ph:** Soil pH level
* **crop:** Target variable (crop label)

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn (`LogisticRegression`, `train_test_split`, `metrics.accuracy_score`)
* **Environment:** Jupyter Notebook (`notebook.ipynb`)

---

## 📁 Project Structure
```text
soil-measures-classification/
├── README.md
├── notebook.ipynb
└── soil_measures.csv
