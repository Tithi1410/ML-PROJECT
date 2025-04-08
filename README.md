# Crop Recommendation using Machine Learning

## Overview
This project aims to recommend the most suitable **crop to grow** based on various soil and climatic conditions. A **Random Forest Classifier** was trained on a dataset from **Kaggle**, analyzing features like Nitrogen, Phosphorous, Potassium, pH, rainfall, temperature, and humidity to predict the ideal crop out of 22 options.

---

### 🔹 Problem Type: Multi-class Classification  
### 🔹 Algorithm: Random Forest (Supervised ML)  
### 🔹 Web Interface: Streamlit  
### 🔹 Accuracy: 99%  
### 🔹 Dataset Source: Kaggle – Crop Recommendation Dataset

---

## Features

- **Numerical:**
  - `N` – Nitrogen content in soil
  - `P` – Phosphorous content in soil
  - `K` – Potassium content in soil
  - `temperature` – Temperature in °C
  - `humidity` – Relative humidity in %
  - `ph` – pH value of the soil
  - `rainfall` – Rainfall in mm

- **Target:**
  - `label` – Recommended crop (e.g., rice, maize, mango, cotton, etc.)

- **Size:**
  - 2200+ records × 8 columns

---

## Workflow

### 1) Exploratory Data Analysis (EDA)
- Visualized feature distributions using histograms and pair plots.
- Analyzed feature correlations with a heatmap.
- Verified class distribution across crops.

### 2) Data Preprocessing
- Checked for missing/null values (none found).
- Normalized inputs (optional for tree models).
- Split data into **80% training** and **20% testing**.

### 3) Model Training
- Trained using **Random Forest Classifier** with default hyperparameters.
- Saved trained model and column structure using `joblib`.

### 4) Evaluation Metrics
- **Accuracy:** 99%  
- **Confusion Matrix:** High precision and recall across all crop classes  
- **Top Features:**  
  - Nitrogen and rainfall had strong predictive power

---

## Results & Insights

### Model Strengths
- Excellent accuracy and generalization.
- Works well across a wide range of crop classes.
- Feature importance can be extracted easily from the model.

### Practical Use-Cases
- Helps farmers make informed decisions about crop selection.
- Can be used as a module in larger smart agriculture systems.

---

## Web App (Streamlit)

A Streamlit-based web interface allows users to input values for N, P, K, temperature, humidity, pH, and rainfall, and receive a recommended crop in real-time.

```bash
streamlit run app.py
