# Task 1: Multi-Class Benchmark Classification Pipeline (Iris Dataset)

**Reference Track ID:** DT-2026-NKPSG  
**Focus:** Exploratory Data Analysis, Feature Scaling, Multi-Class Model Benchmarking  

---

## 1. Objective

The goal of this task is to build and evaluate an end-to-end multi-class classification pipeline on the benchmark Iris dataset.

Key steps covered:
1. Exploratory Data Analysis (EDA) using Seaborn pairplots and correlation matrices.
2. Feature scaling with `StandardScaler` and stratified 80/20 train/test split.
3. Training and comparing two distinct classifiers:
   - **Logistic Regression** (linear decision boundaries)
   - **Decision Tree Classifier** (non-linear orthogonal splits)
4. Model evaluation using precision, recall, F1-score, confusion matrix heatmaps, and 2D decision boundary visualization.

---

## 2. Directory Structure

```text
01-iris-classification/
├── data/
│   └── iris.csv                 # Raw dataset
├── notebooks/
│   └── iris_pipeline.ipynb      # Step-by-step Jupyter Notebook
├── src/
│   └── pipeline.py              # Modular Python pipeline script
├── assets/                      # Generated figures and evaluation plots
└── README.md                    # Task documentation
```

---

## 3. How to Run

From the repository root:

```bash
python 01-iris-classification/src/pipeline.py
```
