"""
Predictive Customer Churn Classification Pipeline (Random Forest & XGBoost)
Reference Track ID: DT-2026-NKPSG

End-to-end churn prediction pipeline addressing class imbalance:
1. Loads the Telco Customer Churn dataset.
2. Converts TotalCharges to numeric and imputes missing values.
3. Encodes categorical variables via OneHotEncoder and extracts binary target.
4. Stratifies an 80/20 train/test split.
5. Balances training class distributions using SMOTE oversampling.
6. Trains and evaluates RandomForestClassifier and XGBClassifier.
7. Evaluates ROC-AUC, classification metrics, and confusion matrices.
8. Extracts and plots top 10 feature importances and ROC curves.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE


def setup_directories(base_dir: str) -> str:
    """Create output folders if they do not exist."""
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    return assets_dir


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_and_preprocess(df: pd.DataFrame):
    """
    Handle blank entries in TotalCharges, drop identifier columns,
    and encode categorical variables with OneHotEncoder.
    """
    data = df.copy()

    # Convert TotalCharges to numeric (blank strings become NaN)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    # Target variable
    y = (data["Churn"] == "Yes").astype(int).values

    # Drop ID and target
    features_df = data.drop(columns=["customerID", "Churn"])

    cat_cols = features_df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = features_df.select_dtypes(exclude=["object"]).columns.tolist()

    # One-hot encoding
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    cat_encoded = ohe.fit_transform(features_df[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

    X = np.hstack([features_df[num_cols].values, cat_encoded])
    feature_names = num_cols + cat_feature_names

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names


def balance_and_train(X_train, X_test, y_train, y_test):
    """
    Apply SMOTE to balance the training set, then train Random Forest
    and XGBoost classifiers.
    """
    print(f"Original training distribution: Class 0 = {np.sum(y_train == 0)}, Class 1 = {np.sum(y_train == 1)}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled training distribution: Class 0 = {np.sum(y_train_res == 0)}, Class 1 = {np.sum(y_train_res == 1)}")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss"),
    }

    results = {}
    reports = {}
    matrices = {}

    for name, clf in models.items():
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test)
        y_probs = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "model_obj": clf,
            "y_pred": y_pred,
            "y_probs": y_probs,
        }
        reports[name] = classification_report(y_test, y_pred, target_names=["Retained (0)", "Churned (1)"])
        matrices[name] = confusion_matrix(y_test, y_pred)

    return results, reports, matrices


def plot_roc_curves(results, y_test, assets_dir: str):
    """Plot ROC-AUC curves comparing Random Forest and XGBoost against diagonal baseline."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6.5))

    colors = {"Random Forest": "#2b5c8f", "XGBoost": "#d95f02"}

    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics["y_probs"])
        auc = metrics["ROC-AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors[name], linewidth=2.2)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Random Guess (AUC = 0.5000)")
    plt.axhline(0.80, linestyle=":", color="green", alpha=0.7, label="Target Threshold (0.80)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("ROC-AUC Comparison: Random Forest vs. XGBoost", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    roc_path = os.path.join(assets_dir, "roc_auc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to: {roc_path}")


def plot_confusion_matrices(matrices, assets_dir: str):
    """Plot side-by-side confusion matrix heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = ["Retained", "Churned"]

    for ax, (name, cm) in zip(axes, matrices.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, weight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    cm_path = os.path.join(assets_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to: {cm_path}")


def plot_feature_importances(results, feature_names, assets_dir: str):
    """Plot top 10 most influential feature importances for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (name, metrics) in zip(axes, results.items()):
        model = metrics["model_obj"]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]

        top_names = [feature_names[i] for i in indices]
        top_scores = importances[indices]

        sns.barplot(x=top_scores, y=top_names, ax=ax, hue=top_names, palette="mako", legend=False)
        ax.set_title(f"Top 10 Feature Importances: {name}", fontsize=12, weight="bold")
        ax.set_xlabel("Normalized Importance Weight", fontsize=10)
        ax.set_ylabel("Feature Name", fontsize=10)

    plt.tight_layout()
    fi_path = os.path.join(assets_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importances to: {fi_path}")


def run_pipeline():
    task_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(task_dir, "data", "Telco-Customer-Churn.csv")
    assets_dir = setup_directories(task_dir)

    print("=== Task 3: Customer Churn Classification Pipeline ===")
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test, feature_names = clean_and_preprocess(df)
    results, reports, matrices = balance_and_train(X_train, X_test, y_train, y_test)

    plot_roc_curves(results, y_test, assets_dir)
    plot_confusion_matrices(matrices, assets_dir)
    plot_feature_importances(results, feature_names, assets_dir)

    print("\n=== Model Benchmark Comparison ===")
    summary_data = []
    for name, metrics in results.items():
        summary_data.append({
            "Model": name,
            "Accuracy": f"{metrics['Accuracy'] * 100:.2f}%",
            "Precision": f"{metrics['Precision'] * 100:.2f}%",
            "Recall": f"{metrics['Recall'] * 100:.2f}%",
            "F1-Score": f"{metrics['F1-Score'] * 100:.2f}%",
            "ROC-AUC": f"{metrics['ROC-AUC']:.4f}",
        })
        print(f"\nClassification Report for {name}:\n")
        print(reports[name])

    summary_df = pd.DataFrame(summary_data)
    print("\nBenchmark Summary Table:")
    print(summary_df.to_string(index=False))
    print(f"\nExecution complete. Figures saved in: {assets_dir}")


if __name__ == "__main__":
    run_pipeline()

