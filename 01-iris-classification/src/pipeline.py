"""
Multi-Class Benchmark Classification Pipeline (Iris Flower Dataset)
Reference Track ID: DT-2026-NKPSG

This script implements an end-to-end classification pipeline:
1. Loads the Iris dataset.
2. Generates exploratory visualizations (pairplots and correlation matrix).
3. Prepares data with StandardScaler and splits into train/test sets (80/20).
4. Trains Logistic Regression and Decision Tree models.
5. Evaluates model performance using classification metrics, confusion matrices,
   and 2D decision boundary plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def setup_directories(base_dir: str) -> str:
    """Create output folders if they do not exist."""
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    return assets_dir


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the Iris dataset from a CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def generate_eda(df: pd.DataFrame, assets_dir: str):
    """Create and save exploratory data analysis charts."""
    sns.set_theme(style="whitegrid")

    # Pairplot to inspect feature separation across species
    pairplot = sns.pairplot(df, hue="species", palette="viridis", diag_kind="kde", height=2.2)
    pairplot.fig.subplots_adjust(top=0.93)
    pairplot.fig.suptitle("Iris Feature Distributions and Pairwise Relationships", fontsize=13, weight="bold")
    pairplot_path = os.path.join(assets_dir, "iris_pairplot.png")
    pairplot.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pairplot to: {pairplot_path}")

    # Feature correlation heatmap
    plt.figure(figsize=(7, 5.5))
    numeric_df = df.drop(columns=["species"])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1, square=True)
    plt.title("Iris Feature Correlation Matrix", fontsize=13, weight="bold", pad=12)
    heatmap_path = os.path.join(assets_dir, "feature_correlation.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_path}")


def preprocess_data(df: pd.DataFrame):
    """
    Encode labels, perform stratified 80/20 train/test split,
    and normalize feature columns using StandardScaler.
    """
    X = df.drop(columns=["species"]).values
    y_raw = df["species"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_

    # Stratified 80/20 split so class balance is preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, class_names, scaler


def train_and_benchmark(X_train, X_test, y_train, y_test, class_names):
    """Train Logistic Regression and Decision Tree models and calculate metrics."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42),
    }

    results = {}
    reports = {}
    matrices = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            "model_obj": clf,
            "y_pred": y_pred,
        }
        reports[name] = classification_report(y_test, y_pred, target_names=class_names)
        matrices[name] = confusion_matrix(y_test, y_pred)

    return results, reports, matrices


def plot_confusion_matrices(matrices, class_names, assets_dir: str):
    """Plot side-by-side confusion matrix heatmaps for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, (name, cm) in zip(axes, matrices.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=ax,
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, weight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    cm_path = os.path.join(assets_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to: {cm_path}")


def plot_decision_boundaries(df: pd.DataFrame, assets_dir: str):
    """Plot 2D decision boundaries using petal length and petal width."""
    X_2d = df[["petal_length", "petal_width"]].values
    le = LabelEncoder()
    y = le.fit_transform(df["species"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    lr_2d = LogisticRegression(max_iter=200, random_state=42)
    dt_2d = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)

    lr_2d.fit(X_train_s, y_train)
    dt_2d.fit(X_train_s, y_train)

    # Create meshgrid for boundary contour
    x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
    y_min, y_max = X_train_s[:, 1].min() - 1, X_train_s[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    model_pairs = [
        ("Logistic Regression (Linear Boundaries)", lr_2d),
        ("Decision Tree (Non-Linear Splits)", dt_2d),
    ]

    for ax, (title, clf) in zip(axes, model_pairs):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

        ax.scatter(
            X_train_s[:, 0],
            X_train_s[:, 1],
            c=y_train,
            cmap="viridis",
            edgecolor="black",
            s=40,
            label="Training Samples",
        )
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Petal Length (Standardized)", fontsize=10)
        ax.set_ylabel("Petal Width (Standardized)", fontsize=10)

    plt.tight_layout()
    db_path = os.path.join(assets_dir, "decision_boundaries.png")
    plt.savefig(db_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved decision boundaries to: {db_path}")


def run_pipeline():
    task_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(task_dir, "data", "iris.csv")
    assets_dir = setup_directories(task_dir)

    print("=== Task 1: Iris Classification Pipeline ===")
    df = load_dataset(data_path)
    generate_eda(df, assets_dir)

    X_train, X_test, y_train, y_test, class_names, scaler = preprocess_data(df)
    results, reports, matrices = train_and_benchmark(X_train, X_test, y_train, y_test, class_names)

    plot_confusion_matrices(matrices, class_names, assets_dir)
    plot_decision_boundaries(df, assets_dir)

    print("\n--- Performance Metrics Summary ---")
    summary_data = []
    for name, metrics in results.items():
        summary_data.append({
            "Model": name,
            "Accuracy": f"{metrics['Accuracy'] * 100:.2f}%",
            "Precision (Weighted)": f"{metrics['Precision'] * 100:.2f}%",
            "Recall (Weighted)": f"{metrics['Recall'] * 100:.2f}%",
            "F1-Score (Weighted)": f"{metrics['F1-Score'] * 100:.2f}%",
        })
        print(f"\nClassification Report for {name}:\n")
        print(reports[name])

    summary_df = pd.DataFrame(summary_data)
    print("\nComparison Table:")
    print(summary_df.to_string(index=False))
    print(f"\nCompleted. Generated assets are stored in: {assets_dir}")


if __name__ == "__main__":
    run_pipeline()
