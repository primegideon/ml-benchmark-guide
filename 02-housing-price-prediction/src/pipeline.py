"""
Continuous Regression Pipeline (California Housing Price Prediction)
Reference Track ID: DT-2026-NKPSG

End-to-end regression pipeline comparing OLS, Ridge (L2), and Lasso (L1):
1. Loads the California Housing dataset.
2. Imputes missing values in total_bedrooms using median strategy.
3. Engineers interaction ratios (rooms/household, bedrooms/room, population/household).
4. One-hot encodes categorical attribute (ocean_proximity).
5. Splits data into 80/20 train/test sets and standardizes numeric features.
6. Tunes alpha hyperparameters for Ridge and Lasso via 5-fold GridSearchCV.
7. Benchmarks models on RMSE, MAE, and R^2 score.
8. Generates evaluation plots (correlation heatmap, predicted vs actual, residual distribution).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def setup_directories(base_dir: str) -> str:
    """Create output folders if they do not exist."""
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    return assets_dir


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the California housing dataset from CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values and construct interaction ratios:
    - rooms_per_household = total_rooms / households
    - bedrooms_per_room = total_bedrooms / total_rooms
    - population_per_household = population / households
    """
    data = df.copy()

    # Median imputation for total_bedrooms
    imputer = SimpleImputer(strategy="median")
    data["total_bedrooms"] = imputer.fit_transform(data[["total_bedrooms"]])

    # Interaction features
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]

    return data


def generate_eda(df: pd.DataFrame, assets_dir: str):
    """Generate and save feature correlation heatmap."""
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("California Housing Correlation Matrix (with Interaction Features)", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    corr_path = os.path.join(assets_dir, "correlation_matrix.png")
    plt.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to: {corr_path}")


def preprocess_data(df: pd.DataFrame):
    """
    One-hot encode ocean_proximity, split into 80/20 train/test sets,
    and scale continuous predictor variables with StandardScaler.
    """
    target_col = "median_house_value"
    cat_col = "ocean_proximity"

    X_raw = df.drop(columns=[target_col])
    y = df[target_col].values

    # One-hot encode categorical column
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    cat_encoded = ohe.fit_transform(X_raw[[cat_col]])
    cat_names = [f"ocean_{c}" for c in ohe.get_feature_names_out([cat_col])]

    num_cols = [c for c in X_raw.columns if c != cat_col]
    X_num = X_raw[num_cols].values

    X_combined = np.hstack([X_num, cat_encoded])
    feature_names = num_cols + cat_names

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.20, random_state=42
    )

    # Standardize all features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def train_and_benchmark(X_train, X_test, y_train, y_test):
    """
    Fit Ordinary Least Squares (OLS) baseline, then perform 5-fold GridSearchCV
    for Ridge and Lasso across alpha values.
    """
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]

    # 1. OLS
    ols = LinearRegression()
    ols.fit(X_train, y_train)

    # 2. Ridge (L2) with GridSearchCV
    ridge_grid = GridSearchCV(
        Ridge(),
        param_grid={"alpha": alphas},
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    ridge_grid.fit(X_train, y_train)
    best_ridge = ridge_grid.best_estimator_

    # 3. Lasso (L1) with GridSearchCV
    lasso_grid = GridSearchCV(
        Lasso(max_iter=5000, random_state=42),
        param_grid={"alpha": alphas},
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    lasso_grid.fit(X_train, y_train)
    best_lasso = lasso_grid.best_estimator_

    models = {
        "OLS Linear Regression": ols,
        f"Ridge Regression (alpha={ridge_grid.best_params_['alpha']})": best_ridge,
        f"Lasso Regression (alpha={lasso_grid.best_params_['alpha']})": best_lasso,
    }

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "model_obj": model,
            "y_pred": y_pred,
        }

    return results, best_ridge, best_lasso


def plot_predicted_vs_actual(results, y_test, assets_dir: str):
    """Scatter plot of predicted vs actual house prices with ideal 1:1 reference line."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (name, metrics) in zip(axes, results.items()):
        y_pred = metrics["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.15, s=15, color="#1f77b4", edgecolor="none")

        # Identity reference line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.8, label="Ideal (y=x)")

        ax.set_title(f"{name}\nR2 = {metrics['R2']:.4f}", fontsize=11, weight="bold")
        ax.set_xlabel("Actual Median House Value ($)", fontsize=10)
        ax.set_ylabel("Predicted Value ($)", fontsize=10)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(assets_dir, "predicted_vs_actual.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved predicted vs actual plot to: {plot_path}")


def plot_residual_distributions(results, y_test, assets_dir: str):
    """Plot residual error distributions (y - y_pred) across models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, metrics) in zip(axes, results.items()):
        residuals = y_test - metrics["y_pred"]
        sns.histplot(residuals, kde=True, ax=ax, color="#2b5c8f", bins=50)

        mean_res = np.mean(residuals)
        std_res = np.std(residuals)

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
        ax.set_title(f"Residuals: {name}\nMean={mean_res:,.0f}, Std={std_res:,.0f}", fontsize=11, weight="bold")
        ax.set_xlabel("Residual Error (Actual - Predicted, $)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(loc="upper right")

    plt.tight_layout()
    res_path = os.path.join(assets_dir, "residual_distributions.png")
    plt.savefig(res_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residual distribution plot to: {res_path}")


def plot_coefficient_shrinkage(feature_names, ols, best_ridge, best_lasso, assets_dir: str):
    """Compare learned coefficients across OLS, Ridge, and Lasso to show L1/L2 shrinkage."""
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "OLS": ols.coef_,
        "Ridge": best_ridge.coef_,
        "Lasso": best_lasso.coef_,
    }).sort_values(by="OLS", key=abs, ascending=False)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(feature_names))
    width = 0.25

    plt.bar(x - width, coef_df["OLS"], width=width, label="OLS (Unregularized)", color="#4c72b0")
    plt.bar(x, coef_df["Ridge"], width=width, label="Ridge (L2 Shrinkage)", color="#55a868")
    plt.bar(x + width, coef_df["Lasso"], width=width, label="Lasso (L1 Sparsity)", color="#c44e52")

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xticks(x, coef_df["Feature"], rotation=45, ha="right", fontsize=9)
    plt.ylabel("Standardized Coefficient Value", fontsize=10)
    plt.title("Model Coefficient Comparison: OLS vs Ridge vs Lasso", fontsize=12, weight="bold")
    plt.legend()
    plt.tight_layout()

    coef_path = os.path.join(assets_dir, "coefficient_comparison.png")
    plt.savefig(coef_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved coefficient comparison to: {coef_path}")


def run_pipeline():
    task_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(task_dir, "data", "housing.csv")
    assets_dir = setup_directories(task_dir)

    print("=== Task 2: California Housing Price Regression Pipeline ===")
    raw_df = load_dataset(data_path)
    df = engineer_features(raw_df)
    generate_eda(df, assets_dir)

    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    results, best_ridge, best_lasso = train_and_benchmark(X_train, X_test, y_train, y_test)

    ols_model = list(results.values())[0]["model_obj"]
    plot_predicted_vs_actual(results, y_test, assets_dir)
    plot_residual_distributions(results, y_test, assets_dir)
    plot_coefficient_shrinkage(feature_names, ols_model, best_ridge, best_lasso, assets_dir)

    print("\n=== Model Benchmark Comparison ===")
    summary_data = []
    for name, metrics in results.items():
        summary_data.append({
            "Model": name,
            "RMSE ($)": f"{metrics['RMSE']:,.2f}",
            "MAE ($)": f"{metrics['MAE']:,.2f}",
            "R2 Score": f"{metrics['R2']:.4f}",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print(f"\nExecution complete. Output figures saved in: {assets_dir}")


if __name__ == "__main__":
    run_pipeline()

