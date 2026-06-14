"""
Analyze the performance of the best model separately for each cell type
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
PROCESSED_DIR = Path("data/processed")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# manualy selecting the best performing model - could be improved and automated :) 
BEST_MODEL = "embedding"

CELL_COLORS = {
    "HepG2": "#4C78A8",
    "K562": "#54A24B",
    "WTC11": "#E45756",
}


def compute_cell_type_metrics(df):
    """
    Calculate evaluation metrics for each cell type
    """
    rows = []

    # dividing test results into separate cell gropus
    for cell_type, group in df.groupby("cell_type"):
        
        y_true = group["target"]
        y_pred = group["prediction"]

        mse = mean_squared_error(y_true, y_pred)

        rows.append({
            "model": BEST_MODEL,
            "cell_type": cell_type,
            "n": len(group),
            "mse": mse,
            "rmse": mse ** 0.5,
            "mae": mean_absolute_error(y_true, y_pred),
            "pearson": pearsonr(y_true, y_pred)[0],
            "spearman": spearmanr(y_true, y_pred)[0],
            "r2": r2_score(y_true, y_pred),
        })

    return pd.DataFrame(rows)


def plot_predicted_vs_true_by_cell_type(df):
    """
    Predicted vs true scatter plots for separate cell types
    """
    for cell_type, group in df.groupby("cell_type"):
        plt.figure(figsize=(6, 6))

        plt.scatter(
            group["target"],
            group["prediction"],
            alpha=0.15,
            s=10,
            color=CELL_COLORS.get(cell_type, "#4C78A8"),
        )

        plt.xlabel("True activity")
        plt.ylabel("Predicted activity")
        plt.title(f"Predicted vs True: {cell_type}")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(FIGURES_DIR / f"{cell_type}_predicted_vs_true.png", dpi=300)
        plt.close()


def plot_cell_type_performance(metrics_df, metric):
    """
    Bar plot comparing one metric across cell types
    """
    plt.figure(figsize=(6, 4))

    # one plot for each cell type
    bars = plt.bar(
        metrics_df["cell_type"],
        metrics_df[metric],
        color=[CELL_COLORS.get(cell_type, "#4C78A8") for cell_type in metrics_df["cell_type"]],
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Cell type")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} by cell type")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / f"cell_type_{metric}_comparison.png", dpi=300)
    plt.close()


def main():
    pred_df = pd.read_csv(RESULTS_DIR / f"{BEST_MODEL}_test_results.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    df = pred_df.copy()
    df["cell_type"] = test_df["cell_type"].values

    metrics_df = compute_cell_type_metrics(df)
    metrics_df.to_csv(RESULTS_DIR / "cell_type_metrics.csv", index=False)

    print(metrics_df)

    plot_predicted_vs_true_by_cell_type(df)
    plot_cell_type_performance(metrics_df, "pearson")
    plot_cell_type_performance(metrics_df, "r2")


if __name__ == "__main__":
    main()