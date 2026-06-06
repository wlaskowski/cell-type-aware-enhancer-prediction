from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = "embedding"


def main():
    pred_df = pd.read_csv(RESULTS_DIR / f"{BEST_MODEL}_test_results.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    df = pred_df.copy()
    df["cell_type"] = test_df["cell_type"].values

    rows = []

    for cell_type, group in df.groupby("cell_type"):
        y_true = group["target"]
        y_pred = group["prediction"]

        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
        r2 = r2_score(y_true, y_pred)

        rows.append({
            "model": BEST_MODEL,
            "cell_type": cell_type,
            "n": len(group),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "pearson": pearson,
            "spearman": spearman,
            "r2": r2,
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "cell_type_metrics.csv", index=False)
    print(metrics)

    for cell_type, group in df.groupby("cell_type"):
        plt.figure(figsize=(6, 6))

        plt.scatter(
            group["target"],
            group["prediction"],
            alpha=0.3
        )


        plt.xlabel("True activity")
        plt.ylabel("Predicted activity")
        plt.title(f"Predicted vs True: {cell_type}")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"{cell_type}_predicted_vs_true.png",
            dpi=300
        )
        plt.close()


if __name__ == "__main__":
    main()