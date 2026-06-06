from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["baseline", "embedding", "attention", "full"]

MODEL_COLORS = {
    "baseline": "#4C78A8",
    "embedding": "#54A24B",
    "attention": "#F58518",
    "full": "#E45756",
}


def compute_metrics():
    rows = []

    for variant in VARIANTS:
        path = RESULTS_DIR / f"{variant}_test_results.csv"
        df = pd.read_csv(path)

        y_true = df["target"]
        y_pred = df["prediction"]

        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
        r2 = r2_score(y_true, y_pred)

        rows.append({
            "variant": variant,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "pearson": pearson,
            "spearman": spearman,
            "r2": r2,
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    return metrics_df


def plot_training_curves():
    plt.figure(figsize=(8, 5))

    for variant in VARIANTS:
        history = pd.read_csv(RESULTS_DIR / f"{variant}_history.csv")
        plt.plot(
            history["epoch"],
            history["val_loss"],
            label=f"{variant} val",
            color=MODEL_COLORS[variant],
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("Validation loss across model variants")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_loss_curves.png", dpi=300)
    plt.close()


def plot_metric_bar(metrics, metric):
    plt.figure(figsize=(7, 5))

    colors = [MODEL_COLORS[variant] for variant in metrics["variant"]]

    bars = plt.bar(
        metrics["variant"],
        metrics[metric],
        color=colors,
    )

    plt.xlabel("Model variant")
    plt.ylabel(metric.upper())
    plt.title(f"Model comparison by {metric.upper()}")
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"model_{metric}_comparison.png", dpi=300)
    plt.close()


def plot_predicted_vs_true(best_variant):
    df = pd.read_csv(RESULTS_DIR / f"{best_variant}_test_results.csv")

    y_min = min(df["target"].min(), df["prediction"].min())
    y_max = max(df["target"].max(), df["prediction"].max())

    plt.figure(figsize=(6, 6))
    plt.scatter(
        df["target"],
        df["prediction"],
        alpha=0.15,
        color=MODEL_COLORS[best_variant],
    )

    plt.plot(
        [y_min, y_max],
        [y_min, y_max],
        linestyle="--",
        color="black",
        linewidth=1,
        label="Ideal prediction",
    )

    plt.xlabel("True activity")
    plt.ylabel("Predicted activity")
    plt.title(f"Predicted vs True activity: {best_variant}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{best_variant}_predicted_vs_true.png", dpi=300)
    plt.close()


def main():
    metrics = compute_metrics()
    print(metrics)

    plot_training_curves()
    plot_metric_bar(metrics, "mse")
    plot_metric_bar(metrics, "rmse")
    plot_metric_bar(metrics, "mae")
    plot_metric_bar(metrics, "pearson")
    plot_metric_bar(metrics, "spearman")
    plot_metric_bar(metrics, "r2")

    best_variant = metrics.loc[metrics["mse"].idxmin(), "variant"]
    plot_predicted_vs_true(best_variant)

    print(f"Best model variant: {best_variant}")


if __name__ == "__main__":
    main()