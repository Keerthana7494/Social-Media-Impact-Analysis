import matplotlib

# Use a non-GUI backend for saving charts
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import FIGURES_DIR, TARGET


# Set visualization style
sns.set_theme(style="whitegrid")


# ============================================================
# 1. SAVE DATA ANALYSIS VISUALIZATIONS
# ============================================================

def save_data_visuals(df):

    """Save feature distributions, target balance and correlation."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    numeric = df.select_dtypes(include="number")

    # Feature distributions
    if not numeric.empty:

        columns = list(numeric.columns[:12])

        rows = (len(columns) + 2) // 3

        fig, axes = plt.subplots(
            rows,
            3,
            figsize=(15, 4 * rows)
        )

        axes = axes.flatten()

        for axis, column in zip(axes, columns):

            sns.histplot(
                df[column].dropna(),
                kde=True,
                ax=axis
            )

            axis.set_title(f"Distribution: {column}")

        for axis in axes[len(columns):]:
            axis.remove()

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR / "numeric_feature_distributions.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        # Correlation heatmap
        if len(numeric.columns) >= 2:

            fig, ax = plt.subplots(figsize=(12, 8))

            sns.heatmap(
                numeric.corr(),
                cmap="coolwarm",
                center=0,
                ax=ax
            )

            ax.set_title("Numerical Feature Correlation")

            fig.tight_layout()

            fig.savefig(
                FIGURES_DIR / "correlation_heatmap.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

    # Target distribution
    if TARGET in df.columns:

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.countplot(
            data=df,
            x=TARGET,
            order=df[TARGET].value_counts().index,
            ax=ax
        )

        ax.set_title("Target Class Distribution")

        ax.tick_params(axis="x", rotation=15)

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR / "target_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


# ============================================================
# 2. SAVE MODEL COMPARISON VISUALIZATION
# ============================================================

def save_model_comparison(results):

    """
    Compare all trained models using validation Macro F1.

    Models are displayed from highest to lowest Macro F1.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    ordered = results_df.sort_values(
        "macro_f1",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=ordered,
        x="model",
        y="macro_f1",
        hue="model",
        legend=False,
        ax=ax
    )

    ax.set_ylim(0, 1.10)

    ax.set_title("Model Comparison - Validation Macro F1")

    ax.set_xlabel("Machine Learning Models")

    ax.set_ylabel("Macro F1 Score")

    ax.tick_params(
        axis="x",
        rotation=20
    )

    # Display scores above bars
    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.3f",
            padding=3
        )

    fig.tight_layout()

    chart_path = FIGURES_DIR / "model_comparison.png"

    fig.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Model comparison chart saved:", chart_path)

    return chart_path


# ============================================================
# 3. SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(matrix, labels):

    """Save confusion matrix of the selected final model."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    ax.set_title("Best Model - Confusion Matrix")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Confusion matrix saved.")


# ============================================================
# 4. HYPERPARAMETER TUNING VISUALIZATIONS
# ============================================================

def save_hyperparameter_search_chart(cv_results):
    """Save a chart of the best GridSearchCV candidates."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = (
        cv_results.copy()
        .sort_values("rank_test_score")
        .head(15)
    )

    results = results.sort_values(
        "mean_test_score",
        ascending=True
    )

    labels = [
        f"Candidate {i}"
        for i in range(1, len(results) + 1)
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(
        labels,
        results["mean_test_score"]
    )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Cross-validation Macro F1")
    ax.set_ylabel("Top parameter combinations")
    ax.set_title("Hyperparameter Tuning - Top CV Scores")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, score in zip(bars, results["mean_test_score"]):

        ax.text(
            score + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center"
        )

    fig.tight_layout()

    path = FIGURES_DIR / "hyperparameter_tuning_scores.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Hyperparameter tuning chart saved:", path)

    return path


def save_validation_before_after_chart(metrics_df):
    """Compare original and tuned validation metrics."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1"
    ]

    plot_df = metrics_df.set_index("model_version")[metric_columns]

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_df.T.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylim(0, 1.05)
    ax.set_title("Validation Performance: Before vs After Tuning")
    ax.set_xlabel("Evaluation Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()

    path = FIGURES_DIR / "validation_before_after_tuning.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Validation before/after chart saved:", path)

    return path


# ============================================================
# 5. FINAL TEST METRICS VISUALIZATION
# ============================================================

def save_final_test_metrics_chart(metrics):
    """Save the final held-out test metrics as a bar chart."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    names = list(metrics.keys())
    values = [float(metrics[name]) for name in names]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        names,
        values
    )

    ax.set_ylim(0, 1.05)
    ax.set_title("Final Model - Test Metrics")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center"
        )

    fig.tight_layout()

    path = FIGURES_DIR / "final_test_metrics.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Final test metrics chart saved:", path)

    return path


# ============================================================
# 6. TRAINING VS TESTING PERFORMANCE VISUALIZATION
# ============================================================

def save_train_test_comparison_chart(train_metrics, test_metrics):
    """
    Compare final-model scores on the data used for fitting
    (training + validation) and on the unseen test dataset.

    A large positive difference can indicate possible overfitting.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_names = [
        "Accuracy",
        "Macro F1"
    ]

    train_scores = [
        float(train_metrics["accuracy"]),
        float(train_metrics["macro_f1"])
    ]

    test_scores = [
        float(test_metrics["accuracy"]),
        float(test_metrics["macro_f1"])
    ]

    x_positions = list(range(len(metric_names)))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    train_bars = ax.bar(
        [x - bar_width / 2 for x in x_positions],
        train_scores,
        bar_width,
        label="Training + Validation"
    )

    test_bars = ax.bar(
        [x + bar_width / 2 for x in x_positions],
        test_scores,
        bar_width,
        label="Unseen Test"
    )

    # Display the score above each bar.
    for bars in (train_bars, test_bars):

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom"
            )

    ax.set_xticks(
        x_positions,
        metric_names
    )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Final Model - Training vs Test Performance")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()

    path = FIGURES_DIR / "train_test_comparison.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Train/test comparison chart saved:", path)

    return path
