import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix
)

from .data_loader import get_dataset
from .preprocessing import preprocess_data
from .modeling import train_model


FIGURE_DIR = "outputs/figures"
METRICS_DIR = "outputs/metrics"


def create_output_directories():
    """Create output directories."""

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def get_probabilities(model, X_test):
    """
    Get probability of heart disease for each test patient.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    return probabilities


def calculate_threshold_metrics(
    y_test,
    probabilities,
    threshold
):
    """
    Calculate precision, recall and referral rate
    for a particular probability threshold.
    """

    # Convert probability into prediction
    y_pred = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Number of patients referred/alerted
    referral_rate = y_pred.mean()

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "referral_rate": referral_rate,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    }


def analyze_thresholds(y_test, probabilities):
    """
    Compare model performance across multiple thresholds.
    """

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_threshold_metrics(
            y_test,
            probabilities,
            threshold
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    return results_df


def display_threshold_results(results_df):
    """
    Display threshold comparison table.
    """

    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)

    display_df = results_df.copy()

    display_df["precision"] = (
        display_df["precision"].round(3)
    )

    display_df["recall"] = (
        display_df["recall"].round(3)
    )

    display_df["referral_rate"] = (
        display_df["referral_rate"].round(3)
    )

    print(display_df.to_string(index=False))


def plot_precision_recall(results_df):
    """
    Plot precision and recall against threshold.
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        results_df["threshold"],
        results_df["recall"],
        marker="o",
        label="Recall"
    )

    plt.xlabel("Probability Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall at Different Thresholds")

    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "threshold_precision_recall.png"
        ),
        dpi=300
    )

    plt.show()
    plt.close()


def plot_referral_rate(results_df):
    """
    Plot referral rate against threshold.
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        results_df["threshold"],
        results_df["referral_rate"],
        marker="o"
    )

    plt.xlabel("Probability Threshold")
    plt.ylabel("Referral Rate")
    plt.title("Referral Rate at Different Thresholds")

    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "threshold_referral_rate.png"
        ),
        dpi=300
    )

    plt.show()
    plt.close()


def save_threshold_results(results_df):
    """
    Save threshold analysis results as CSV.
    """

    output_path = os.path.join(
        METRICS_DIR,
        "threshold_analysis.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nThreshold results saved to: {output_path}"
    )


def main():

    create_output_directories()

    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = get_dataset()

    # --------------------------------------------------
    # 2. Preprocess data
    # --------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = preprocess_data(df)

    # --------------------------------------------------
    # 3. Train model
    # --------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    print("\nModel trained successfully.")

    # --------------------------------------------------
    # 4. Get probabilities
    # --------------------------------------------------

    probabilities = get_probabilities(
        model,
        X_test
    )

    print("\nFirst 10 predicted probabilities:")

    for i, probability in enumerate(
        probabilities[:10]
    ):
        print(
            f"Patient {i + 1}: "
            f"{probability:.4f}"
        )

    # --------------------------------------------------
    # 5. Analyze thresholds
    # --------------------------------------------------

    results_df = analyze_thresholds(
        y_test,
        probabilities
    )

    # --------------------------------------------------
    # 6. Display results
    # --------------------------------------------------

    display_threshold_results(
        results_df
    )

    # --------------------------------------------------
    # 7. Create plots
    # --------------------------------------------------

    plot_precision_recall(
        results_df
    )

    plot_referral_rate(
        results_df
    )

    # --------------------------------------------------
    # 8. Save results
    # --------------------------------------------------

    save_threshold_results(
        results_df
    )

    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()