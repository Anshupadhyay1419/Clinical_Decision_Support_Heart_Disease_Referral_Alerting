import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from .data_loader import get_dataset
from .preprocessing import preprocess_data
from .modeling import train_model


FIGURE_DIR = "outputs/figures"
METRICS_DIR = "outputs/metrics"


def create_output_directories():
    """Create directories for evaluation outputs."""

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def evaluate_classification(model, X_test, y_test):
    """
    Calculate classification metrics.
    """

    # Class predictions using default threshold = 0.5
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

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

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Disease",
                "Disease"
            ],
            zero_division=0
        )
    )

    return y_pred


def plot_confusion_matrix(y_test, y_pred):
    """Create and save confusion matrix."""

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "No Disease",
            "Disease"
        ],
        yticklabels=[
            "No Disease",
            "Disease"
        ]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.show()
    plt.close()

    print("\nConfusion matrix:")
    print(cm)


def plot_roc_curve(model, X_test, y_test):
    """Create and save ROC curve."""

    # Probability of class 1
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    # Calculate AUC
    auc_score = roc_auc_score(
        y_test,
        y_probability
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"Logistic Regression (AUC = {auc_score:.3f})"
    )

    # Random classifier line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "roc_curve.png"
        ),
        dpi=300
    )

    plt.show()
    plt.close()

    print(f"\nROC-AUC: {auc_score:.4f}")

    return y_probability, auc_score


def save_metrics(
    accuracy,
    precision,
    recall,
    f1,
    auc_score
):
    """Save evaluation metrics to a text file."""

    metrics_path = os.path.join(
        METRICS_DIR,
        "evaluation_metrics.txt"
    )

    with open(metrics_path, "w") as file:

        file.write("HEART DISEASE MODEL EVALUATION\n")
        file.write("=" * 40 + "\n\n")

        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1-score: {f1:.4f}\n")
        file.write(f"ROC-AUC: {auc_score:.4f}\n")

    print(f"\nMetrics saved to: {metrics_path}")


def main():

    create_output_directories()

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Load dataset
    df = get_dataset()

    # Preprocess dataset
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = preprocess_data(df)

    # Train model
    model = train_model(
        X_train,
        y_train
    )

    print("\nModel trained successfully.")

    # Classification metrics
    y_pred = evaluate_classification(
        model,
        X_test,
        y_test
    )

    # Calculate metrics again for saving
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_test,
        y_pred
    )

    # ROC and AUC
    y_probability, auc_score = plot_roc_curve(
        model,
        X_test,
        y_test
    )

    # Save metrics
    save_metrics(
        accuracy,
        precision,
        recall,
        f1,
        auc_score
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()