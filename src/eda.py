import os

import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import get_dataset


# Output directory for EDA figures
FIGURE_DIR = "outputs/figures"


def create_output_directory():
    """Create the output directory if it does not exist."""

    os.makedirs(FIGURE_DIR, exist_ok=True)


def target_distribution(df):
    """Plot distribution of heart disease target."""

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="target"
    )

    plt.title("Heart Disease Target Distribution")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel("Number of Patients")

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURE_DIR, "target_distribution.png"),
        dpi=300
    )

    plt.show()
    plt.close()


def target_proportion(df):
    """Display percentage of patients with and without heart disease."""

    proportions = df["target"].value_counts(normalize=True) * 100

    print("\nTarget proportions:")
    print(proportions)


def numerical_distributions(df):
    """Plot histograms of important numerical variables."""

    numerical_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    for column in numerical_columns:

        plt.figure(figsize=(7, 5))

        sns.histplot(
            data=df,
            x=column,
            kde=True
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                f"{column}_distribution.png"
            ),
            dpi=300
        )

        plt.show()
        plt.close()


def boxplots(df):
    """Plot boxplots to identify possible outliers."""

    numerical_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    for column in numerical_columns:

        plt.figure(figsize=(7, 5))

        sns.boxplot(
            data=df,
            y=column
        )

        plt.title(f"Boxplot of {column}")
        plt.ylabel(column)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                f"{column}_boxplot.png"
            ),
            dpi=300
        )

        plt.show()
        plt.close()


def correlation_matrix(df):
    """Create correlation matrix for numerical variables."""

    plt.figure(figsize=(12, 9))

    correlation = df.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "correlation_matrix.png"
        ),
        dpi=300
    )

    plt.show()
    plt.close()


def categorical_distributions(df):
    """Plot distributions of categorical variables."""

    categorical_columns = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]

    for column in categorical_columns:

        plt.figure(figsize=(7, 5))

        sns.countplot(
            data=df,
            x=column,
            hue="target"
        )

        plt.title(f"{column} Distribution by Heart Disease")
        plt.xlabel(column)
        plt.ylabel("Number of Patients")

        plt.legend(
            title="Heart Disease",
            labels=["No", "Yes"]
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                f"{column}_vs_target.png"
            ),
            dpi=300
        )

        plt.show()
        plt.close()


def run_eda(df):
    """Run the complete EDA pipeline."""

    create_output_directory()

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset information:")
    print(df.info())

    print("\nStatistical summary:")
    print(df.describe())

    print("\nMissing values:")
    print(df.isnull().sum())

    # Target analysis
    target_distribution(df)
    target_proportion(df)

    # Numerical variables
    numerical_distributions(df)

    # Outlier analysis
    boxplots(df)

    # Correlation analysis
    correlation_matrix(df)

    # Categorical variables
    categorical_distributions(df)

    print("\nEDA completed successfully!")
    print(f"Figures saved in: {FIGURE_DIR}")


if __name__ == "__main__":

    # Load dataset
    df = get_dataset()

    # Run EDA
    run_eda(df)