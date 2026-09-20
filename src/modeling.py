import os

import joblib
from sklearn.linear_model import LogisticRegression

from .data_loader import get_dataset
from .preprocessing import preprocess_data


MODEL_DIR = "outputs/metrics"


def train_model(X_train, y_train):
    """
    Train Logistic Regression model.
    """

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def save_model(model):
    """
    Save trained model to disk.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        "logistic_regression_model.pkl"
    )

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")


def display_coefficients(model, feature_names):
    """
    Display Logistic Regression coefficients.
    """

    coefficients = model.coef_[0]

    print("\n" + "=" * 60)
    print("MODEL COEFFICIENTS")
    print("=" * 60)

    for feature, coefficient in zip(
        feature_names,
        coefficients
    ):
        print(
            f"{feature:25s} : {coefficient:.4f}"
        )


def main():

    print("=" * 60)
    print("LOGISTIC REGRESSION MODEL")
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

    print("\nTraining data shape:")
    print(X_train.shape)

    print("\nTesting data shape:")
    print(X_test.shape)

    # Train model
    model = train_model(
        X_train,
        y_train
    )

    print("\nModel trained successfully!")

    # Display coefficients
    display_coefficients(
        model,
        feature_names
    )

    # Save model
    save_model(model)


if __name__ == "__main__":
    main()