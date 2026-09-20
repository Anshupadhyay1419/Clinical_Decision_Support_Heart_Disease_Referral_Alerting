import pandas as pd

from .data_loader import get_dataset
from .preprocessing import preprocess_data
from .modeling import train_model


# Decision thresholds
DEFAULT_THRESHOLD = 0.50
SELECTED_THRESHOLD = 0.40


def create_patient_data():
    """
    Create three sample patients for prediction.
    """

    patients = pd.DataFrame([
        {
            "age": 45,
            "sex": 1,
            "cp": 1,
            "trestbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalach": 170,
            "exang": 0,
            "oldpeak": 0.5,
            "slope": 2,
            "ca": 0,
            "thal": 3
        },
        {
            "age": 60,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 250,
            "fbs": 0,
            "restecg": 1,
            "thalach": 120,
            "exang": 1,
            "oldpeak": 2.5,
            "slope": 1,
            "ca": 2,
            "thal": 2
        },
        {
            "age": 52,
            "sex": 0,
            "cp": 2,
            "trestbps": 130,
            "chol": 220,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "ca": 1,
            "thal": 3
        }
    ])

    patients.index = [
        "Patient A",
        "Patient B",
        "Patient C"
    ]

    return patients


def preprocess_patients(
    patients,
    training_columns,
    scaler
):
    """
    Apply the same preprocessing used during model training.
    """

    patients = patients.copy()

    # One-hot encode categorical columns
    patients = pd.get_dummies(
        patients,
        columns=[
            "cp",
            "restecg",
            "slope",
            "thal"
        ],
        drop_first=False,
        dtype=int
    )

    # Make sure patient data has exactly the same
    # columns as the training data.
    patients = patients.reindex(
        columns=training_columns,
        fill_value=0
    )

    # Apply the already-fitted scaler.
    patients_scaled = scaler.transform(patients)

    return patients_scaled


def make_prediction(
    model,
    patients_scaled
):
    """
    Generate disease probabilities.
    """

    probabilities = model.predict_proba(
        patients_scaled
    )[:, 1]

    return probabilities


def display_predictions(
    patients,
    probabilities
):
    """
    Display predictions using both thresholds.
    """

    print("\n" + "=" * 80)
    print("PATIENT PREDICTIONS")
    print("=" * 80)

    for patient, probability in zip(
        patients.index,
        probabilities
    ):

        default_decision = (
            "ALERT"
            if probability >= DEFAULT_THRESHOLD
            else "NO ALERT"
        )

        selected_decision = (
            "ALERT"
            if probability >= SELECTED_THRESHOLD
            else "NO ALERT"
        )

        print(f"\n{patient}")

        print(
            f"Heart disease probability: "
            f"{probability:.4f} "
            f"({probability * 100:.2f}%)"
        )

        print(
            f"Decision at 0.50: "
            f"{default_decision}"
        )

        print(
            f"Decision at 0.40: "
            f"{selected_decision}"
        )


def main():

    print("=" * 80)
    print("CLINICAL DECISION SUPPORT - PATIENT PREDICTION")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = get_dataset()

    # --------------------------------------------------
    # 2. Preprocess training data
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
    # 4. Create sample patients
    # --------------------------------------------------

    patients = create_patient_data()

    print("\nSample patients:")
    print(patients)

    # --------------------------------------------------
    # 5. Apply same preprocessing
    # --------------------------------------------------

    patients_scaled = preprocess_patients(
        patients,
        feature_names,
        scaler
    )

    # --------------------------------------------------
    # 6. Predict probabilities
    # --------------------------------------------------

    probabilities = make_prediction(
        model,
        patients_scaled
    )

    # --------------------------------------------------
    # 7. Display results
    # --------------------------------------------------

    display_predictions(
        patients,
        probabilities
    )

    print("\n" + "=" * 80)
    print("PATIENT PREDICTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()