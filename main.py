from src.data_loader import get_dataset
from src.preprocessing import preprocess_data
from src.modeling import train_model
from src.evaluation import (
    evaluate_classification,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.threshold_analysis import (
    get_probabilities,
    analyze_thresholds,
    display_threshold_results,
)
from src.patient_prediction import (
    create_patient_data,
    preprocess_patients,
    make_prediction,
    display_predictions,
)


def main():

    print("=" * 80)
    print("HEART DISEASE CLINICAL DECISION SUPPORT SYSTEM")
    print("=" * 80)

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    print("\n[1/6] Loading dataset...")

    df = get_dataset()

    print(f"Dataset shape: {df.shape}")
    print("Dataset loaded successfully.")

    # ==================================================
    # 2. PREPROCESS DATA
    # ==================================================

    print("\n[2/6] Preprocessing data...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = preprocess_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print("Preprocessing completed.")

    # ==================================================
    # 3. TRAIN MODEL
    # ==================================================

    print("\n[3/6] Training Logistic Regression model...")

    model = train_model(
        X_train,
        y_train
    )

    print("Model trained successfully.")

    # ==================================================
    # 4. MODEL EVALUATION
    # ==================================================

    print("\n[4/6] Evaluating model...")

    y_pred = evaluate_classification(
        model,
        X_test,
        y_test
    )

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    probabilities, auc_score = plot_roc_curve(
        model,
        X_test,
        y_test
    )

    print(f"\nROC-AUC: {auc_score:.4f}")

    # ==================================================
    # 5. THRESHOLD ANALYSIS
    # ==================================================

    print("\n[5/6] Analyzing decision thresholds...")

    results_df = analyze_thresholds(
        y_test,
        probabilities
    )

    display_threshold_results(
        results_df
    )

    print("\nThreshold analysis completed.")

    # ==================================================
    # 6. SAMPLE PATIENT PREDICTIONS
    # ==================================================

    print("\n[6/6] Generating sample patient predictions...")

    patients = create_patient_data()

    patients_scaled = preprocess_patients(
        patients,
        feature_names,
        scaler
    )

    patient_probabilities = make_prediction(
        model,
        patients_scaled
    )

    display_predictions(
        patients,
        patient_probabilities
    )

    # ==================================================
    # COMPLETE
    # ==================================================

    print("\n" + "=" * 80)
    print("COMPLETE CDS PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()