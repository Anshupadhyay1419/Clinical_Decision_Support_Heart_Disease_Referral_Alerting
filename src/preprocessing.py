import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLUMNS = ["cp","restecg","slope","thal"]

def handle_missing_values(df):
    df = df.copy()
    for column in ["ca", "thal"]:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df


def prepare_features(df):
    df = df.copy()
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def encode_categorical_features(X):
    X = X.copy()

    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False,
        dtype=int
    )

    return X


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(df):
    # 1. Handle missing values
    df = handle_missing_values(df)

    # 2. Separate features and target
    X, y = prepare_features(df)

    # 3. One-hot encode categorical variables
    X = encode_categorical_features(X)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        X_train.columns
    )


if __name__ == "__main__":
    from data_loader import get_dataset

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # Load dataset
    df = get_dataset()

    print("\nOriginal dataset shape:")
    print(df.shape)

    # Handle missing values
    df = handle_missing_values(df)

    print("\nMissing values after imputation:")
    print(df.isnull().sum())

    # Prepare features
    X, y = prepare_features(df)

    print("\nOriginal feature shape:")
    print(X.shape)

    # Encode categorical features
    X = encode_categorical_features(X)

    print("\nFeature shape after one-hot encoding:")
    print(X.shape)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining data shape:")
    print(X_train.shape)

    print("\nTesting data shape:")
    print(X_test.shape)

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    print("\nScaled training data shape:")
    print(X_train_scaled.shape)

    print("\nScaled testing data shape:")
    print(X_test_scaled.shape)

    print("\nPreprocessing completed successfully!")