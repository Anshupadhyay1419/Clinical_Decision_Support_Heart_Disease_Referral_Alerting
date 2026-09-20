import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
COLUMN_NAMES = ["age","sex", "cp", "trestbps","chol","fbs", "restecg","thalach","exang","oldpeak", "slope", "ca","thal","target"]

def load_data ():
    df = pd.read_csv(DATA_URL, names=COLUMN_NAMES , na_values="?")
    return df

def prepare_target(df):
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)
    return df

def get_dataset():
    df = load_data()
    df = prepare_target(df)
    return df

if __name__ == "__main__":
    df = get_dataset()
    print("=" * 60)
    print("HEART DISEASE DATASET")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nDataset loaded successfully!")
