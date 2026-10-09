
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT/"data"/"processed"/"forest_fires_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT/"data"/"processed"/"splits"

RANDOM_STATE = 42

# Features used by the classifiers
NUMERICAL_FEATURES = [
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI",
]

REGION_FEATURES = [
    "Bejaia",
    "Sidi_Bel_Abbes",
]

TARGET = "Target"

def load_and_prepare_data():
    """Load the cleaned CSV and prepare X and y."""

    print("\n1. LOADING AND PREPARING DATA")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}\n"
            "Run preprocessing.py and select option 7 first."
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Separate the target from the input features.
    y = df[TARGET].astype(int)

    # Select numerical features.
    X_numerical = df[NUMERICAL_FEATURES].copy()

    # Encode Region as two readable one-hot features.
    X_region = pd.get_dummies(
        df["Region"],
        prefix="Region",
        dtype=int
    )

    # Ensure both region columns exist and have a consistent order.
    X_region = X_region.reindex(
        columns=["Region_0", "Region_1"],
        fill_value=0
    )

    X_region = X_region.rename(
        columns={
            "Region_0": "Bejaia",
            "Region_1": "Sidi_Bel_Abbes",
        }
    )

    # Combine numerical and categorical features.
    X = pd.concat(
        [X_numerical, X_region],
        axis=1
    )

    print(f"Number of features: {X.shape[1]}")
    print(f"Features: {list(X.columns)}")

    return X, y

def split_data(X, y):
    """Split the data into training, validation, and test sets."""

    print("\n2. SPLITTING DATA")

    # First split: 70% training, 30% temporary data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Second split: divide the remaining 30% equally.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Total samples:      {len(X_train) + len(X_val) + len(X_test)}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

def save_unscaled_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):
    """Save the original, unscaled splits."""

    print("\n3. SAVING UNSCALED DATA")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train.csv": (X_train, y_train),
        "validation.csv": (X_val, y_val),
        "test.csv": (X_test, y_test),
    }

    for filename, (X_split, y_split) in splits.items():
        # Store the target alongside the features.
        dataset = X_split.copy()
        dataset[TARGET] = y_split

        output_path = OUTPUT_DIR / filename
        dataset.to_csv(output_path, index=False)

        print(f"Saved: {output_path.name}")


def scale_and_save_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):
    """Scale numerical features and save the scaled splits."""

    print("\n4. SCALING DATA")

    scaler = StandardScaler()

    # Fit the scaler ONLY on the training data.
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(
        X_train[NUMERICAL_FEATURES]
    )

    # Use the same fitted scaler for validation and test data.
    X_val_scaled[NUMERICAL_FEATURES] = scaler.transform(
        X_val[NUMERICAL_FEATURES]
    )

    X_test_scaled[NUMERICAL_FEATURES] = scaler.transform(
        X_test[NUMERICAL_FEATURES]
    )

    # Region features remain as 0/1 indicators.
    # Only numerical features are scaled.

    # Save the fitted scaler for future reuse.
    scaler_path = OUTPUT_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    print(f"Saved fitted scaler: {scaler_path.name}")

    # Save the scaled splits, including their targets.
    splits = {
        "train_scaled.csv": (X_train_scaled, y_train),
        "validation_scaled.csv": (X_val_scaled, y_val),
        "test_scaled.csv": (X_test_scaled, y_test),
    }

    for filename, (X_split, y_split) in splits.items():
        dataset = X_split.copy()
        dataset[TARGET] = y_split

        output_path = OUTPUT_DIR / filename
        dataset.to_csv(output_path, index=False)

        print(f"Saved: {output_path.name}")

    print("\nScaling verification:")
    print("Training feature means (should be approximately zero):")
    print(X_train_scaled[NUMERICAL_FEATURES].mean().round(3))

    print("\nTraining feature standard deviations (should be approximately one):")
    print(X_train_scaled[NUMERICAL_FEATURES].std(ddof=0).round(3))

def main():
    """Run the complete data preparation pipeline."""

    X, y = load_and_prepare_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, y)

    save_unscaled_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    scale_and_save_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    print("\nData preparation completed successfully!")

if __name__ == "__main__":
    main()
