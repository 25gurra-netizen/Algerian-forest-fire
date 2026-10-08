import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def load_and_clean_raw_csv(filepath):
    with open(filepath, "r") as f:
        raw_lines = f.readlines()

    cleaned_rows = []
    current_region = 0

    for line in raw_lines:
        s = line.strip()

        if not s:
            continue

        if "Bejaia Region" in s:
            current_region = 0
            continue

        if "Sidi-Bel Abbes" in s:
            current_region = 1
            continue

        if "day" in s and "month" in s:
            continue

        parts = [p.strip() for p in s.split(',')]

        if len(parts) >= 14:
            cleaned_rows.append(parts[:14] + [current_region])

    headers = [
        'day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain',
        'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes', 'Region'
    ]

    return pd.DataFrame(cleaned_rows, columns=headers)


def data_loading_and_initial_exploration(df):
    print("1: DATA LOADING & INITIAL EXPLORATION")
    
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("First 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nStatistical summary:")
    print(df.astype(str).describe())


def categorical_values_and_data_types(df):
    print("2: HANDLING CATEGORICAL VALUES & DATA TYPES")

    df['Classes'] = df['Classes'].str.strip()

    print("Target Class Distribution:")
    print(df['Classes'].value_counts())

    df['Target'] = df['Classes'].map({
        'not fire': 0,
        'fire': 1
    })

    num_cols = [
        'Temperature', 'RH', 'Ws', 'Rain',
        'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("\nDataset Summary Statistics:")
    print(df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])


def missing_values(df):
    print("3: MISSING VALUE ANALYSIS & IMPUTATION")

    num_cols = [
        'Temperature', 'RH', 'Ws', 'Rain',
        'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'
    ]

    missing_counts = df[num_cols].isna().sum()

    print("Missing values per numeric feature:")
    print(missing_counts)

def preprocessing_and_selection(df):
    print("4: FEATURE PREPROCESSING & SELECTION")

    df['Classes'] = df['Classes'].str.strip()

    df['Target'] = df['Classes'].map({
        'not fire': 0,
        'fire': 1
    })

    numerical_cols = [
        'Temperature',
        'RH',
        'Ws',
        'Rain',
        'FFMC',
        'DMC',
        'DC',
        'ISI',
        'BUI',
        'FWI'
    ]

    categorical_cols = ['Region']

    X = df[numerical_cols + categorical_cols]
    y = df['Target']

    print("Target:")
    print("0 = not fire")
    print("1 = fire")

    print("\nNumerical features:")
    print(numerical_cols)

    print("\nCategorical features:")
    print(categorical_cols)

    scaler = StandardScaler()

    X_numerical_scaled = pd.DataFrame(
        scaler.fit_transform(X[numerical_cols]),
        columns=numerical_cols
    )

    X_categorical = pd.get_dummies(
        X[categorical_cols],
        columns=categorical_cols,
        dtype=int
    )

    X_categorical = X_categorical.rename(
        columns={ 'Region_0': 'Bejaia',
                  'Region_1': 'Sidi_Bel_Abbes' }
    )

    X_processed = pd.concat(
        [X_numerical_scaled, X_categorical],
        axis=1
    )

    print("\nProcessed features:")
    print(list(X_processed.columns))

    print("\nFirst 3 rows of processed data:")
    print(X_processed.head(3))

def data_visualization(df):
    print("5: DATA VISUALIZATION")

    df['Classes'] = df['Classes'].str.strip()

    df['Target'] = df['Classes'].map({
        'not fire': 0,
        'fire': 1
    })

    num_cols = [
        'Temperature', 'RH', 'Ws', 'Rain',
        'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    corr_df = df[num_cols].copy()
    corr_df['Target'] = df['Target']

    corr_matrix = corr_df.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Matrix of Features and Fire Occurrence")
    plt.tight_layout()
    plt.savefig("results/task1/correlation_matrix.png")
    plt.close()

    print("-> Saved 'correlation_matrix.png'")

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        x=df['Classes'],
        y=df['FFMC'],
        hue=df['Classes'],
        palette='Set2',
        legend=False
    )

    plt.title("Fine Fuel Moisture Code (FFMC) Distribution by Class")
    plt.xlabel("Fire Class")
    plt.ylabel("FFMC Value")
    plt.tight_layout()
    plt.savefig("results/task1/ffmc_distribution.png")
    plt.close()

    print("-> Saved 'ffmc_distribution.png'")

def main():

    file_path = "data/raw/Algerian_forest_fires_dataset_UPDATE.csv"
    df = load_and_clean_raw_csv(file_path)

    while True:
        print("\nSelect a step to run:")
        print("1. Data loading & exploration")
        print("2. Categorical values & data types")
        print("3. Missing values & imputation")
        print("4. Feature preprocessing")
        print("5. Visualization")
        print("6. Run all steps")
        print("0. Exit")

        choice = input("\nEnter your choice: ")

        match choice:
            case "1":
                data_loading_and_initial_exploration(df)

            case "2":
                categorical_values_and_data_types(df)

            case "3":
                missing_values(df)

            case "4":
                preprocessing_and_selection(df)

            case "5":
                data_visualization(df)

            case "6":
                data_loading_and_initial_exploration(df)
                categorical_values_and_data_types(df)
                missing_values(df)
                preprocessing_and_selection(df)
                data_visualization(df)

            case "0":
                print("Exiting...")
                break

            case _:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
