
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        ConfusionMatrixDisplay,
        classification_report,
        )
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT/"data"/"processed"/"splits"
RESULT_DIR = PROJECT_ROOT/"results"/"task2"/"decision_tree"

TARGET = "Target"

def load_data():
    # Load training, validation and test datasets

    train_df = pd.read_csv(DATA_DIR/"train.csv")
    val_df = pd.read_csv(DATA_DIR/"validation.csv")
    test_df = pd.read_csv(DATA_DIR/"test.csv")

    # Separate features (X) from targets (Y)
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train):
    """Train the Decision Tree classifier."""

    print("\n2. TRAINING DECISION TREE")

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Decision Tree training completed.")

    return model


def evaluate_model(model, X, y, dataset_name):
    """Evaluate the model and print its performance metrics."""

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)

    print(f"\n{dataset_name} RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Not fire", "Fire"],
        )
    )

    return predictions, accuracy, f1


def save_confusion_matrix(y, predictions, dataset_name):
    """Create and save a confusion matrix."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not fire", "Fire"],
    )

    display.plot()

    plt.title(f"Decision Tree - {dataset_name} Confusion Matrix")
    plt.tight_layout()

    output_path = RESULT_DIR / f"{dataset_name.lower()}_confusion_matrix.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix: {output_path}")


def save_results(results):
    """Save evaluation metrics to a CSV file."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    output_path = RESULT_DIR / "metrics.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved metrics: {output_path}")


def save_tree_visualization(model, feature_names):
    """Save a visualization of the trained Decision Tree."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(24, 12))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Not fire", "Fire"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
    )

    plt.title("Decision Tree (First Three Levels)")
    plt.tight_layout()

    output_path = RESULT_DIR / "decision_tree.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved tree visualization: {output_path}")


def main():
    """Run the Decision Tree training and evaluation pipeline."""

    print("=" * 50)
    print("DECISION TREE CLASSIFIER")
    print("=" * 50)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Train using only the training set.
    model = train_model(X_train, y_train)

    # Evaluate on the validation set.
    val_predictions, val_accuracy, val_f1 = evaluate_model(
        model,
        X_val,
        y_val,
        "Validation",
    )

    save_confusion_matrix(
        y_val,
        val_predictions,
        "Validation",
    )

    # Evaluate on the test set.
    test_predictions, test_accuracy, test_f1 = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
    )

    save_confusion_matrix(
        y_test,
        test_predictions,
        "Test",
    )

    # Save metrics for later comparison with other classifiers.
    results = [
        {
            "Model": "Decision Tree",
            "Dataset": "Validation",
            "Accuracy": val_accuracy,
            "F1-score": val_f1,
        },
        {
            "Model": "Decision Tree",
            "Dataset": "Test",
            "Accuracy": test_accuracy,
            "F1-score": test_f1,
        },
    ]

    save_results(results)

    # Save a visualization of the trained tree.
    save_tree_visualization(
        model,
        X_train.columns.tolist(),
    )

    print("\nDecision Tree pipeline completed successfully.")


if __name__ == "__main__":
    main()
