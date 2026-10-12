
from pathlib import Path

import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


DATA_DIR = Path("data/processed/splits")
RESULTS_DIR = Path("results/task2/voting_classifier")
TARGET = "Target"


def load_data():
    """Load the scaled train, validation, and test datasets."""

    train = pd.read_csv(DATA_DIR / "train_scaled.csv")
    validation = pd.read_csv(DATA_DIR / "validation_scaled.csv")
    test = pd.read_csv(DATA_DIR / "test_scaled.csv")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_validation = validation.drop(columns=[TARGET])
    y_validation = validation[TARGET]

    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


def create_model():
    """Create the individual classifiers and the Voting Classifier."""

    decision_tree = DecisionTreeClassifier(
        random_state=42
    )

    knn = KNeighborsClassifier(
        n_neighbors=5
    )

    svm = SVC(
        probability=True,
        random_state=42
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=42
    )

    model = VotingClassifier(
        estimators=[
            ("decision_tree", decision_tree),
            ("knn", knn),
            ("svm", svm),
            ("mlp", mlp),
        ],
        voting="soft",
    )

    return model


def evaluate_model(model, X, y, dataset_name):
    """Evaluate the model and return its metrics and confusion matrix."""

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    confusion = confusion_matrix(y, predictions)

    print(f"\n{dataset_name} RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["Not fire", "Fire"],
        )
    )

    print("Confusion matrix:")
    print(confusion)

    return accuracy, f1, confusion


def save_results(
    validation_accuracy,
    validation_f1,
    validation_cm,
    test_accuracy,
    test_f1,
    test_cm,
):
    """Save metrics and confusion matrices to the results directory."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame({
        "Dataset": ["Validation", "Test"],
        "Accuracy": [
            validation_accuracy,
            test_accuracy,
        ],
        "F1": [
            validation_f1,
            test_f1,
        ],
    })

    metrics.to_csv(
        RESULTS_DIR / "metrics.csv",
        index=False,
    )

    pd.DataFrame(
        validation_cm,
        index=["Not fire", "Fire"],
        columns=["Not fire", "Fire"],
    ).to_csv(
        RESULTS_DIR / "validation_confusion_matrix.csv"
    )

    pd.DataFrame(
        test_cm,
        index=["Not fire", "Fire"],
        columns=["Not fire", "Fire"],
    ).to_csv(
        RESULTS_DIR / "test_confusion_matrix.csv"
    )


def main():
    """Train and evaluate the Voting Classifier."""

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = load_data()

    model = create_model()

    model.fit(X_train, y_train)

    (
        validation_accuracy,
        validation_f1,
        validation_cm,
    ) = evaluate_model(
        model,
        X_validation,
        y_validation,
        "VALIDATION",
    )

    (
        test_accuracy,
        test_f1,
        test_cm,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        "TEST",
    )

    save_results(
        validation_accuracy,
        validation_f1,
        validation_cm,
        test_accuracy,
        test_f1,
        test_cm,
    )


if __name__ == "__main__":
    main()

