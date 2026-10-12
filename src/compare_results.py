
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results/task2")
COMPARISON_DIR = RESULTS_DIR / "comparison"


ALGORITHMS = {
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "k-NN": "knn",
    "SVM": "svm",
    "MLP": "mlp",
    "Voting Classifier": "voting_classifier",
}


def load_results():
    """Load the metrics.csv file from each algorithm."""

    results = []

    for algorithm_name, folder_name in ALGORITHMS.items():
        file_path = RESULTS_DIR / folder_name / "metrics.csv"

        if not file_path.exists():
            print(f"Warning: {file_path} not found.")
            continue

        metrics = pd.read_csv(file_path)

        for _, row in metrics.iterrows():
            results.append({
                "Algorithm": algorithm_name,
                "Dataset": row["Dataset"],
                "Accuracy": row["Accuracy"],
                "F1-score": row["F1-score"],
            })

    return pd.DataFrame(results)


def save_combined_results(results):
    """Save all algorithm results to one CSV file."""

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    output_path = COMPARISON_DIR / "combined_metrics.csv"

    results.to_csv(
        output_path,
        index=False,
    )

    print(f"\nCombined results saved to: {output_path}")


def create_accuracy_plot(results):
    """Create a bar plot comparing accuracy."""

    pivot = results.pivot(
        index="Algorithm",
        columns="Dataset",
        values="Accuracy",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("Algorithm Accuracy Comparison")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=45)
    plt.legend(title="Dataset")
    plt.tight_layout()

    output_path = COMPARISON_DIR / "accuracy_comparison.png"

    plt.savefig(output_path)
    plt.close()

    print(f"Accuracy plot saved to: {output_path}")


def create_f1_plot(results):
    """Create a bar plot comparing F1-score."""

    pivot = results.pivot(
        index="Algorithm",
        columns="Dataset",
        values="F1-score",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("Algorithm F1-score Comparison")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=45)
    plt.legend(title="Dataset")
    plt.tight_layout()

    output_path = COMPARISON_DIR / "f1_comparison.png"

    plt.savefig(output_path)
    plt.close()

    print(f"F1-score plot saved to: {output_path}")


def print_results(results):
    """Print the combined results in a readable format."""

    print("\nALL RESULTS")
    print("=" * 60)

    print(results.to_string(index=False))

    print("\nTEST RESULTS")
    print("=" * 60)

    test_results = results[
        results["Dataset"] == "Test"
    ]

    print(
        test_results.to_string(index=False)
    )


def main():
    """Load, combine, save, and plot algorithm results."""

    results = load_results()

    if results.empty:
        print("No results found.")
        return

    print_results(results)

    save_combined_results(results)

    create_accuracy_plot(results)

    create_f1_plot(results)


if __name__ == "__main__":
    main()
