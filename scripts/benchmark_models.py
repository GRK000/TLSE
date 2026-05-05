import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tlse.train import load_dataset, split_dataset


def build_models(random_state: int) -> dict[str, object]:
    return {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVC": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=random_state)),
        "KNeighborsClassifier": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "MLPClassifier": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=random_state),
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline classifiers on a processed TLSE dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/landmarks.pickle"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/model_benchmark.json"))
    parser.add_argument("--split-strategy", choices=["random", "group"], default="random")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data, labels, _, groups = load_dataset(args.dataset)
    x_train, x_test, y_train, y_test, split_metadata = split_dataset(
        data,
        labels,
        groups,
        random_state=args.random_state,
        test_size=args.test_size,
        split_strategy=args.split_strategy,
    )

    results = {"dataset": str(args.dataset), **split_metadata, "models": {}}
    for name, model in build_models(args.random_state).items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = model.predict(x_test)
        predict_seconds = time.perf_counter() - started

        results["models"][name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "test_samples": int(len(y_test)),
        }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    for name, metrics in results["models"].items():
        print(f"{name}: macro_f1={metrics['macro_f1']:.3f}, accuracy={metrics['accuracy']:.3f}")
    print(f"Saved benchmark to {args.report_output}")


if __name__ == "__main__":
    main()
