import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with path.open("rb") as file:
        dataset = pickle.load(file)

    data = np.asarray(dataset["data"])
    labels = np.asarray(dataset["labels"])
    class_names = list(dataset.get("class_names", sorted(set(labels))))
    return data, labels, class_names


def train_model(
    data: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    n_iter: int,
    cv: int,
    n_jobs: int,
) -> tuple[RandomForestClassifier, dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.20,
        random_state=random_state,
        stratify=labels,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=1,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "num_classes": int(len(set(labels))),
        "num_samples": int(len(labels)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "best_params": search.best_params_,
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))
    return model, metrics


def save_outputs(model_path: Path, report_path: Path, model: RandomForestClassifier, class_names: list[str], metrics: dict) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump({"model": model, "class_names": class_names}, file)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the TLSE Random Forest recognizer.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/landmarks.pickle"))
    parser.add_argument("--model-output", type=Path, default=Path("models/model_depth47.p"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data, labels, class_names = load_dataset(args.dataset)
    model, metrics = train_model(data, labels, args.random_state, args.n_iter, args.cv, args.n_jobs)
    save_outputs(args.model_output, args.report_output, model, class_names, metrics)
    print(f"Saved model to {args.model_output}")
    print(f"Saved metrics to {args.report_output}")


if __name__ == "__main__":
    main()
