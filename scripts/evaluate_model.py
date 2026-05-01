import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def patch_legacy_random_forest(model: object) -> None:
    """Allow evaluation of older scikit-learn tree pickles on newer versions."""
    for estimator in getattr(model, "estimators_", []):
        if not hasattr(estimator, "monotonic_cst"):
            estimator.monotonic_cst = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained TLSE model on a processed landmark dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/landmarks.pickle"))
    parser.add_argument("--model", type=Path, default=Path("models/model_depth47.p"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/eval_metrics.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with args.dataset.open("rb") as file:
        dataset = pickle.load(file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        with args.model.open("rb") as file:
            model_obj = pickle.load(file)

    data = np.asarray(dataset["data"])
    labels = np.asarray(dataset["labels"])
    model = model_obj["model"]
    patch_legacy_random_forest(model)
    predictions = model.predict(data)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "num_classes": int(len(set(labels))),
        "num_samples": int(len(labels)),
        "classification_report": classification_report(labels, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(classification_report(labels, predictions, zero_division=0))
    print(confusion_matrix(labels, predictions))
    print(f"Saved metrics to {args.report_output}")


if __name__ == "__main__":
    main()
