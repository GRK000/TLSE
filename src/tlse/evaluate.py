from __future__ import annotations

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


def _require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {path}")
    return path


def load_model_artifact(model_path: Path) -> dict:
    """Load a trusted TLSE model pickle.

    Pickle can execute arbitrary code while loading. Only use artifacts created
    locally or received from a trusted source.
    """
    _require_file(model_path, "Model artifact")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        with model_path.open("rb") as file:
            model_obj = pickle.load(file)
    if not isinstance(model_obj, dict) or "model" not in model_obj:
        raise ValueError(f"Model artifact must contain a 'model' entry: {model_path}")
    return model_obj


def evaluate(dataset_path: Path, model_path: Path, report_path: Path, warning: str | None = None) -> dict:
    from tlse.train import load_dataset

    data, labels, _, _ = load_dataset(dataset_path)
    model_obj = load_model_artifact(model_path)

    model = model_obj["model"]
    patch_legacy_random_forest(model)
    predictions = model.predict(data)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "num_classes": int(len(set(labels))),
        "num_samples": int(len(labels)),
        "warning": warning,
        "classification_report": classification_report(labels, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(classification_report(labels, predictions, zero_division=0))
    print(confusion_matrix(labels, predictions))
    print(f"Saved metrics to {report_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained TLSE model on a processed landmark dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/landmarks.pickle"))
    parser.add_argument("--model", type=Path, default=Path("models/model_depth47.p"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/eval_metrics.json"))
    parser.add_argument("--warning", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args.dataset, args.model, args.report_output, args.warning)


if __name__ == "__main__":
    main()
