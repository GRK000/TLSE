from pathlib import Path

from tlse.evaluate import evaluate


def main() -> None:
    evaluate(
        dataset_path=Path("Pickles/data.pickle"),
        model_path=Path("Pickles/model.p"),
        report_path=Path("reports/legacy_model_eval_metrics.json"),
        warning="Legacy in-sample evaluation. Do not treat this as held-out generalization.",
    )


if __name__ == "__main__":
    main()
