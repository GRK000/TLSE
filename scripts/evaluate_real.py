from pathlib import Path

from tlse.evaluate import evaluate


def main() -> None:
    evaluate(
        dataset_path=Path("data/processed/data2_landmarks.pickle"),
        model_path=Path("models/model_depth47.p"),
        report_path=Path("reports/real_eval_metrics.json"),
        warning="Use only with processed held-out or group-split real data.",
    )


if __name__ == "__main__":
    main()
