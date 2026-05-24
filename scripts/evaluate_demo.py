from pathlib import Path
from types import SimpleNamespace

from tlse.cli import run_demo


def main() -> None:
    run_demo(
        SimpleNamespace(
            dataset=Path("demo/sample_data/landmarks_demo.pickle"),
            model_output=Path("models/demo_model.p"),
            report_output=Path("reports/demo_metrics.json"),
            samples_per_class=40,
            num_groups=4,
            random_state=42,
            n_iter=2,
            cv=2,
            n_jobs=1,
            test_size=0.25,
        )
    )


if __name__ == "__main__":
    main()
