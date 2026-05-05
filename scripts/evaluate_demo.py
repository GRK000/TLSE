from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tlse.demo import make_demo_dataset
from tlse.train import build_model_metadata, load_dataset, save_outputs, train_model


def main() -> None:
    dataset_path = Path("demo/sample_data/landmarks_demo.pickle")
    model_path = Path("models/demo_model.p")
    report_path = Path("reports/demo_metrics.json")

    make_demo_dataset(dataset_path, samples_per_class=40, random_state=42, num_groups=4)
    data, labels, class_names, groups = load_dataset(dataset_path)
    model, metrics = train_model(
        data=data,
        labels=labels,
        random_state=42,
        n_iter=2,
        cv=2,
        n_jobs=1,
        groups=groups,
        split_strategy="group",
        test_size=0.25,
    )
    save_outputs(model_path, report_path, model, class_names, metrics, build_model_metadata(dataset_path, report_path))
    print(f"Demo evaluation written to {report_path}")


if __name__ == "__main__":
    main()
