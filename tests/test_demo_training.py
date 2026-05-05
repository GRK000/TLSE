import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import tlse
from tlse.demo import make_demo_dataset
from tlse.train import build_model_metadata, load_dataset, save_outputs, train_model


TEST_OUTPUT_DIR = Path("reports/pytest")
DEMO_DATASET = Path("demo/sample_data/landmarks_demo.pickle")


def test_package_imports():
    assert tlse.__version__ == "0.1.0"


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "tlse", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "demo" in result.stdout
    assert "evaluate" in result.stdout
    assert "benchmark" in result.stdout


def test_demo_dataset_fixture_exists():
    assert DEMO_DATASET.exists()
    assert DEMO_DATASET.is_file()


def test_demo_dataset_contains_groups():
    dataset_path = TEST_OUTPUT_DIR / "landmarks_demo_groups.pickle"

    make_demo_dataset(dataset_path, samples_per_class=8, random_state=42, num_groups=4)
    data, labels, class_names, groups = load_dataset(dataset_path)

    assert data.shape == (32, 47)
    assert len(labels) == 32
    assert class_names == ["A", "B", "C", "None"]
    assert groups is not None
    assert len(set(groups)) == 4


def test_load_dataset_reports_missing_file():
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(TEST_OUTPUT_DIR / "missing.pickle")


@pytest.mark.slow
def test_training_writes_model_with_metadata_and_metrics():
    dataset_path = TEST_OUTPUT_DIR / "landmarks_demo_train.pickle"
    model_path = TEST_OUTPUT_DIR / "demo_model.p"
    report_path = TEST_OUTPUT_DIR / "metrics.json"

    make_demo_dataset(dataset_path, samples_per_class=12, random_state=42, num_groups=4)
    data, labels, class_names, groups = load_dataset(dataset_path)
    model, metrics = train_model(
        data=data,
        labels=labels,
        random_state=42,
        n_iter=1,
        cv=2,
        n_jobs=1,
        groups=groups,
        split_strategy="group",
        test_size=0.25,
    )
    metadata = build_model_metadata(dataset_path, report_path)
    save_outputs(model_path, report_path, model, class_names, metrics, metadata)

    assert model_path.exists()
    assert report_path.exists()
    assert metrics["accuracy"] >= 0.95
    assert metrics["macro_f1"] >= 0.95
    assert metrics["split_strategy"] == "GroupShuffleSplit"

    with model_path.open("rb") as file:
        artifact = pickle.load(file)

    assert artifact["class_names"] == class_names
    assert artifact["metadata"]["feature_version"] == "depth47-v1"
    assert artifact["metadata"]["dataset_hash"]
    assert artifact["metadata"]["metrics_path"] == str(report_path)
