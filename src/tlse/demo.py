import argparse
import pickle
from pathlib import Path

import numpy as np


def make_demo_dataset(output_path: Path, samples_per_class: int, random_state: int, num_groups: int) -> None:
    rng = np.random.default_rng(random_state)
    class_names = ["A", "B", "C", "None"]
    prototypes = rng.normal(0.0, 1.0, size=(len(class_names), 47)).astype(np.float32)

    data = []
    labels = []
    groups = []
    for idx, class_name in enumerate(class_names):
        samples = prototypes[idx] + rng.normal(0.0, 0.08, size=(samples_per_class, 47)).astype(np.float32)
        for sample_idx, sample in enumerate(samples):
            data.append(sample)
            labels.append(class_name)
            groups.append(f"session_{sample_idx % num_groups:02d}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(
            {
                "data": np.asarray(data, dtype=np.float32),
                "labels": np.asarray(labels),
                "groups": np.asarray(groups),
                "class_names": class_names,
                "description": "Synthetic 47-feature dataset for smoke tests only; not a real sign-language benchmark.",
            },
            file,
        )
    print(f"Saved demo dataset to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a deterministic synthetic demo dataset.")
    parser.add_argument("--output", type=Path, default=Path("demo/sample_data/landmarks_demo.pickle"))
    parser.add_argument("--samples-per-class", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-groups", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    make_demo_dataset(args.output, args.samples_per_class, args.random_state, args.num_groups)


if __name__ == "__main__":
    main()
