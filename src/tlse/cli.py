from __future__ import annotations

import argparse
from pathlib import Path


def run_demo(args: argparse.Namespace) -> None:
    from tlse.demo import make_demo_dataset
    from tlse.train import build_model_metadata, load_dataset, save_outputs, train_model

    make_demo_dataset(
        args.dataset,
        samples_per_class=args.samples_per_class,
        random_state=args.random_state,
        num_groups=args.num_groups,
    )
    data, labels, class_names, groups = load_dataset(args.dataset)
    model, metrics = train_model(
        data=data,
        labels=labels,
        random_state=args.random_state,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=args.n_jobs,
        groups=groups,
        split_strategy="group",
        test_size=args.test_size,
    )
    save_outputs(
        args.model_output,
        args.report_output,
        model,
        class_names,
        metrics,
        build_model_metadata(args.dataset, args.report_output),
    )
    print(f"Demo evaluation written to {args.report_output}")


def run_evaluate(args: argparse.Namespace) -> None:
    from tlse.evaluate import evaluate

    evaluate(
        dataset_path=args.dataset,
        model_path=args.model,
        report_path=args.report_output,
        warning=args.warning,
    )


def run_benchmark(args: argparse.Namespace) -> None:
    from tlse.benchmark import benchmark_models

    benchmark_models(
        dataset_path=args.dataset,
        report_path=args.report_output,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tlse",
        description="TLSE utilities for demo training, model evaluation, and classifier benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Generate and train on the synthetic demo dataset.")
    demo.add_argument("--dataset", type=Path, default=Path("demo/sample_data/landmarks_demo.pickle"))
    demo.add_argument("--model-output", type=Path, default=Path("models/demo_model.p"))
    demo.add_argument("--report-output", type=Path, default=Path("reports/demo_metrics.json"))
    demo.add_argument("--samples-per-class", type=int, default=40)
    demo.add_argument("--num-groups", type=int, default=4)
    demo.add_argument("--random-state", type=int, default=42)
    demo.add_argument("--n-iter", type=int, default=2)
    demo.add_argument("--cv", type=int, default=2)
    demo.add_argument("--n-jobs", type=int, default=1)
    demo.add_argument("--test-size", type=float, default=0.25)
    demo.set_defaults(func=run_demo)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model on a processed dataset.")
    evaluate_parser.add_argument("--dataset", type=Path, default=Path("data/processed/landmarks.pickle"))
    evaluate_parser.add_argument("--model", type=Path, default=Path("models/model_depth47.p"))
    evaluate_parser.add_argument("--report-output", type=Path, default=Path("reports/eval_metrics.json"))
    evaluate_parser.add_argument("--warning", default=None)
    evaluate_parser.set_defaults(func=run_evaluate)

    benchmark = subparsers.add_parser("benchmark", help="Compare baseline classifiers on a processed dataset.")
    benchmark.add_argument("--dataset", type=Path, default=Path("data/processed/landmarks.pickle"))
    benchmark.add_argument("--report-output", type=Path, default=Path("reports/model_benchmark.json"))
    benchmark.add_argument("--split-strategy", choices=["random", "group"], default="random")
    benchmark.add_argument("--test-size", type=float, default=0.20)
    benchmark.add_argument("--random-state", type=int, default=42)
    benchmark.set_defaults(func=run_benchmark)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
