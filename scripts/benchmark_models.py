from tlse.benchmark import benchmark_models, build_parser


def main() -> None:
    args = build_parser().parse_args()
    benchmark_models(
        dataset_path=args.dataset,
        report_path=args.report_output,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
