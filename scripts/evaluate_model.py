from tlse.evaluate import build_parser, evaluate


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args.dataset, args.model, args.report_output, args.warning)


if __name__ == "__main__":
    main()
