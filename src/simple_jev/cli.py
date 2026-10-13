"""Download a GGUF model or try the local classifier from the terminal."""

import argparse
import json

from .classifier import DEFAULT_INSTRUCTION, classify, download_model


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Cache the default Qwen GGUF model")
    classification = commands.add_parser(
        "classify", help="Score choices for an input", allow_abbrev=False
    )
    classification.add_argument("--text", required=True)
    classification.add_argument("--choices", nargs="+", required=True)
    classification.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    classification.add_argument("--model-path", help="Local Qwen-compatible GGUF")
    classification.add_argument("--n-ctx", type=int, default=512)
    classification.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(download_model())
            return 0
        scores = classify(
            args.text,
            args.choices,
            instruction=args.instruction,
            model_path=args.model_path,
            n_ctx=args.n_ctx,
        )
        if args.json:
            print(json.dumps(scores, indent=2))
        else:
            print(f"Prediction: {max(scores, key=scores.get)}")
            for choice, probability in scores.items():
                print(f"  {choice}: {probability:.4f}")
            print(
                "Probabilities are normalized over choices, not calibrated confidence."
            )
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
