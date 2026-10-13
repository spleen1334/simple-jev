"""Download an Ollama model or try the classifier from the terminal."""

import argparse
import json

import httpx

from .classifier import (
    DEFAULT_INSTRUCTION,
    DEFAULT_MODEL,
    DEFAULT_SERVER_URL,
    classify,
    download_model,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Pull a model through Ollama")
    classification = commands.add_parser("classify", help="Score choices for an input")
    for command in (download, classification):
        command.add_argument("--model", default=DEFAULT_MODEL)
        command.add_argument("--server-url", default=DEFAULT_SERVER_URL)
        command.add_argument(
            "--timeout", type=float, default=600.0 if command is download else 120.0
        )
    classification.add_argument("--text", required=True)
    classification.add_argument("--choices", nargs="+", required=True)
    classification.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    classification.add_argument("--n-ctx", type=int, default=2048)
    classification.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    settings = {
        "model": args.model,
        "server_url": args.server_url,
        "timeout": args.timeout,
    }
    try:
        if args.command == "download":
            print(download_model(**settings))
            return 0
        scores = classify(
            args.text,
            args.choices,
            instruction=args.instruction,
            n_ctx=args.n_ctx,
            **settings,
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
    except (ValueError, httpx.HTTPError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
