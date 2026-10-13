"""Shared example output, using built-ins rather than an extra dependency.

Use `make -s examples` to hide Make's command echoes. Color is automatic in a
terminal; set NO_COLOR=1 for plain text or FORCE_COLOR=1 to force ANSI colors.
"""

import os
import shutil
import sys
from string import ascii_uppercase
from textwrap import wrap

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from llama_cpp import Llama, llama_cpp
from llama_cpp._logger import llama_log_callback as _default_model_log

from simple_jev import download_model, load_model
from simple_jev.classifier import DEFAULT_FILENAME, DEFAULT_REPO


def _color(text: str, code: str) -> str:
    enabled = "NO_COLOR" not in os.environ and (
        os.environ.get("FORCE_COLOR") == "1"
        or (sys.stdout.isatty() and os.environ.get("TERM") != "dumb")
    )
    return f"\033[{code}m{text}\033[0m" if enabled else text


@llama_cpp.llama_log_callback
def _model_log(_level, message, _user_data):
    text = message.decode(errors="replace")
    # A deliberately small context is fine for these short examples. Delegate
    # other messages to the pinned library's normal verbosity/error handling.
    if "n_ctx_seq" in text and "full capacity of the model" in text:
        return
    _default_model_log(_level, message, _user_data)


def load_example_model():
    """Prefer the downloaded file, without a network metadata check on every run."""
    # Keep the callback at module scope: C holds a pointer to this Python object.
    llama_cpp.llama_log_set(_model_log, None)
    try:
        path = hf_hub_download(
            repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME, local_files_only=True
        )
    except LocalEntryNotFoundError:
        # A fresh setup still shows real download progress and download errors.
        path = download_model()
    return load_model(model_path=path)


def show_result(
    title: str,
    input_text: str,
    scores: dict[str, float],
    *,
    model: Llama | None = None,
    note: str = "Relative scores across these choices, not calibrated confidence.",
    details: dict[str, dict[str, float]] | None = None,
) -> None:
    """Show the input, probabilities, and raw scores for the latest classification."""
    if details is None and model is not None:
        # Read the model immediately after classify(), before evaluating another input.
        labels = ascii_uppercase[: len(scores)]
        token_ids = [
            model.tokenize(label.encode(), add_bos=False)[0] for label in labels
        ]
        logits = np.asarray(model.scores[model.n_tokens - 1, token_ids], dtype=float)
        # Log-softmax stays finite even when a tiny probability rounds down to zero.
        shifted = logits - logits.max()
        logprobs = shifted - np.logaddexp.reduce(shifted)
        details = {
            name: dict(zip(scores, (float(value) for value in values), strict=True))
            for name, values in (("Logits", logits), ("Log probabilities", logprobs))
        }

    width = max(40, min(88, shutil.get_terminal_size().columns))
    winner = max(scores, key=scores.get)
    label_width = min(28, max(12, max(len(choice) for choice in scores)))
    bar_width = max(8, min(24, width - label_width - 20))
    separator = _color("━" * width, "36")

    print(f"\n{separator}")
    print(_color(title, "1;36"))
    print()
    print(_color("INPUT", "1;34"))
    for paragraph in input_text.splitlines():
        for line in wrap(paragraph, width=width - 4) or [""]:
            print(f"  {line}")

    print()
    print(_color("OUTPUT", "1;34"))
    print(f"  Best match: {_color(winner, '1;32')}")
    print()
    print(_color(f"  {'Choice':<{label_width}} {'Share':>7}   Distribution", "2"))
    for choice, probability in scores.items():
        filled = round(probability * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        share = "<0.1%" if 0 < probability < 0.001 else f"{probability:.1%}"
        lines = wrap(choice, width=label_width)
        row = f"  {lines[0]:<{label_width}} {share:>7}   {bar}"
        print(_color(row, "1;32" if choice == winner else "36"))
        for line in lines[1:]:
            print(f"  {line}")

    if details:
        print()
        print(_color("MODEL SCORES · for learning", "1;34"))
        for name, values in details.items():
            text = f"{name}: " + " · ".join(
                f"{choice} {value:.3f}" for choice, value in values.items()
            )
            for line in wrap(text, width=width - 4):
                print(f"  {line}")

    print()
    for line in wrap(note, width=width - 4):
        print(_color(f"  {line}", "2"))
    print(separator)
    print("\n")
