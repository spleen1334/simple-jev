"""Classify emails sequentially with the same model; tweak or extend the texts.

Load once into this Python process and reuse it for every input.
"""

from simple_jev import classify
from simple_jev.helpers import load_example_model, show_result


def main() -> None:
    texts = (
        "Our meeting is confirmed for Tuesday at 10 a.m.",
        "Huge sale! Buy now and get a second item free!",
        "Send your password to verify your account and avoid suspension.",
    )
    choices = ("Legitimate", "Spam", "Phishing")
    model = load_example_model()

    # Pass the same local model to each call; classify resets its evaluation state.
    for index, text in enumerate(texts, start=1):
        scores = classify(
            text,
            choices,
            instruction="Classify this email by its primary purpose.",
            model=model,
        )
        show_result(f"Email batch · {index}/{len(texts)}", text, scores, model=model)


if __name__ == "__main__":
    main()
