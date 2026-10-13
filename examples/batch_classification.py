"""Classify emails sequentially with the same model; tweak or extend the texts.

Load once into this Python process and reuse it for every input.
"""

from simple_jev import classify, load_model


def main() -> None:
    texts = (
        "Our meeting is confirmed for Tuesday at 10 a.m.",
        "Huge sale! Buy now and get a second item free!",
        "Send your password to verify your account and avoid suspension.",
    )
    choices = ("Legitimate", "Spam", "Phishing")
    model = load_model()

    # Pass the same local model to each call; classify resets its evaluation state.
    for text in texts:
        scores = classify(
            text,
            choices,
            instruction="Classify this email by its primary purpose.",
            model=model,
        )
        print(f"\n{text}")
        print(f"Prediction: {max(scores, key=scores.get)}")
        for choice, probability in scores.items():
            print(f"  {choice}: {probability:.3f}")


if __name__ == "__main__":
    main()
