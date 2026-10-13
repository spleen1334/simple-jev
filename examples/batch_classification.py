"""Classify emails sequentially with the same model; tweak or extend the texts.

Each call is an HTTP request. Ollama, not Python, keeps the model warm for 5m.
"""

from simple_jev import classify


def main() -> None:
    texts = (
        "Our meeting is confirmed for Tuesday at 10 a.m.",
        "Huge sale! Buy now and get a second item free!",
        "Send your password to verify your account and avoid suspension.",
    )
    choices = ("Legitimate", "Spam", "Phishing")
    model = "qwen3:0.6b"

    # No batch API or Python model lifecycle is needed for sequential requests.
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
