"""Classify an email's purpose; tweak the message, choices, or instruction.

Compare benign promotions with credential requests, not just obvious examples.
"""

from simple_jev import classify


def main() -> None:
    email = (
        "Your account will be suspended today. Verify your password immediately "
        "at https://account-verification.example to keep access."
    )
    choices = ("Legitimate", "Spam", "Phishing")

    # The score is relative to these choices, not a calibrated phishing risk.
    scores = classify(
        email,
        choices,
        instruction="Classify this email by its primary purpose.",
    )
    print(f"Prediction: {max(scores, key=scores.get)}")
    for choice, probability in scores.items():
        print(f"  {choice}: {probability:.3f}")


if __name__ == "__main__":
    main()
