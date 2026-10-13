"""Classify an email's purpose; tweak the message, choices, or instruction.

Compare benign promotions with credential requests, not just obvious examples.
"""

from simple_jev import classify
from simple_jev.display import load_example_model, show_result


def main() -> None:
    email = (
        "Your account will be suspended today. Verify your password immediately "
        "at https://account-verification.example to keep access."
    )
    choices = ("Legitimate", "Spam", "Phishing")
    model = load_example_model()

    # The score is relative to these choices, not a calibrated phishing risk.
    scores = classify(
        email,
        choices,
        instruction="Classify this email by its primary purpose.",
        model=model,
    )
    show_result("Email classification", email, scores, model=model)


if __name__ == "__main__":
    main()
