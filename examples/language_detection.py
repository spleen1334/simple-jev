"""Identify a message's language; try another sentence or mix two languages.

This is classification, not translation: the output is a score per language.
"""

from simple_jev import classify


def main() -> None:
    text = "Können wir uns morgen um zehn Uhr treffen?"
    choices = ("English", "German", "Spanish")

    # A language not in the choices still gets forced into one of these categories.
    scores = classify(
        text, choices, instruction="Identify the language of this message."
    )
    print(f"Message: {text}")
    print(f"Language: {max(scores, key=scores.get)}")
    for choice, probability in scores.items():
        print(f"  {choice}: {probability:.3f}")


if __name__ == "__main__":
    main()
