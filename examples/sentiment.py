"""Score a mixed review; tweak the text or add a 'Mixed' sentiment choice.

Compare how the available categories change the interpretation of the same text.
"""

from simple_jev import classify


def main() -> None:
    text = "The delivery was late, but the product works beautifully."
    choices = ("Positive", "Neutral", "Negative")

    # A forced choice can hide nuance: 'Neutral' need not mean mixed sentiment.
    scores = classify(
        text,
        choices,
        instruction="Classify the overall sentiment of this review.",
    )
    print(f"Review: {text}")
    print(f"Prediction: {max(scores, key=scores.get)}")
    for choice, probability in scores.items():
        print(f"  {choice}: {probability:.3f}")


if __name__ == "__main__":
    main()
