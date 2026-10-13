"""Score a mixed review; tweak the text or add a 'Mixed' sentiment choice.

Compare how the available categories change the interpretation of the same text.
"""

from simple_jev import classify
from simple_jev.display import load_example_model, show_result


def main() -> None:
    text = "The delivery was late, but the product works beautifully."
    choices = ("Positive", "Neutral", "Negative")
    model = load_example_model()

    # A forced choice can hide nuance: 'Neutral' need not mean mixed sentiment.
    scores = classify(
        text,
        choices,
        instruction="Classify the overall sentiment of this review.",
        model=model,
    )
    show_result("Sentiment", text, scores, model=model)


if __name__ == "__main__":
    main()
