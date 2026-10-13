"""Compare choice order and wording on the same review and instruction.

Tweak the permutations or synonyms; this demonstration is not a bias benchmark.
"""

from simple_jev import classify
from simple_jev.display import load_example_model, show_result


def main() -> None:
    text = "The delivery was late, but the product works beautifully."
    choice_sets = (
        ("Positive", "Neutral", "Negative"),
        ("Negative", "Neutral", "Positive"),
        ("Satisfied", "Mixed or indifferent", "Dissatisfied"),
    )

    titles = ("Original order", "Reversed order", "Different wording")
    model = load_example_model()

    # Reordering changes letter assignments; rewording changes category meaning.
    # Keep the text and instruction fixed to isolate those two experiments.
    for title, choices in zip(titles, choice_sets, strict=True):
        scores = classify(
            text,
            choices,
            instruction="Classify the overall sentiment of this review.",
            model=model,
        )
        show_result(
            f"Choice sensitivity · {title}",
            text,
            scores,
            model=model,
            note=(
                "Only answer order/wording changed. Changes are not guaranteed. "
                "Scores are not calibrated confidence."
            ),
        )


if __name__ == "__main__":
    main()
