"""Compare choice order and wording on the same review and instruction.

Tweak the permutations or synonyms; this demonstration is not a bias benchmark.
"""

from simple_jev import classify


def main() -> None:
    text = "The delivery was late, but the product works beautifully."
    choice_sets = (
        ("Positive", "Neutral", "Negative"),
        ("Negative", "Neutral", "Positive"),
        ("Satisfied", "Mixed or indifferent", "Dissatisfied"),
    )

    # Reordering changes letter assignments; rewording changes category meaning.
    # Keep the text and instruction fixed to isolate those two experiments.
    for choices in choice_sets:
        scores = classify(
            text,
            choices,
            instruction="Classify the overall sentiment of this review.",
        )
        print(f"\nChoices: {choices}")
        print(f"Prediction: {max(scores, key=scores.get)}")
        for choice, probability in scores.items():
            print(f"  {choice}: {probability:.3f}")
    print("\nOrdering and wording can change scores; changes are not guaranteed.")


if __name__ == "__main__":
    main()
