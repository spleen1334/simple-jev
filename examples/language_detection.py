"""Identify a message's language; try another sentence or mix two languages.

This is classification, not translation: the output is a score per language.
"""

from simple_jev import classify
from simple_jev.display import load_example_model, show_result


def main() -> None:
    text = "Können wir uns morgen um zehn Uhr treffen?"
    choices = ("English", "German", "Spanish")
    model = load_example_model()

    # A language not in the choices still gets forced into one of these categories.
    scores = classify(
        text,
        choices,
        instruction="Identify the language of this message.",
        model=model,
    )
    show_result("Language detection", text, scores, model=model)


if __name__ == "__main__":
    main()
