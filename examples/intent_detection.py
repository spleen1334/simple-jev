"""Identify what someone is asking for; try a request with two different intents.

The example only suggests a category. It does not execute commands or call tools.
"""

from _display import load_example_model, show_result

from simple_jev import classify


def main() -> None:
    text = "Could you play some relaxing music?"
    choices = ("Play music", "Check weather", "Control lights", "Other")
    model = load_example_model()

    # 'Other' offers an answer for unrelated requests, not guaranteed abstention.
    scores = classify(
        text,
        choices,
        instruction=(
            "Choose the user's main intent. Use Other if none of the tasks fit."
        ),
        model=model,
    )
    show_result("Intent detection", text, scores, model=model)


if __name__ == "__main__":
    main()
