"""Identify what someone is asking for; try a request with two different intents.

The example only suggests a category. It does not execute commands or call tools.
"""

from simple_jev import classify


def main() -> None:
    text = "Could you play some relaxing music?"
    choices = ("Play music", "Check weather", "Control lights", "Other")

    # 'Other' offers an answer for unrelated requests, not guaranteed abstention.
    scores = classify(
        text,
        choices,
        instruction=(
            "Choose the user's main intent. Use Other if none of the tasks fit."
        ),
    )
    print(f"Request: {text}")
    print(f"Intent: {max(scores, key=scores.get)}")
    for choice, probability in scores.items():
        print(f"  {choice}: {probability:.3f}")


if __name__ == "__main__":
    main()
