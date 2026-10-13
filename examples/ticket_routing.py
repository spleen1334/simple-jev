"""Suggest a support team; tweak the ticket and the team's responsibilities.

Try tickets that span departments. A routing suggestion is not a resolution.
"""

from simple_jev import classify
from simple_jev.helpers import load_example_model, show_result


def main() -> None:
    ticket = "I was charged twice for my subscription. Can you refund one charge?"
    choices = ("Billing", "Technical support", "Account access", "Sales")
    model = load_example_model()

    # Descriptions ground short team names in this organization's routing policy.
    scores = classify(
        ticket,
        choices,
        instruction=(
            "Choose the support team best suited to handle this ticket. "
            "Billing handles charges, invoices, and refunds. "
            "Technical support handles bugs and product failures. "
            "Account access handles sign-in, passwords, and locked accounts. "
            "Sales handles pricing questions and new purchases."
        ),
        model=model,
    )
    show_result("Support ticket routing", ticket, scores, model=model)


if __name__ == "__main__":
    main()
