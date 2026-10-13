.PHONY: setup serve download-model demo examples test lint format check

setup:
	uv sync --extra dev

serve:
	ollama serve

download-model:
	uv run simple-jev download

demo:
	uv run simple-jev classify --text 'Payroll asks for your password on a non-company sign-in page.' --choices Legitimate Spam Phishing

examples:
	uv run python examples/email_classification.py
	uv run python examples/sentiment.py
	uv run python examples/ticket_routing.py
	uv run python examples/batch_classification.py
	uv run python examples/choice_sensitivity.py
	uv run python examples/language_detection.py
	uv run python examples/intent_detection.py

test:
	uv run --extra dev pytest

lint:
	uv run --extra dev ruff check .
	uv run --extra dev ruff format --check .

format:
	uv run --extra dev ruff format .

check: lint test
