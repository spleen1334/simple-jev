"""Classify text by comparing the model's preference for a few possible answers.

Read build_prompt() for the question, classify() for the model request, and
probabilities_from_logprobs() for turning its scores into readable probabilities.
Ollama runs the model; the prompt here is written specifically for Qwen3.
"""

from collections.abc import Sequence
from string import ascii_uppercase

import httpx
import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_MODEL = "qwen3:0.6b"
DEFAULT_SERVER_URL = "http://127.0.0.1:11434"
DEFAULT_INSTRUCTION = "Classify the input. Choose exactly one option."


def build_prompt(
    text: str, choices: Sequence[str], instruction: str = DEFAULT_INSTRUCTION
) -> str:
    """Turn choices into a Qwen multiple-choice question with an answer prefix."""
    if isinstance(choices, str) or not 2 <= len(choices) <= 20:
        raise ValueError("Supply between 2 and 20 choices.")
    if any(not choice.strip() for choice in choices):
        raise ValueError("Choices must not be blank.")
    if len({choice.strip().casefold() for choice in choices}) != len(choices):
        raise ValueError("Choices must be unique.")
    if not text.strip() or not instruction.strip():
        raise ValueError("Input text and instruction must not be blank.")

    # A token is a piece of text from the model's vocabulary, not always a word.
    # Qwen treats A/B/C as single tokens. Using them instead of long choice names
    # lets us compare every answer at the same position: the very next token.
    labels = ascii_uppercase[: len(choices)]
    options = "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
    )
    # ChatML marks the roles. The empty thinking block puts us at the answer,
    # so the next token can be A/B/C rather than a generated reasoning section.
    return f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
Input: {text}

{options}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


def probabilities_from_logprobs(logprobs: ArrayLike) -> NDArray[np.float64]:
    """Turn token log probabilities into probabilities over our choices.

    A log probability is log(p): another way of writing a probability p.
    For example, log(0.5) is about -0.693; exp(-0.693) gets back about 0.5.
    More negative means less likely, not an error. These are natural logarithms
    (base e), not base-2 logarithms measured in bits.

    Softmax means: exponentiate each score, then divide by the total. The result
    is nonnegative and sums to one. We apply it only to our answer tokens, so
    their probabilities become relative to those answers, not all possible text.

    Example: the model gives A/B/C probabilities 0.05/0.10/0.20, with the remaining
    0.65 on other tokens. Among A/B/C alone, divide by 0.35 to get 14%/29%/57%:

    >>> probabilities_from_logprobs(np.log([0.05, 0.10, 0.20])).round(3)
    array([0.143, 0.286, 0.571])

    Why not use raw logits? A logit is the model's unnormalized score, not a
    probability. log P(token) = logit - a constant shared by all tokens. Softmax
    ignores such shared offsets, so Ollama's log probabilities work just as well.
    """
    values = np.asarray(logprobs, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Supply at least two finite log probabilities.")
    # Subtracting the largest score keeps exp() numerically safe. In the example
    # above it produces weights [0.25, 0.5, 1.0], with the same 1:2:4 ratio.
    weights = np.exp(values - values.max())
    # Divide by their total (1.75 in the example). This is the softmax step.
    return weights / weights.sum()


def classify(
    text: str,
    choices: Sequence[str],
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    model: str = DEFAULT_MODEL,
    server_url: str = DEFAULT_SERVER_URL,
    n_ctx: int = 2048,
    timeout: float = 120.0,
) -> dict[str, float]:
    """Return {choice: probability}; pick a winner with max(scores, key=scores.get).

    Start `ollama serve` and pull the model first. Use a trusted local server and
    model: custom URLs or cloud models can send your input outside your machine.
    """
    # 1. Present the task as a multiple-choice question.
    prompt = build_prompt(text, choices, instruction)
    if n_ctx < 1:
        raise ValueError("n_ctx must be positive.")

    # 2. Ask for the first token's scores. raw=True avoids double-formatting the
    # prompt. Disable sampling filters so they don't discard candidate labels.
    response = httpx.post(
        f"{server_url.rstrip('/')}/api/generate",
        timeout=timeout,
        trust_env=False,
        json={
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "truncate": False,
            "shift": False,
            "keep_alive": "5m",
            "options": {
                "num_predict": 1,
                "num_ctx": n_ctx,
                "temperature": 1,
                "top_k": 0,
                "top_p": 1,
                "min_p": 0,
                "repeat_penalty": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
        },
    )
    response.raise_for_status()
    data = response.json()
    if data.get("prompt_eval_count", 0) >= n_ctx:
        raise ValueError("Prompt fills the context. Shorten it or increase n_ctx.")
    if not data.get("logprobs"):
        raise ValueError("No token log probabilities returned. Update Ollama.")
    # Ollama reports log probabilities for the next text token, not probabilities
    # that our categories are correct. Keep the alternatives at position [0];
    # we don't need the random token that happened to be generated.
    token_scores = {
        token["token"]: token["logprob"]
        for token in data["logprobs"][0]["top_logprobs"]
    }
    labels = ascii_uppercase[: len(choices)]
    missing = [label for label in labels if label not in token_scores]
    if missing:
        raise ValueError(f"Labels {missing} are missing from Ollama's top 20 tokens.")

    # 3. Normalize only our labels. "Normalize" means make their total equal 1.
    # A result like {"Spam": 0.9, "Legitimate": 0.1} expresses relative preference,
    # NOT a tested 90% chance of being correct. Even unsuitable choices get a winner.
    probabilities = probabilities_from_logprobs(
        [token_scores[label] for label in labels]
    )
    return {
        choice: float(probability)
        for choice, probability in zip(choices, probabilities, strict=True)
    }


def download_model(
    *,
    model: str = DEFAULT_MODEL,
    server_url: str = DEFAULT_SERVER_URL,
    timeout: float = 600.0,
) -> str:
    """Pull a model into Ollama's cache; inference itself never downloads a model."""
    response = httpx.post(
        f"{server_url.rstrip('/')}/api/pull",
        json={"model": model, "stream": False},
        timeout=timeout,
        trust_env=False,
    )
    response.raise_for_status()
    return model
