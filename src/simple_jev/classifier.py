"""Multiple-choice classification: build a prompt, read logits, apply softmax.

The Python wheel contains the compiled runtime; no server or compiler is needed.
The prompt is written for Qwen3, not every model supported by llama.cpp.
"""

from collections.abc import Sequence
from functools import lru_cache
from string import ascii_uppercase

import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from numpy.typing import ArrayLike, NDArray

DEFAULT_REPO = "Qwen/Qwen3-0.6B-GGUF"
DEFAULT_FILENAME = "Qwen3-0.6B-Q8_0.gguf"
DEFAULT_INSTRUCTION = "Classify the input. Choose exactly one option."


def download_model() -> str:
    """Download the GGUF once; later calls reuse Hugging Face's model cache."""
    return hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME)


@lru_cache(maxsize=1)
def load_model(*, model_path: str | None = None, n_ctx: int = 512) -> Llama:
    """Keep one model loaded per Python process, including across classify() calls.

    Pass a local model_path to skip Hugging Face entirely. Otherwise the first
    load downloads the default model if needed. n_ctx is the prompt token budget.
    """
    if n_ctx < 1:
        raise ValueError("n_ctx must be positive.")
    return Llama(
        model_path=model_path or download_model(),
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        logits_all=True,
        verbose=False,
    )


def build_prompt(
    text: str, choices: Sequence[str], instruction: str = DEFAULT_INSTRUCTION
) -> str:
    """Turn choices into a Qwen multiple-choice question with an answer prefix."""
    if isinstance(choices, str) or not 2 <= len(choices) <= 26:
        raise ValueError("Supply between 2 and 26 choices.")
    if any(not choice.strip() for choice in choices):
        raise ValueError("Choices must not be blank.")
    if len({choice.strip().casefold() for choice in choices}) != len(choices):
        raise ValueError("Choices must be unique.")
    if not text.strip() or not instruction.strip():
        raise ValueError("Input text and instruction must not be blank.")

    # Tokens are pieces of text, not always whole words. Letters let us compare
    # each answer at the same position; full category names can be many tokens.
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


def softmax(scores: ArrayLike) -> NDArray[np.float64]:
    """Convert raw scores into proportions that sum to one.

    A logit is an unnormalized model score: higher means more preferred, but it
    is not a probability or a number of bits. Softmax exponentiates each score
    and divides by the total. The highest score gets the largest share:

    >>> softmax([1, 2, 3]).round(3)
    array([0.09 , 0.245, 0.665])

    A log probability is log(p), e.g. log(0.5) is about -0.693. Negative is normal;
    exp() undoes the log. These are natural logs (base e), not base-2 logs (bits).
    Softmax also accepts log probabilities: they differ from logits by a shared
    constant, which cancels when dividing by the total. Here we use logits directly.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Supply at least two finite scores.")
    # Subtract max to avoid overflow. For [1, 2, 3], weights are [0.135, 0.368, 1].
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def classify(
    text: str,
    choices: Sequence[str],
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    model: Llama | None = None,
    model_path: str | None = None,
    n_ctx: int = 512,
) -> dict[str, float]:
    """Return {choice: probability}; pick a winner with max(scores, key=scores.get).

    These scores express relative preference, not calibrated chances of being
    correct. Even unsuitable choices get a winner. No text is sent to a server.
    """
    prompt = build_prompt(text, choices, instruction)
    if model is None:
        model = load_model(model_path=model_path, n_ctx=n_ctx)

    labels = ascii_uppercase[: len(choices)]
    token_ids = []
    for label in labels:
        tokens = model.tokenize(label.encode(), add_bos=False)
        if len(tokens) != 1:
            raise ValueError(f"Label {label!r} must be exactly one token.")
        token_ids.append(tokens[0])

    tokens = model.tokenize(prompt.encode(), add_bos=False, special=True)
    if len(tokens) > model.n_ctx():
        raise ValueError("Prompt exceeds context. Shorten it or increase n_ctx.")
    # eval() appends to the model's state, so reset it for each independent input.
    model.reset()
    model.eval(tokens)

    # The last prompt row scores the NEXT token. Select our letter scores from
    # the full vocabulary; we need neither generated text nor a top-token shortlist.
    logits = model.scores[model.n_tokens - 1, token_ids]
    probabilities = softmax(logits)
    return {
        choice: float(probability)
        for choice, probability in zip(choices, probabilities, strict=True)
    }
