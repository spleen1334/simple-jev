# simple-jev

A small learning project for classifying text without training. Give it choices
such as **Legitimate**, **Spam**, and **Phishing**; it returns a score for each.

Inspired by the parody article
[“Jev in 25 lines”](https://www.nobodywho.ai/posts/jev-in-25-lines/), not a
reproduction of Jev's training or calibrated decision-making.

## The basic idea

Think of this as a multiple-choice question:

```text
Email: Please send your password to verify your account.

A. Legitimate
B. Spam
C. Phishing
```

Instead of generating an explanation, we inspect the model's preference for each
answer letter. Those scores become proportions adding up to one, such as
`Legitimate: 0.03, Spam: 0.08, Phishing: 0.89` (illustrative; actual values vary).

**0.89 is a relative preference among these choices, not an 89% chance of being
correct.** The model must favor something even when none of the choices fit.

A few words you'll encounter in the code:

- **Token:** a piece of text the model processes. Each answer letter uses one.
- **Logit:** a raw preference score, not yet a probability.
- **Softmax:** converts those scores into proportions adding up to one.
- **Log probability:** a probability expressed using a logarithm, not a different
  kind of confidence. The math and numerical examples live beside the code.

## Run your first example

You need [uv](https://docs.astral.sh/uv/getting-started/installation/).
The default tested setup is **macOS on Apple Silicon (ARM64), with Python 3.12**.
The project pins Python to `>=3.12,<3.13`; uv can install it if needed.


```sh
uv sync --extra dev
uv run simple-jev download
uv run python examples/email_classification.py
```

No separate server or native compilation is needed. uv installs the pinned
`llama-cpp-python==0.3.19` **precompiled wheel** from the explicit `metal` index:
`https://abetlen.github.io/llama-cpp-python/whl/metal`. The model runs inside
Python with Metal GPU offload. `no-build-package = ["llama-cpp-python"]` prevents
source builds: platforms without a compatible wheel fail fast instead.

The default model is `Qwen/Qwen3-0.6B-GGUF`, file `Qwen3-0.6B-Q8_0.gguf`.
A fresh setup downloads roughly **639 MB**, then reuses the Hugging Face cache.
The download command takes no options and prints the cache path. It is optional:
the first model load downloads the file automatically if it isn't cached.

## Try your own text

```python
from simple_jev import classify

scores = classify("I was charged twice.", ["Billing", "Technical support"])
print(scores)
print("Best match:", max(scores, key=scores.get))
```


```sh
uv run simple-jev classify --text 'I was charged twice.' --choices Billing 'Technical support'
```

Supply **2–26 distinct, nonblank choices**. Letters A–Z represent the choices;
each is checked to be one token. All selected letters are scored—there is no
top-20 cutoff.

Use `instruction="..."` or CLI `--instruction '...'` to explain the task.
The default is `"Classify the input. Choose exactly one option."`. CLI options
include `--json` for the score mapping, `--model-path`, and `--n-ctx` (default
`512`). See `uv run simple-jev classify --help`.

### Reuse a model and work offline

`classify()` automatically reuses a cached model within the Python process.
To load once explicitly, as in the batch example:

```python
from simple_jev import classify, load_model

model = load_model()
scores = classify("I was charged twice.", ["Billing", "Technical support"], model=model)
```

`download_model()` returns the default file's cache path without loading it.
Both `load_model()` and `classify()` accept `model_path` and `n_ctx`. An existing
local file path bypasses Hugging Face; alternatively, set `HF_HUB_OFFLINE=1`
after caching the default file. Normal cache lookup may contact Hugging Face for
metadata, but inference is local and prompts are not sent off-device.

## Explore the examples

Run each with `uv run python examples/<name>.py`:

| Example | What to explore |
| --- | --- |
| `email_classification.py` | Change the email and compare category scores. |
| `sentiment.py` | Try a review with mixed opinions. |
| `ticket_routing.py` | Explain what each support team handles. |
| `batch_classification.py` | Load one model and classify several inputs. |
| `choice_sensitivity.py` | Change answer order or wording, keeping text fixed. |
| `language_detection.py` | Try another language or a mixed-language message. |
| `intent_detection.py` | Try unrelated or multi-intent requests. |
| `original_25_lines.py` | Follow the original idea's logits and softmax math. |

## Read the implementation

Start with `src/simple_jev/classifier.py`: build the Qwen3 prompt, reset and
evaluate the model, select the answer-letter logits, then apply softmax.
Comments and docstrings explain the numerical details beside the calculation.
`examples/original_25_lines.py` preserves the core logits math and teaching
comments, adapted to the shared `load_model()` setup—not a byte-for-byte copy.
Run it like the other examples; no separate dependencies or compilation needed.

## Things to keep in mind

- **Context:** the entire prompt, including instructions, choices, and chat
  markers, must fit the default **512-token** window. Exactly filling it is
  allowed because no answer token is generated. Shorten inputs or increase
  `n_ctx` / `--n-ctx`; larger contexts and their logits buffers need more memory.
- **Compatibility:** the prompt is written for Qwen3. Other GGUFs may need
  different formatting and checks that answer letters are single tokens.
- **Privacy:** inference stays local; uncached downloads need internet access.
- **Learning, not guarantees:** wording, answer order, and instructions inside
  the input can affect results. Try an “Other” category and compare against
  answers you labeled yourself. Don't rely on these scores alone for security
  or other high-impact decisions.

## Development shortcuts

```sh
make test     # Model doubles; no real model needed.
make lint     # Lint and formatting checks.
make demo     # Real model; downloads it if not cached.
```

Also: `make setup`, `make download-model`, `make examples`, `make format`, and
`make check` (lint plus tests). Keep `uv.lock` for reproducible dependencies.
