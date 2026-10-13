# simple-jev

A small learning project that asks a language model to choose between categories.
Give it some text and possible answers—such as **Legitimate**, **Spam**, and
**Phishing**—and it returns a score for each answer. No training is required.

It is inspired by the parody article
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

Instead of asking the model to write an explanation, we inspect how strongly it
prefers each answer letter. We turn those preferences into numbers that add up
to one and map the letters back to the category names.

An illustrative result might look like this (actual values vary):

```text
Legitimate: 0.03
Spam:       0.08
Phishing:   0.89
```

**Read 0.89 as a relative preference among these choices, not an 89% chance of
being correct.** The model can be wrong, and changing the choices can change the
result. It must favor something even when none of the choices fit well.

### A few words you'll encounter in the code

- **Token:** a piece of text the model processes, such as a letter or part of a
  word. We use letters so each answer is represented by one token.
- **Logit:** a raw model score for a possible next token. Higher means more
  preferred, but the score is not a probability.
- **Log probability (`logprob`):** a probability written using a logarithm.
  It's a numerical representation, not a different kind of confidence.
- **Softmax:** a calculation that converts scores into proportions adding up to
  one. It preserves their ordering: the highest score gets the largest share.

You don't need the formulas to run the examples. The worked numerical example
and explanations of logarithms, `exp()`, and softmax live beside the calculation
in `src/simple_jev/classifier.py`.

## Run your first example

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and Ollama.
The project pins Python 3.12; Ollama runs the model separately from Python.

**1. Install Ollama on macOS:**

```sh
brew install ollama
```

**2. Start it in a separate terminal and leave it running:**

```sh
ollama serve
```

**3. In the project terminal, install dependencies, download the model, and try it:**

```sh
uv sync --extra dev
uv run simple-jev download
uv run python examples/email_classification.py
```

The default model is `qwen3:0.6b`, a small Qwen language model. Downloading it
requires internet access; subsequent inference runs locally with the default
settings. Ollama keeps the model loaded between requests for five minutes.
The normal setup does not install or compile Python model bindings.

## Try your own text

From Python:

```python
from simple_jev import classify

scores = classify("I was charged twice.", ["Billing", "Technical support"])
print(scores)
print("Best match:", max(scores, key=scores.get))
```

Or from the terminal:

```sh
uv run simple-jev classify --text 'I was charged twice.' --choices Billing 'Technical support'
```

Supply 2–20 distinct, nonblank choices. Use `instruction="..."` in Python or
`--instruction '...'` in the CLI to explain the task. Add `--json` to print the
probability dictionary. Run `uv run simple-jev classify --help` for other options.

## Explore the examples

Run each with `uv run python examples/<name>.py`:

| Example | What to explore |
| --- | --- |
| `email_classification.py` | Change the email and compare the category scores. |
| `sentiment.py` | Try a review with mixed positive and negative opinions. |
| `ticket_routing.py` | Explain what each support team handles. |
| `batch_classification.py` | Classify several inputs with a simple loop. |
| `choice_sensitivity.py` | Keep the text fixed; change answer order or wording. |
| `language_detection.py` | Identify a language; try a mixed-language message. |
| `intent_detection.py` | Identify a requested task; try unrelated or multi-intent requests. |

Try adding an “Other” category or comparing predictions with answers you labeled
yourself. A few successful examples are not evidence of general accuracy.

## Read the implementation

Start with `src/simple_jev/classifier.py`:

1. **`build_prompt()`** writes the multiple-choice question in the model's format.
2. **`classify()`** asks Ollama for the next-token scores and selects our letters.
3. **`probabilities_from_logprobs()`** converts those scores into the returned
   proportions. Its docstring walks through a small numerical example.

The inline comments explain why we use letters, how the prompt works, and what
the scores mean. `download_model()` is just a separate download convenience.

The original article implementation is also preserved and annotated at
`examples/original_25_lines.py`. It uses Python bindings rather than Ollama.
Running it with `uv run --script examples/original_25_lines.py` is optional:
it may compile `llama-cpp-python` and downloads a roughly 639 MB GGUF model.
Its model quantization differs from Ollama's default, so scores need not match.

## Things to keep in mind

- **Missing scores:** Ollama returns only its top 20 token alternatives. If one
  of our letters is missing, we stop rather than invent its probability.
- **Long inputs:** the question and answer must fit the model's context window
  (its token budget). Shorten the text or increase `n_ctx` / `--n-ctx` if needed;
  the code disables automatic truncation and context shifting.
- **Model compatibility:** the prompt is written for Qwen3. Swapping in another
  model may require changing the prompt and checking how letters become tokens.
- **Privacy:** defaults use a local model and server. Custom remote URLs or cloud
  models can send text off-device; this experiment does not prevent that.
- **Reliability:** wording, answer order, and instructions inside the input can
  influence results. Don't use these scores alone for security or other
  high-impact decisions.

## Development shortcuts

```sh
make test     # Tests use fake HTTP responses; no model needed.
make lint     # Lint and formatting checks.
make demo     # A real-model example; Ollama must be running.
```

Also available: `make setup`, `make serve`, `make download-model`, `make examples`,
and `make format`. Keep `uv.lock` in version control for reproducible dependencies.
