"""The article's original idea, annotated and adapted to the shared project setup."""

import numpy

from simple_jev.display import load_example_model, show_result

# Use the same local model loader as the other examples.
# logits_all=True retains score rows so we can inspect the final prompt position.
# Other GGUFs need compatible prompt formatting and single-token letter labels.

model = load_example_model()

# A/B/C are assumed to each tokenize to one token with this tokenizer.
# Long choice words may need several tokens; scoring their first token is not
# the likelihood of the whole word. Letters let us compare one next-token score.
labels = ["A", "B", "C"]
choices = ["Legitimate", "Spam", "Phishing"]
email = "Payroll asks for your password on a non-company sign-in page."
options = "\n".join(
    f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
)

# Qwen-specific chat markers and an empty thinking block put the next token at
# the answer position. special=True below recognizes those markers as tokens;
# this raw prompt is not a universal template for arbitrary GGUF models.
prompt = f"""<|im_start|>system
Choose one option.<|im_end|>
<|im_start|>user
Email: {email}\n\n{options}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n"""
model.eval(tokens=model.tokenize(text=prompt.encode(), add_bos=False, special=True))

# A logit is a raw score for a possible next token: higher means more preferred.
# It can be any real number and is NOT yet a probability or a number of bits.
# The final prompt row predicts the next token; earlier rows predict tokens at
# earlier positions. Select only the letter logits, not the entire vocabulary.
logits = model.scores[model.n_tokens - 1]
token_ids = [model.tokenize(text=label.encode(), add_bos=False)[0] for label in labels]
choice_logits = numpy.asarray([logits[token_id] for token_id in token_ids])

# Softmax turns scores into probabilities: exp(each score) / sum(exp(scores)).
# Example: logits [1, 2, 3] become roughly [0.090, 0.245, 0.665]. They sum to 1.
# Here we calculate in log space to avoid overflowing exp() for large logits.
# logaddexp.reduce computes log(sum(exp(logits))) safely; subtracting that total
# gives log probabilities. Natural log(0.5) is about -0.693: negative is normal.
logprobs = choice_logits - numpy.logaddexp.reduce(choice_logits)
# exp() undoes the natural log, giving ordinary probabilities between 0 and 1.
# These are relative to A/B/C only, not calibrated chances of being correct.
probabilities = numpy.exp(logprobs)

show_result(
    "Original article · local model",
    email,
    dict(zip(choices, (float(value) for value in probabilities), strict=True)),
    details={
        "Logits": dict(
            zip(choices, (float(value) for value in choice_logits), strict=True)
        ),
        "Log probabilities": dict(
            zip(choices, (float(value) for value in logprobs), strict=True)
        ),
    },
)

# Logits: {'Legitimate': 26.254, 'Spam': 27.262, 'Phishing': 29.614}
# Log probabilities: {'Legitimate': -3.482, 'Spam': -2.474, 'Phishing': -0.122}
# Probabilities: {'Legitimate': 0.031, 'Spam': 0.084, 'Phishing': 0.885}
