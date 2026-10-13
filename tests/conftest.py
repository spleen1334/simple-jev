"""Fake direct-wheel inference; no model files or downloads are needed."""

from string import ascii_uppercase

import numpy as np
import pytest

from simple_jev import classifier as classifier_module


class FakeModel:
    def __init__(self):
        self.label_tokens = {
            label.encode(): [2 * index + 1]
            for index, label in enumerate(ascii_uppercase)
        }
        self.prompt_tokens = [56, 57, 58]
        self.context_size = 512
        self.tokenize_calls = []
        self.events = []
        self.n_tokens = 2
        self.scores = np.full((5, 64), 100.0, dtype=np.float32)
        token_ids = [tokens[0] for tokens in self.label_tokens.values()]
        self.scores[0, token_ids] = np.arange(26, 0, -1)
        self.scores[2, token_ids] = [-3.0, -2.0, -0.2, *range(23)]
        self.scores[-1, token_ids] = np.arange(26, 0, -1)

    def tokenize(self, text, *, add_bos=True, special=False):
        self.tokenize_calls.append((text, {"add_bos": add_bos, "special": special}))
        if text in self.label_tokens:
            return list(self.label_tokens[text])
        return list(self.prompt_tokens)

    def n_ctx(self):
        return self.context_size

    def reset(self):
        self.events.append(("reset",))
        self.n_tokens = 0

    def eval(self, tokens):
        assert self.n_tokens == 0, "Classification must reset previous model state."
        self.events.append(("eval", list(tokens)))
        self.n_tokens = len(tokens)


@pytest.fixture(autouse=True)
def clear_model_cache():
    classifier_module.load_model.cache_clear()
    yield
    classifier_module.load_model.cache_clear()


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def model_loader(monkeypatch, fake_model):
    constructors = []
    downloads = []

    def construct(**kwargs):
        constructors.append(kwargs)
        return fake_model

    def download():
        downloads.append(())
        return "/cache/Qwen3-0.6B-Q8_0.gguf"

    monkeypatch.setattr(classifier_module, "Llama", construct)
    monkeypatch.setattr(classifier_module, "download_model", download)
    return constructors, downloads
