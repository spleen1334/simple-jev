from string import ascii_uppercase

import numpy as np
import pytest

from simple_jev import classifier as classifier_module
from simple_jev import classify, download_model
from simple_jev.classifier import (
    DEFAULT_FILENAME,
    DEFAULT_INSTRUCTION,
    DEFAULT_REPO,
    build_prompt,
    load_model,
    softmax,
)


def test_softmax_is_float64_stable_normalized_and_offset_invariant():
    values = np.array([-3.0, -2.0, -0.2], dtype=np.float32)
    probabilities = softmax(values)

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.dtype == np.float64
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.argmax(probabilities) == np.argmax(values)
    np.testing.assert_allclose(softmax(values.astype(np.float64) + 1e6), probabilities)
    np.testing.assert_allclose(softmax(values.tolist()), probabilities)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        np.testing.assert_allclose(softmax([10000, -10000]), [1.0, 0.0])
        np.testing.assert_allclose(softmax([1e300, 1e300]), [0.5, 0.5])


def test_softmax_rejects_invalid_shape_size_and_nonfinite_values():
    for values in (0, [], [1], [[1, 2]], [0, np.nan], [0, np.inf], [-np.inf, 0]):
        with pytest.raises(ValueError):
            softmax(values)


def test_softmax_normalizes_only_the_supplied_choice_scores():
    np.testing.assert_allclose(
        softmax(np.log([0.05, 0.10, 0.20])), [1 / 7, 2 / 7, 4 / 7]
    )


def test_build_prompt_has_exact_qwen_structure_and_custom_instruction():
    assert DEFAULT_INSTRUCTION == "Classify the input. Choose exactly one option."

    def expected(instruction):
        return (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            "<|im_start|>user\nInput: café\n\n"
            "A. first\nB. second<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    assert build_prompt("café", ["first", "second"]) == expected(DEFAULT_INSTRUCTION)
    assert build_prompt("café", ["first", "second"], "Pick one.") == expected(
        "Pick one."
    )


def test_build_prompt_accepts_two_to_twenty_six_and_preserves_choices():
    for count in (2, 26):
        choices = [f" Choice {index} " for index in range(count)]
        prompt = build_prompt("input", choices)
        for label, choice in zip(ascii_uppercase[:count], choices, strict=True):
            assert f"{label}. {choice}" in prompt


def test_invalid_choices_are_rejected_before_model_loading(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid choices must not load or download a model.")

    monkeypatch.setattr(classifier_module, "load_model", unexpected)
    monkeypatch.setattr(classifier_module, "download_model", unexpected)
    for choices in (
        [],
        ["only"],
        [str(index) for index in range(27)],
        ["first", " \t"],
        [" Straße ", "STRASSE"],
        "AB",
    ):
        with pytest.raises(ValueError):
            build_prompt("input", choices)
        with pytest.raises(ValueError):
            classify("input", choices)


def test_blank_text_and_instruction_are_rejected_before_loading(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid prompts must not load or download a model.")

    monkeypatch.setattr(classifier_module, "load_model", unexpected)
    monkeypatch.setattr(classifier_module, "download_model", unexpected)
    for text, instruction in ((" \t", DEFAULT_INSTRUCTION), ("input", " \n")):
        with pytest.raises(ValueError):
            build_prompt(text, ["first", "second"], instruction)
        with pytest.raises(ValueError):
            classify(text, ["first", "second"], instruction=instruction)


def test_download_uses_default_hugging_face_repo_filename_and_standard_cache(
    monkeypatch,
):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return "/cache/snapshots/revision/Qwen3-0.6B-Q8_0.gguf"

    monkeypatch.setattr(classifier_module, "hf_hub_download", download)
    assert DEFAULT_REPO == "Qwen/Qwen3-0.6B-GGUF"
    assert DEFAULT_FILENAME == "Qwen3-0.6B-Q8_0.gguf"
    assert download_model() == "/cache/snapshots/revision/Qwen3-0.6B-Q8_0.gguf"
    assert calls == [{"repo_id": DEFAULT_REPO, "filename": DEFAULT_FILENAME}]


def test_load_model_downloads_once_and_caches_metal_constructor(
    model_loader, fake_model
):
    constructors, downloads = model_loader
    assert load_model() is fake_model
    assert load_model() is fake_model
    assert downloads == [()]
    assert constructors == [
        {
            "model_path": "/cache/Qwen3-0.6B-Q8_0.gguf",
            "n_ctx": 512,
            "n_gpu_layers": -1,
            "logits_all": True,
            "verbose": False,
        }
    ]
    assert load_model.cache_info().maxsize == 1
    assert load_model.cache_info().currsize == 1


def test_load_model_explicit_path_skips_download(model_loader, fake_model):
    constructors, downloads = model_loader
    assert load_model(model_path="/models/custom.gguf", n_ctx=128) is fake_model
    assert downloads == []
    assert constructors == [
        {
            "model_path": "/models/custom.gguf",
            "n_ctx": 128,
            "n_gpu_layers": -1,
            "logits_all": True,
            "verbose": False,
        }
    ]


def test_load_model_cache_retains_only_one_configuration(model_loader):
    constructors, downloads = model_loader
    load_model(model_path="/models/first.gguf", n_ctx=128)
    load_model(model_path="/models/first.gguf", n_ctx=128)
    load_model(model_path="/models/second.gguf", n_ctx=256)
    load_model(model_path="/models/first.gguf", n_ctx=128)
    assert [call["model_path"] for call in constructors] == [
        "/models/first.gguf",
        "/models/second.gguf",
        "/models/first.gguf",
    ]
    assert downloads == []
    assert load_model.cache_info().currsize == 1


def test_load_model_rejects_nonpositive_context_before_any_side_effect(model_loader):
    constructors, downloads = model_loader
    for n_ctx in (0, -1):
        with pytest.raises(ValueError, match="(?i)n_ctx|context"):
            load_model(n_ctx=n_ctx)
    assert constructors == []
    assert downloads == []


def test_classify_forwards_loading_settings_and_reuses_cached_model(model_loader):
    constructors, downloads = model_loader
    for _ in range(2):
        assert classify(
            "input", ["first", "second"], model_path="/models/custom.gguf", n_ctx=128
        )
    assert len(constructors) == 1
    assert constructors[0]["model_path"] == "/models/custom.gguf"
    assert constructors[0]["n_ctx"] == 128
    assert downloads == []


def test_classify_provided_model_uses_final_prompt_logits_and_exact_tokenization(
    monkeypatch, fake_model
):
    def unexpected(*args, **kwargs):
        pytest.fail("A provided model must not trigger loading or downloading.")

    monkeypatch.setattr(classifier_module, "load_model", unexpected)
    monkeypatch.setattr(classifier_module, "download_model", unexpected)
    choices = [" first ", "second", "third"]
    result = classify(
        "café",
        choices,
        instruction="Pick one.",
        model=fake_model,
        model_path="/ignored.gguf",
        n_ctx=1,
    )

    assert isinstance(result, dict)
    assert list(result) == choices
    assert all(isinstance(value, float) for value in result.values())
    assert result == pytest.approx(
        dict(zip(choices, softmax([-3, -2, -0.2]), strict=True))
    )
    assert sum(result.values()) == pytest.approx(1.0)
    assert fake_model.tokenize_calls == [
        (label.encode(), {"add_bos": False, "special": False}) for label in "ABC"
    ] + [
        (
            build_prompt("café", choices, "Pick one.").encode(),
            {"add_bos": False, "special": True},
        )
    ]
    assert fake_model.events == [("reset",), ("eval", fake_model.prompt_tokens)]
    assert fake_model.n_tokens == len(fake_model.prompt_tokens)


def test_repeated_classification_resets_before_each_prompt_eval(fake_model):
    first = classify("input", ["first", "second"], model=fake_model)
    classify("unrelated input", ["other", "different"], model=fake_model)
    repeated = classify("input", ["first", "second"], model=fake_model)
    assert first == repeated
    assert fake_model.events == [
        ("reset",),
        ("eval", fake_model.prompt_tokens),
        ("reset",),
        ("eval", fake_model.prompt_tokens),
        ("reset",),
        ("eval", fake_model.prompt_tokens),
    ]
    prompts = [text for text, kwargs in fake_model.tokenize_calls if kwargs["special"]]
    assert prompts[0] == prompts[2]
    assert prompts[0] != prompts[1]


def test_classify_supports_all_twenty_six_labels_without_top_twenty_limit(fake_model):
    choices = [f"choice {label}" for label in ascii_uppercase]
    result = classify("input", choices, model=fake_model)
    token_ids = [
        fake_model.label_tokens[label.encode()][0] for label in ascii_uppercase
    ]
    assert list(result) == choices
    np.testing.assert_allclose(
        list(result.values()), softmax(fake_model.scores[2, token_ids])
    )
    assert max(result, key=result.get) == "choice Z"
    assert len(fake_model.tokenize_calls) == 27


def test_classify_rejects_labels_that_are_not_single_tokens_before_eval(fake_model):
    for tokens in ([], [3, 4]):
        fake_model.label_tokens[b"B"] = tokens
        with pytest.raises(ValueError, match="(?i)token"):
            classify("input", ["first", "second"], model=fake_model)
    assert fake_model.events == []


def test_exact_context_is_allowed_but_overflow_fails_before_eval(fake_model):
    fake_model.context_size = len(fake_model.prompt_tokens)
    assert classify("input", ["first", "second"], model=fake_model)
    assert fake_model.n_tokens == fake_model.n_ctx()
    fake_model.events.clear()
    fake_model.context_size -= 1
    with pytest.raises(ValueError, match="(?i)context"):
        classify("input", ["first", "second"], model=fake_model)
    assert fake_model.events == []
