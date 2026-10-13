import httpx
import numpy as np
import pytest

from simple_jev import classify, download_model
from simple_jev.classifier import (
    DEFAULT_INSTRUCTION,
    build_prompt,
    probabilities_from_logprobs,
)


def test_probabilities_are_stable_normalized_and_offset_invariant():
    values = np.array([-3.0, -2.0, -0.2])
    probabilities = probabilities_from_logprobs(values.tolist())

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.argmax(probabilities) == np.argmax(values)
    np.testing.assert_allclose(probabilities_from_logprobs(values + 1e6), probabilities)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        np.testing.assert_allclose(
            probabilities_from_logprobs([10000, -10000]), [1.0, 0.0]
        )
        np.testing.assert_allclose(
            probabilities_from_logprobs([1e300, 1e300]), [0.5, 0.5]
        )


def test_probabilities_reject_invalid_shape_size_and_nonfinite_values():
    for values in ([], [1], [[1, 2]], [0, np.nan], [0, np.inf], [-np.inf, 0]):
        with pytest.raises(ValueError):
            probabilities_from_logprobs(values)


def test_build_prompt_has_qwen_structure_and_custom_instruction():
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


def test_build_prompt_accepts_choice_count_boundaries_and_preserves_choices():
    for count in (2, 20):
        choices = [f" Choice {index} " for index in range(count)]
        prompt = build_prompt("input", choices)
        for label, choice in zip("ABCDEFGHIJKLMNOPQRST", choices, strict=False):
            assert f"{label}. {choice}" in prompt


@pytest.mark.parametrize(
    "choices",
    [
        ["only"],
        [str(index) for index in range(21)],
        ["first", " \t"],
        [" Straße ", "STRASSE"],
        "AB",
    ],
)
def test_build_prompt_rejects_invalid_choices(choices):
    with pytest.raises(ValueError):
        build_prompt("input", choices)


def test_build_prompt_rejects_blank_text_and_instruction():
    with pytest.raises(ValueError, match="(?i)input"):
        build_prompt(" \t", ["first", "second"])
    with pytest.raises(ValueError, match="(?i)instruction"):
        build_prompt("input", ["first", "second"], " \n")


def test_classify_returns_choice_mapping_using_only_first_token_scores(ollama):
    ollama.generate_payload["response"] = "unrelated"
    ollama.generate_payload["logprobs"].append(
        {
            "top_logprobs": [
                {"token": "A", "logprob": 0.0},
                {"token": "B", "logprob": -100.0},
            ]
        }
    )
    choices = [" first ", "second"]
    result = classify("input", choices)

    assert isinstance(result, dict)
    assert list(result) == choices
    assert all(isinstance(value, float) for value in result.values())
    assert result == pytest.approx({" first ": 0.2689414214, "second": 0.7310585786})
    assert sum(result.values()) == pytest.approx(1.0)
    assert len(ollama.requests) == 1
    request, kwargs = ollama.requests[0]
    assert str(request.url) == "http://127.0.0.1:11434/api/generate"
    assert kwargs["json"]["model"] == "qwen3:0.6b"
    assert kwargs["json"]["options"]["num_ctx"] == 2048
    assert kwargs["timeout"] == 120.0
    assert kwargs["trust_env"] is False


def test_classify_posts_exact_unfiltered_payload_to_configured_url(ollama):
    classify(
        "café",
        ["first", "second"],
        instruction="Pick one.",
        model="qwen3:custom",
        server_url="https://ollama.example///",
        n_ctx=512,
        timeout=17.5,
    )

    assert len(ollama.requests) == 1
    request, kwargs = ollama.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://ollama.example/api/generate"
    assert kwargs == {
        "json": {
            "model": "qwen3:custom",
            "prompt": build_prompt("café", ["first", "second"], "Pick one."),
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "truncate": False,
            "shift": False,
            "keep_alive": "5m",
            "options": {
                "num_predict": 1,
                "num_ctx": 512,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
        },
        "timeout": 17.5,
        "trust_env": False,
    }


def test_repeated_classification_does_not_reuse_context_or_model_state(ollama):
    first = classify("input", ["first", "second"])
    classify("unrelated input", ["other", "different"])
    repeated = classify("input", ["first", "second"])

    assert first == repeated
    assert len(ollama.requests) == 3
    bodies = [kwargs["json"] for _, kwargs in ollama.requests]
    assert bodies[0] == bodies[2]
    assert bodies[1]["prompt"] != bodies[0]["prompt"]
    assert all("context" not in body for body in bodies)


def test_missing_exact_label_fails_without_selected_token_fallback(ollama):
    first = ollama.generate_payload["logprobs"][0]
    first["token"] = "A"
    first["logprob"] = -3.0
    first["top_logprobs"] = [
        {"token": " A", "logprob": -3.0},
        {"token": "B", "logprob": -2.0},
    ]

    with pytest.raises(ValueError, match="(?i)top 20"):
        classify("input", ["first", "second"])


@pytest.mark.parametrize("logprobs", [None, []])
def test_missing_logprobs_requests_ollama_update(logprobs, ollama):
    if logprobs is None:
        ollama.generate_payload.pop("logprobs")
    else:
        ollama.generate_payload["logprobs"] = logprobs

    with pytest.raises(ValueError, match="(?i)update Ollama"):
        classify("input", ["first", "second"])


def test_context_must_leave_room_for_generated_token(ollama):
    ollama.generate_payload["prompt_eval_count"] = 127
    assert classify("input", ["first", "second"], n_ctx=128)
    for count in (128, 129):
        ollama.generate_payload["prompt_eval_count"] = count
        with pytest.raises(ValueError, match="(?i)context"):
            classify("input", ["first", "second"], n_ctx=128)


def test_download_posts_model_without_streaming_and_returns_model(ollama):
    assert download_model() == "qwen3:0.6b"
    assert (
        download_model(
            model="qwen3:custom", server_url="https://ollama.example/", timeout=30.0
        )
        == "qwen3:custom"
    )

    assert len(ollama.requests) == 2
    for (request, kwargs), url, model, timeout in zip(
        ollama.requests,
        ["http://127.0.0.1:11434/api/pull", "https://ollama.example/api/pull"],
        ["qwen3:0.6b", "qwen3:custom"],
        [600.0, 30.0],
        strict=True,
    ):
        assert str(request.url) == url
        assert kwargs == {
            "json": {"model": model, "stream": False},
            "timeout": timeout,
            "trust_env": False,
        }


def test_http_status_errors_propagate_before_response_processing(ollama):
    ollama.status_code = 503
    ollama.generate_payload = {}
    with pytest.raises(httpx.HTTPStatusError):
        classify("input", ["first", "second"])
    with pytest.raises(httpx.HTTPStatusError):
        download_model()
