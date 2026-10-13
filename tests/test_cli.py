import json

import pytest

from simple_jev import classifier as classifier_module
from simple_jev import cli
from simple_jev.classifier import DEFAULT_INSTRUCTION


def test_classify_json_is_pure_choice_mapping_and_forwards_flags(monkeypatch, capsys):
    calls = []

    def classify(text, choices, **kwargs):
        calls.append((text, choices, kwargs))
        return {"first": 0.25, "second": 0.75}

    monkeypatch.setattr(cli, "classify", classify)
    assert (
        cli.main(
            [
                "classify",
                "--text",
                "café",
                "--choices",
                "first",
                "second",
                "--instruction",
                "Pick one.",
                "--model-path",
                "/models/custom.gguf",
                "--n-ctx",
                "128",
                "--json",
            ]
        )
        == 0
    )

    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {"first": 0.25, "second": 0.75}
    assert calls == [
        (
            "café",
            ["first", "second"],
            {
                "instruction": "Pick one.",
                "model_path": "/models/custom.gguf",
                "n_ctx": 128,
            },
        )
    ]


def test_classify_default_settings_are_forwarded_without_explicit_download(
    monkeypatch, capsys
):
    calls = []

    def classify(text, choices, **kwargs):
        calls.append((text, choices, kwargs))
        return {"first": 0.25, "second": 0.75}

    def unexpected():
        pytest.fail("The classify command must leave model loading to classify().")

    monkeypatch.setattr(cli, "classify", classify)
    monkeypatch.setattr(cli, "download_model", unexpected)
    assert (
        cli.main(
            ["classify", "--text", "input", "--choices", "first", "second", "--json"]
        )
        == 0
    )
    assert calls == [
        (
            "input",
            ["first", "second"],
            {"instruction": DEFAULT_INSTRUCTION, "model_path": None, "n_ctx": 512},
        )
    ]
    assert capsys.readouterr().err == ""


def test_classify_human_output_is_unchanged_and_notes_uncalibrated_confidence(
    monkeypatch, fake_model, capsys
):
    monkeypatch.setattr(classifier_module, "load_model", lambda **kwargs: fake_model)
    assert (
        cli.main(["classify", "--text", "input", "--choices", "first", "second"]) == 0
    )

    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.splitlines() == [
        "Prediction: second",
        "  first: 0.2689",
        "  second: 0.7311",
        "Probabilities are normalized over choices, not calibrated confidence.",
    ]


def test_download_takes_no_arguments_and_prints_cached_model_path(monkeypatch, capsys):
    calls = []

    def download():
        calls.append(())
        return "/cache/snapshots/revision/Qwen3-0.6B-Q8_0.gguf"

    monkeypatch.setattr(cli, "download_model", download)
    assert cli.main(["download"]) == 0
    assert calls == [()]
    output = capsys.readouterr()
    assert output.out == "/cache/snapshots/revision/Qwen3-0.6B-Q8_0.gguf\n"
    assert output.err == ""


def test_invalid_choices_report_error_without_loading_or_downloading(
    monkeypatch, capsys
):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid choices must not load or download a model.")

    monkeypatch.setattr(classifier_module, "load_model", unexpected)
    monkeypatch.setattr(classifier_module, "download_model", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main(["classify", "--text", "input", "--choices", "only"])

    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("Error: ")
    assert "2" in output.err and "26" in output.err


@pytest.mark.parametrize("exception", [ValueError, RuntimeError, OSError])
def test_errors_exit_cleanly_without_traceback(exception, monkeypatch, capsys):
    failure = exception("model unavailable")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(cli, "classify", fail)
    monkeypatch.setattr(cli, "download_model", fail)
    for args in (
        ["classify", "--text", "input", "--choices", "first", "second"],
        ["download"],
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(args)
        assert error.value.code == 1
        output = capsys.readouterr()
        assert output.out == ""
        assert output.err == f"Error: {failure}\n"


def test_removed_server_options_and_download_options_are_parser_errors(capsys):
    classify_args = ["classify", "--text", "input", "--choices", "first", "second"]
    for option, value in (
        ("--server-url", "http://localhost:11434"),
        ("--timeout", "30"),
        ("--model", "qwen3:custom"),
    ):
        with pytest.raises(SystemExit) as error:
            cli.main([*classify_args, option, value])
        assert error.value.code == 2
        output = capsys.readouterr()
        assert output.out == ""
        assert "unrecognized arguments" in output.err

    for option, value in (
        ("--model", "qwen3:custom"),
        ("--server-url", "http://localhost:11434"),
        ("--timeout", "30"),
        ("--model-path", "/models/custom.gguf"),
        ("--n-ctx", "128"),
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(["download", option, value])
        assert error.value.code == 2
        output = capsys.readouterr()
        assert output.out == ""
        assert "unrecognized arguments" in output.err
