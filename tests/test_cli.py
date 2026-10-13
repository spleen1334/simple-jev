import json

import httpx
import pytest

from simple_jev import cli
from simple_jev.classifier import build_prompt


def test_classify_json_is_pure_choice_mapping_and_forwards_flags(ollama, capsys):
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
                "--model",
                "qwen3:custom",
                "--server-url",
                "http://localhost:11434/",
                "--timeout",
                "17.5",
                "--n-ctx",
                "128",
                "--json",
            ]
        )
        == 0
    )

    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == pytest.approx(
        {"first": 0.2689414214, "second": 0.7310585786}
    )
    assert len(ollama.requests) == 1
    request, kwargs = ollama.requests[0]
    assert str(request.url) == "http://localhost:11434/api/generate"
    assert kwargs["timeout"] == 17.5
    assert kwargs["json"]["model"] == "qwen3:custom"
    assert kwargs["json"]["options"]["num_ctx"] == 128
    assert kwargs["json"]["prompt"] == build_prompt(
        "café", ["first", "second"], "Pick one."
    )


def test_classify_human_output_is_simple_and_notes_uncalibrated_confidence(
    ollama, capsys
):
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


def test_download_forwards_keyword_arguments_and_prints_model(monkeypatch, capsys):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return kwargs["model"]

    monkeypatch.setattr(cli, "download_model", download)
    assert cli.main(["download"]) == 0
    assert (
        cli.main(
            [
                "download",
                "--model",
                "qwen3:custom",
                "--server-url",
                "https://ollama.example/",
                "--timeout",
                "30",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "model": "qwen3:0.6b",
            "server_url": "http://127.0.0.1:11434",
            "timeout": 600.0,
        },
        {
            "model": "qwen3:custom",
            "server_url": "https://ollama.example/",
            "timeout": 30.0,
        },
    ]
    output = capsys.readouterr()
    assert output.out == "qwen3:0.6b\nqwen3:custom\n"
    assert output.err == ""


def test_invalid_choices_report_error_without_http(ollama, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["classify", "--text", "input", "--choices", "only"])

    assert error.value.code == 1
    assert ollama.requests == []
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("Error: ")
    assert "2" in output.err and "20" in output.err


@pytest.mark.parametrize(
    ("command", "failure"),
    [
        ("classify", ValueError("Update Ollama to support logprobs.")),
        ("download", httpx.ConnectError("connection refused")),
    ],
)
def test_errors_exit_cleanly_without_traceback(command, failure, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(cli, "classify", fail)
    monkeypatch.setattr(cli, "download_model", fail)
    args = [command]
    if command == "classify":
        args += ["--text", "input", "--choices", "first", "second"]

    with pytest.raises(SystemExit) as error:
        cli.main(args)

    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == f"Error: {failure}\n"
