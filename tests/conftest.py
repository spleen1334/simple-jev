"""Fake top-level HTTP calls; no Ollama server or model download is needed."""

from copy import deepcopy

import httpx
import pytest

from simple_jev import classifier as classifier_module


@pytest.fixture
def ollama(monkeypatch):
    class FakeOllama:
        def __init__(self):
            self.requests = []
            self.status_code = 200
            self.generate_payload = {
                "response": "C",
                "prompt_eval_count": 100,
                "context": [1, 2, 3],
                "logprobs": [
                    {
                        "token": "C",
                        "logprob": -0.2,
                        "top_logprobs": [
                            {"token": "B", "logprob": -2.0},
                            {"token": "C", "logprob": -0.2},
                            {"token": "A", "logprob": -3.0},
                        ],
                    }
                ],
            }

        def post(self, url, **kwargs):
            request = httpx.Request("POST", url, json=kwargs["json"])
            self.requests.append((request, deepcopy(kwargs)))
            if request.url.path == "/api/generate":
                payload = self.generate_payload
            elif request.url.path == "/api/pull":
                payload = {"status": "success"}
            else:
                pytest.fail(f"Unexpected HTTP endpoint: {request.url}")
            return httpx.Response(
                self.status_code, json=deepcopy(payload), request=request
            )

    fake = FakeOllama()
    monkeypatch.setattr(classifier_module.httpx, "post", fake.post)
    return fake
