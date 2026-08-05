from __future__ import annotations

import json
from typing import Any

from temporal_spine import OpenAICompatibleClient


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3},
            }
        ).encode("utf-8")


def test_request_observer_receives_exact_transmitted_utf8_bytes(monkeypatch) -> None:
    observed: list[bytes] = []
    transmitted: list[bytes] = []

    def fake_urlopen(request, timeout):
        transmitted.append(request.data)
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        model="fake",
        base_url="https://example.test/chat/completions",
        api_key="secret",
        request_observer=observed.append,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "你好"}]
    client.complete(messages, [])

    assert observed == transmitted
    assert json.loads(observed[0]) == {
        "model": "fake",
        "temperature": 0.0,
        "max_tokens": 512,
        "messages": messages,
        "tools": [],
    }
