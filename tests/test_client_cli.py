import io
import json
import urllib.error

import pytest

from temporal_spine import OpenAICompatibleClient, ProviderError
from temporal_spine.cli import load_history, main


def payload(name="submit_final_result", arguments=None):
    return {
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": None,
            "reasoning_content": "synthetic reasoning",
            "tool_calls": [{"id": "call-test", "function": {
                "name": name,
                "arguments": json.dumps(arguments or {"result": "done"}),
            }}],
        }}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                  "prompt_tokens_details": {"cached_tokens": 80},
                  "completion_tokens_details": {"reasoning_tokens": 10}},
    }


@pytest.mark.parametrize("url", ["https://provider.example/v1", "https://provider.example/v1/", "https://provider.example/v1/chat/completions"])
def test_api_root_and_full_endpoint_send_same_native_request(monkeypatch, url):
    captured = []

    def transport(request, timeout):
        captured.append((request, timeout))
        return io.BytesIO(json.dumps(payload()).encode())

    monkeypatch.setattr("urllib.request.urlopen", transport)
    client = OpenAICompatibleClient("synthetic-model", url, api_key="synthetic-test-value")
    result = client.complete([{"role": "user", "content": "q"}], [{"type": "function"}])
    request, timeout = captured[0]
    assert request.full_url == "https://provider.example/v1/chat/completions"
    body = json.loads(request.data)
    assert body["max_tokens"] == 8192
    assert body["messages"] == [{"role": "user", "content": "q"}]
    assert "api_key" not in body
    assert timeout == 120
    assert result.usage["prompt_cached_tokens"] == 80
    assert result.usage["completion_reasoning_tokens"] == 10
    assert result.as_assistant_message()["reasoning_content"] == "synthetic reasoning"
    assert "synthetic-test-value" not in repr(client)


def test_provider_specific_output_parameter_and_temperature_omission():
    client = OpenAICompatibleClient("model", "https://provider.example/v1", api_key="test",
                                    token_parameter="max_completion_tokens", temperature=None,
                                    thinking={"type": "disabled"})
    assert client.fixed_parameters == {
        "model": "model", "max_completion_tokens": 8192, "thinking": {"type": "disabled"},
    }


@pytest.mark.parametrize("url", ["http://provider.example/v1", "https://user:password@example.com/v1", "https://example.com/v1?key=private", "not-a-url"])
def test_unsafe_urls_rejected(url):
    with pytest.raises(ValueError):
        OpenAICompatibleClient("model", url, api_key="test")


def test_localhost_api_root_allowed():
    client = OpenAICompatibleClient("local", "http://localhost:8000/v1", api_key="local")
    assert client.base_url == "http://localhost:8000/v1/chat/completions"


@pytest.mark.parametrize("invalid", [
    {}, {"choices": []}, {"choices": [{"message": {"content": ["not text"]}}]},
    {"choices": [{"message": {"tool_calls": [{"id": "x", "function": {"name": "x", "arguments": "[]"}}]}}]},
])
def test_malformed_provider_output_fails_explicitly(monkeypatch, invalid):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps(invalid).encode()))
    client = OpenAICompatibleClient("model", "https://provider.example/v1", api_key="test")
    with pytest.raises(ProviderError, match="malformed"):
        client.complete([], [])


def test_http_error_never_echoes_private_body_or_retries(monkeypatch):
    requests = []

    def fail(request, timeout):
        requests.append(request)
        raise urllib.error.HTTPError(request.full_url, 401, "private-secret", {}, io.BytesIO(b"private-history"))

    monkeypatch.setattr("urllib.request.urlopen", fail)
    client = OpenAICompatibleClient("model", "https://provider.example/v1", api_key="test")
    with pytest.raises(ProviderError, match="HTTP 401") as info:
        client.complete([], [])
    assert "private" not in str(info.value)
    assert len(requests) == 1


def test_cli_demo_is_offline_and_reports_expected_access(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("offline demo attempted network access")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    assert main(["demo"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["loaded_action_ids"] == ["result:1"]
    assert result["model_calls"] == 2 and result["access_calls"] == 1
    assert result["protocol_followed"] is True


def test_cli_run_end_to_end_with_mock_transport(monkeypatch, capsys, tmp_path):
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"turns": [{"turn_id": 7, "user_request": "request", "assistant_result": "PRIVATE_ORIGINAL"}]}))
    requests = []

    def transport(request, timeout):
        body = json.loads(request.data)
        requests.append(body)
        if len(requests) == 1:
            assert "PRIVATE_ORIGINAL" not in json.dumps(body["messages"])
            data = payload("load_components", {"selection": "selected", "action_ids": ["result:7"]})
        else:
            assert "PRIVATE_ORIGINAL" in body["messages"][-1]["content"]
            assert body["messages"][:len(requests[0]["messages"])] == requests[0]["messages"]
            data = payload()
        return io.BytesIO(json.dumps(data).encode())

    monkeypatch.setattr("urllib.request.urlopen", transport)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-value")
    assert main(["run", "--history", str(history), "--question", "q", "--model", "synthetic-model", "--base-url", "https://provider.example/v1"]) == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["answer"] == "done" and result["loaded_action_ids"] == ["result:7"]
    assert result["protocol_followed"] is True
    assert "PRIVATE_ORIGINAL" not in out and "synthetic-test-value" not in out
    assert requests[1]["messages"][-2]["reasoning_content"] == "synthetic reasoning"


@pytest.mark.parametrize("data", [[], {}, {"turns": {}}, {"turns": [{"turn_id": 0}]},
                                       {"turns": [{"turn_id": 0, "user_request": "q", "extra": True}]}])
def test_history_schema_rejects_invalid_input(tmp_path, data):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_history(path)


def test_cli_invalid_history_makes_no_api_request(monkeypatch, tmp_path, capsys):
    path = tmp_path / "history.json"
    path.write_text("{}")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("unexpected API request"))
    assert main(["run", "--history", str(path), "--question", "q", "--model", "x", "--base-url", "https://provider.example/v1"]) == 1
    assert "history must" in capsys.readouterr().err
