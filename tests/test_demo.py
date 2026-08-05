from __future__ import annotations

import pytest

from typing import Any, Mapping

from temporal_spine.demo import RecordingChatModel, parse_history_json, result_metrics
from temporal_spine.model import ModelResponse
from temporal_spine.runner import CallAudit, RunResult


def test_parse_history_json_builds_source_faithful_turns() -> None:
    turns = parse_history_json(
        """
        [
          {
            "turn_id": 3,
            "user_request": "remember this",
            "assistant_result": "saved",
            "execution_records": ["tool output"],
            "timestamp": "2026-08-04"
          }
        ]
        """
    )
    assert turns[0].turn_id == 3
    assert turns[0].execution_records == ("tool output",)
    assert turns[0].timestamp == "2026-08-04"


@pytest.mark.parametrize(
    ("raw_history", "message"),
    [
        ("{}", "must be a list"),
        ('[{"turn_id": true, "user_request": "x"}]', "must be an integer"),
        ('[{"turn_id": 0, "user_request": ""}]', "non-empty string"),
    ],
)
def test_parse_history_json_rejects_bad_shapes(
    raw_history: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_history_json(raw_history)


def test_result_metrics_sums_provider_usage() -> None:
    result = RunResult(
        answer="done",
        loaded_action_ids=("result:1",),
        fallback_used=False,
        calls=(
            CallAudit(0, "agent", "hash", True, {"prompt_tokens": 10}),
            CallAudit(1, "agent", "hash", True, {"prompt_tokens": 14}),
        ),
        trace=(),
    )
    metrics = result_metrics(result)
    assert metrics["model_calls"] == 2
    assert metrics["total_usage"] == {"prompt_tokens": 24}


class _FakeModel:
    @property
    def fixed_parameters(self) -> Mapping[str, Any]:
        return {"model": "fake", "temperature": 0.0}

    def complete(self, messages, tools) -> ModelResponse:
        return ModelResponse(content="done")


def test_recording_model_captures_exact_request_body_without_credentials() -> None:
    recorder = RecordingChatModel(_FakeModel())
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "load_components"}}]

    recorder.complete(messages, tools)
    recorded = recorder.numbered_requests()

    assert recorded == [
        {
            "call_index": 0,
            "request_body": {
                "model": "fake",
                "temperature": 0.0,
                "messages": messages,
                "tools": tools,
            },
        }
    ]
    assert "api_key" not in recorded[0]["request_body"]
