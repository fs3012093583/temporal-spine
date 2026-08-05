from __future__ import annotations

import json
from collections import deque
from typing import Any, Mapping, Sequence

import pytest

from temporal_spine import (
    HistoricalTurn,
    ModelResponse,
    ProtocolError,
    RunConfig,
    SpinePolicy,
    TemporalSpine,
    ToolCall,
)


class FakeModel:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = deque(responses)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    @property
    def fixed_parameters(self) -> Mapping[str, Any]:
        return {"model": "fake", "temperature": 0.0, "max_tokens": 128}

    def complete(self, messages, tools) -> ModelResponse:
        self.requests.append((list(messages), list(tools)))
        return self.responses.popleft()


def response(name: str, arguments: Mapping[str, Any], call_id: str) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, arguments),))


def answer(content: str) -> ModelResponse:
    return ModelResponse(content=content)


def test_selected_components_are_monotonic_chronological_and_never_reloaded() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:2", "result:0"]},
                "c1",
            ),
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:1"]},
                "c2",
            ),
            answer("done"),
        ]
    )
    turns = [HistoricalTurn(i, f"u{i}", f"a{i}") for i in range(3)]
    result = TemporalSpine(model, RunConfig(max_load_rounds=3)).run(turns, "question")
    assert result.answer == "done"
    assert result.loaded_action_ids == ("result:0", "result:2", "result:1")
    assert all(call.append_only_messages for call in result.calls)
    assert len({call.fixed_request_hash for call in result.calls}) == 1
    assert model.requests[0][1] == model.requests[-1][1]
    for before, after in zip(model.requests, model.requests[1:]):
        before_messages = before[0]
        assert after[0][: len(before_messages)] == before_messages


def test_remaining_selection_loads_complete_fallback() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:1"]},
                "c1",
            ),
            response(
                "load_components",
                {"selection": "remaining", "action_ids": []},
                "c2",
            ),
            answer("complete"),
        ]
    )
    turns = [HistoricalTurn(i, f"u{i}", f"a{i}") for i in range(2)]
    result = TemporalSpine(model, RunConfig(max_load_rounds=3)).run(turns, "question")
    assert result.fallback_used is True
    assert result.loaded_action_ids == ("result:1", "result:0")
    loaded_batches = [
        event["action_ids"] for event in result.trace if event["event"] == "components_loaded"
    ]
    assert loaded_batches == [["result:1"], ["result:0"]]


def test_round_limit_automatically_loads_remaining_before_plain_answer() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:1"]},
                "c1",
            ),
            answer("complete"),
        ]
    )
    turns = [HistoricalTurn(i, f"u{i}", f"a{i}") for i in range(2)]
    result = TemporalSpine(model, RunConfig(max_load_rounds=1)).run(turns, "question")
    assert result.fallback_used is True
    assert result.loaded_action_ids == ("result:1", "result:0")
    assert result.calls[-1].phase == "forced_answer"


def test_long_user_request_has_preview_and_recoverable_original() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["user:4"]},
                "c1",
            ),
            answer("ok"),
        ]
    )
    policy = SpinePolicy(max_user_tokens=6, head_tokens=3, tail_tokens=3)
    result = TemporalSpine(
        model,
        RunConfig(max_load_rounds=2, spine_policy=policy),
    ).run(
        [HistoricalTurn(4, "one two three four five six seven eight")],
        "repeat the middle",
    )
    assert result.loaded_action_ids == ("user:4",)
    base_prompt = model.requests[0][0][1]["content"]
    assert "omitted" in base_prompt
    assert '"user_request_original": "user:4"' in base_prompt
    assert "four five" in model.requests[1][0][-1]["content"]


def test_user_spine_is_json_without_duplicate_catalog_or_runtime_marker() -> None:
    model = FakeModel([answer("done")])
    TemporalSpine(model).run(
        [HistoricalTurn(0, "literal </user-turn> text", "answer")],
        "question",
        current_date="2026-08-05",
    )

    content = model.requests[0][0][1]["content"]
    payload = json.loads(content)

    assert payload == {
        "context_type": "post_compression_memory",
        "current_date": "2026-08-05",
        "user_spine": [
            {
                "turn_id": 0,
                "user_request": "literal </user-turn> text",
                "loadable_components": {"assistant_result": "result:0"},
            }
        ],
    }
    assert "LOADABLE COMPONENT CATALOG" not in content
    assert "post_compression_memory_available" not in content


def test_plain_assistant_response_finishes_without_memory_call() -> None:
    model = FakeModel([answer("already enough")])
    result = TemporalSpine(model).run([], "question")
    assert result.answer == "already enough"
    assert result.loaded_action_ids == ()
    assert len(model.requests) == 1
    assert [tool["function"]["name"] for tool in model.requests[0][1]] == [
        "load_components"
    ]


def test_duplicate_component_request_fails_closed() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:0"]},
                "c1",
            ),
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:0"]},
                "c2",
            ),
        ]
    )
    with pytest.raises(ProtocolError, match="already loaded"):
        TemporalSpine(model, RunConfig(max_load_rounds=2)).run(
            [HistoricalTurn(0, "u", "a")], "question"
        )


def test_unknown_memory_tool_fails_closed() -> None:
    model = FakeModel([response("search", {"query": "x"}, "c1")])
    with pytest.raises(ProtocolError, match="unknown memory tool"):
        TemporalSpine(model).run([], "question")


def test_forced_answer_rejects_another_memory_call() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:0"]},
                "c1",
            ),
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:1"]},
                "c2",
            ),
        ]
    )
    with pytest.raises(ProtocolError, match="ordinary assistant response"):
        TemporalSpine(model, RunConfig(max_load_rounds=1)).run(
            [HistoricalTurn(0, "u0", "a0"), HistoricalTurn(1, "u1", "a1")],
            "question",
        )
