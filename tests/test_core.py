from __future__ import annotations

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
            response("commit_final_result", {}, "c3"),
            response("submit_final_result", {"result": "done"}, "c4"),
        ]
    )
    turns = [HistoricalTurn(i, f"u{i}", f"a{i}") for i in range(3)]
    result = TemporalSpine(model, RunConfig(max_control_rounds=3)).run(turns, "question")
    assert result.answer == "done"
    assert result.loaded_action_ids == ("result:0", "result:2", "result:1")
    assert all(call.append_only_messages for call in result.calls)
    assert len({call.fixed_request_hash for call in result.calls}) == 1
    assert model.requests[0][1] == model.requests[-1][1]
    for before, after in zip(model.requests, model.requests[1:]):
        before_messages = before[0]
        assert after[0][: len(before_messages)] == before_messages


def test_full_fallback_loads_only_remaining_components() -> None:
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
            response("submit_final_result", {"result": "complete"}, "c3"),
        ]
    )
    turns = [HistoricalTurn(i, f"u{i}", f"a{i}") for i in range(2)]
    result = TemporalSpine(model, RunConfig(max_control_rounds=2)).run(turns, "question")
    assert result.fallback_used is True
    assert result.loaded_action_ids == ("result:1", "result:0")
    loaded_batches = [event["action_ids"] for event in result.trace if event["event"] == "components_loaded"]
    assert loaded_batches == [["result:1"], ["result:0"]]


def test_long_user_request_has_preview_and_recoverable_original() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["user:4"]},
                "c1",
            ),
            response("commit_final_result", {}, "c2"),
            response("submit_final_result", {"result": "ok"}, "c3"),
        ]
    )
    policy = SpinePolicy(max_user_tokens=6, head_tokens=3, tail_tokens=3)
    result = TemporalSpine(model, RunConfig(max_control_rounds=2, spine_policy=policy)).run(
        [HistoricalTurn(4, "one two three four five six seven eight")],
        "repeat the middle",
    )
    assert result.loaded_action_ids == ("user:4",)
    base_prompt = model.requests[0][0][1]["content"]
    assert "omitted" in base_prompt
    assert "user_original_action_id=user:4" in base_prompt
    assert "four five" in model.requests[1][0][-1]["content"]


def test_submit_before_authorization_fails_closed() -> None:
    model = FakeModel([response("submit_final_result", {"result": "unsafe"}, "c1")])
    with pytest.raises(ProtocolError, match="forbidden before authorization"):
        TemporalSpine(model).run([], "question")


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
        TemporalSpine(model, RunConfig(max_control_rounds=2)).run(
            [HistoricalTurn(0, "u", "a")], "question"
        )


def test_one_shot_commit_can_focus_visible_user_turns() -> None:
    model = FakeModel(
        [
            response("commit_final_result", {"focus_turn_ids": [1]}, "c1"),
            response("submit_final_result", {"result": "personalized"}, "c2"),
        ]
    )
    result = TemporalSpine(model, RunConfig(max_control_rounds=1)).run(
        [HistoricalTurn(1, "I cannot eat dairy.")], "suggest breakfast"
    )
    assert result.answer == "personalized"
    assert result.focused_turn_ids == (1,)
    assert "I cannot eat dairy" in model.requests[1][0][-1]["content"]
    assert "final_result_authorized" in model.requests[1][0][-1]["content"]
