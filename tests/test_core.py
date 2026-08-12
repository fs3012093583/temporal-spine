from __future__ import annotations

import hashlib
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
from temporal_spine.prompt import SYSTEM_PROMPT, TOOLS


class FakeModel:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = deque(responses)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    @property
    def fixed_parameters(self) -> Mapping[str, Any]:
        return {
            "model": "fake",
            "temperature": 0.0,
            "max_tokens": 256,
            "thinking": {"type": "disabled"},
        }

    def complete(self, messages, tools) -> ModelResponse:
        self.requests.append((list(messages), list(tools)))
        return self.responses.popleft()


def response(name: str, arguments: Mapping[str, Any], call_id: str) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, arguments),))


def turns(count: int) -> list[HistoricalTurn]:
    return [HistoricalTurn(i, f"user request {i}", f"assistant result {i}") for i in range(count)]


def test_direct_submit_needs_only_one_model_call() -> None:
    model = FakeModel(
        [response("submit_final_result", {"result": "answer from the spine"}, "c1")]
    )
    result = TemporalSpine(model).run(turns(2), "question")
    assert result.answer == "answer from the spine"
    assert result.loaded_action_ids == ()
    assert result.protocol_followed is True
    assert result.submission_mode == "native_submit_final_result"
    assert len(result.calls) == 1


def test_one_shot_load_then_submit_is_append_only_and_stable() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:2", "result:0"]},
                "c1",
            ),
            response("submit_final_result", {"result": "done"}, "c2"),
        ]
    )
    result = TemporalSpine(model).run(turns(3), "question")
    assert result.answer == "done"
    assert result.loaded_action_ids == ("result:0", "result:2")
    assert len(result.calls) == 2
    assert all(call.append_only_messages for call in result.calls)
    assert len({call.fixed_request_hash for call in result.calls}) == 1
    assert model.requests[0][1] == model.requests[1][1]
    before = model.requests[0][0]
    after = model.requests[1][0]
    assert after[: len(before)] == before
    assert model.requests[0][0][0] == model.requests[1][0][0]


def test_tools_expose_only_load_and_submit_without_fallback_or_focus() -> None:
    model = FakeModel(
        [response("submit_final_result", {"result": "done"}, "c1")]
    )
    TemporalSpine(model).run([], "question")
    tools = model.requests[0][1]
    assert [tool["function"]["name"] for tool in tools] == [
        "load_components",
        "submit_final_result",
    ]
    load_parameters = tools[0]["function"]["parameters"]
    assert set(load_parameters["properties"]) == {"selection", "action_ids"}
    assert load_parameters["properties"]["selection"]["enum"] == ["selected"]
    assert load_parameters["properties"]["action_ids"]["maxItems"] == 5
    serialized = json.dumps(tools)
    assert "remaining" not in serialized
    assert "focus" not in serialized
    assert "commit_final_result" not in serialized


def test_system_and_tool_hashes_match_the_latest_evaluation_protocol() -> None:
    canonical_tools = json.dumps(
        TOOLS, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest() == (
        "1f6116d0ad93cbbd9fe7cf7be0aac1cfac667e313ffa4d568be50dd8bd8ef70e"
    )
    assert hashlib.sha256(canonical_tools).hexdigest() == (
        "61c098bb138dcab0c645d335f846b21662f0501ec0c80b840042ff5dbf1c712a"
    )


def test_overflow_keeps_first_five_then_presents_them_chronologically() -> None:
    requested = [
        "result:6",
        "result:4",
        "result:2",
        "result:0",
        "result:5",
        "result:1",
        "result:3",
    ]
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": requested},
                "c1",
            ),
            response("submit_final_result", {"result": "done"}, "c2"),
        ]
    )
    result = TemporalSpine(model).run(turns(7), "question")
    assert result.loaded_action_ids == (
        "result:0",
        "result:2",
        "result:4",
        "result:5",
        "result:6",
    )
    assert result.dropped_action_ids == ("result:1", "result:3")
    payload = json.loads(model.requests[1][0][-1]["content"])
    assert payload["load_limit_notice"]["retained_requested_action_ids"] == requested[:5]
    assert payload["load_limit_notice"]["dropped_action_ids"] == requested[5:]
    state = payload["history_access_state"]
    assert state["history_access_closed"] is True
    assert state["history_access_calls_remaining"] == 0
    assert state["allowed_next_tools"] == ["submit_final_result"]


def test_loaded_result_places_paired_user_request_before_hidden_content() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:7"]},
                "c1",
            ),
            response("submit_final_result", {"result": "ok"}, "c2"),
        ]
    )
    TemporalSpine(model).run(
        [HistoricalTurn(7, "What did the doctor recommend?", "Rest for two days.")],
        "repeat the recommendation",
    )
    tool_json = model.requests[1][0][-1]["content"]
    component = json.loads(tool_json)["components"][0]
    assert list(component).index("paired_visible_user_request") < list(component).index(
        "authoritative_content"
    )
    assert component["paired_visible_user_request"]["authoritative_content"] == (
        "What did the doctor recommend?"
    )
    assert component["authoritative_content"] == "Rest for two days."


def test_long_user_request_has_preview_and_recoverable_original() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["user:4"]},
                "c1",
            ),
            response("submit_final_result", {"result": "ok"}, "c2"),
        ]
    )
    policy = SpinePolicy(max_user_tokens=6, head_tokens=3, tail_tokens=3)
    result = TemporalSpine(model, RunConfig(spine_policy=policy)).run(
        [HistoricalTurn(4, "one two three four five six seven eight")],
        "repeat the middle",
    )
    assert result.loaded_action_ids == ("user:4",)
    base_prompt = model.requests[0][0][1]["content"]
    assert "omitted" in base_prompt
    assert "user_original_action_id=user:4" in base_prompt
    payload = json.loads(model.requests[1][0][-1]["content"])
    assert "four five" in payload["components"][0]["authoritative_content"]


def test_second_history_access_is_rejected() -> None:
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
            response("submit_final_result", {"result": "recovered"}, "c3"),
        ]
    )
    result = TemporalSpine(model).run(turns(2), "question")
    assert result.answer == "recovered"
    assert result.invalid_action_count == 1
    assert result.protocol_followed is False


def test_unknown_retained_action_id_fails_closed() -> None:
    model = FakeModel(
        [
            response(
                "load_components",
                {"selection": "selected", "action_ids": ["result:999"]},
                "c1",
            ),
            response("submit_final_result", {"result": "fallback answer"}, "c2"),
        ]
    )
    result = TemporalSpine(model).run(turns(1), "question")
    assert result.answer == "fallback answer"
    assert result.invalid_action_count == 1


def test_empty_final_result_is_not_valid() -> None:
    model = FakeModel(
        [
            response("submit_final_result", {"result": "   "}, "c1"),
            response("submit_final_result", {"result": "   "}, "c2"),
        ]
    )
    with pytest.raises(ProtocolError, match="non-empty result"):
        TemporalSpine(model).run([], "question")


def test_natural_language_final_is_captured_and_audited() -> None:
    model = FakeModel([ModelResponse(content="plain answer")])
    result = TemporalSpine(model).run([], "question")
    assert result.answer == "plain answer"
    assert result.protocol_followed is False
    assert result.submission_mode == "runtime_captured_natural_result"


def test_natural_language_capture_can_be_disabled() -> None:
    model = FakeModel(
        [ModelResponse(content="plain answer"), ModelResponse(content="plain answer")]
    )
    with pytest.raises(ProtocolError, match="authorized function"):
        TemporalSpine(model, RunConfig(capture_natural_final=False)).run([], "question")
