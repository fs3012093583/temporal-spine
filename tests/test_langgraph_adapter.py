from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import pytest

pytest.importorskip("langgraph")

from temporal_spine import HistoricalTurn, ModelResponse, RunConfig, ToolCall
from temporal_spine.langgraph_adapter import LangGraphTemporalSpine


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


def test_graph_routes_memory_call_back_to_model() -> None:
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "c1",
                        "load_components",
                        {"selection": "selected", "action_ids": ["result:1"]},
                    ),
                )
            ),
            ModelResponse(content="Oatmeal is safe; avoid peanut-butter toast."),
        ]
    )
    result = LangGraphTemporalSpine(
        model,
        RunConfig(max_load_rounds=2, full_fallback_enabled=False),
    ).run(
        [
            HistoricalTurn(0, "I am allergic to peanuts.", "Understood."),
            HistoricalTurn(1, "Suggest breakfast.", "Peanut toast or oatmeal."),
        ],
        "Which option is safe?",
    )
    assert result.answer == "Oatmeal is safe; avoid peanut-butter toast."
    assert result.loaded_action_ids == ("result:1",)
    assert len(result.calls) == 2
    assert all(call.append_only_messages for call in result.calls)


def test_graph_ends_on_plain_response_without_memory_loading() -> None:
    model = FakeModel([ModelResponse(content="No archived result is needed.")])
    result = LangGraphTemporalSpine(model).run([], "Say hello.")
    assert result.answer == "No archived result is needed."
    assert result.loaded_action_ids == ()
    assert len(model.requests) == 1


def test_graph_keeps_recent_uncompressed_messages_before_current_request() -> None:
    model = FakeModel([ModelResponse(content="The recent answer was blue.")])
    LangGraphTemporalSpine(model).run(
        [HistoricalTurn(0, "Pick a color.", "I picked red.")],
        "What was the most recent answer?",
        recent_messages=[
            {"role": "user", "content": "Change it to blue."},
            {"role": "assistant", "content": "The color is now blue."},
        ],
    )
    messages = model.requests[0][0]
    assert messages[-3:] == [
        {"role": "user", "content": "Change it to blue."},
        {"role": "assistant", "content": "The color is now blue."},
        {
            "role": "user",
            "content": "[CURRENT REQUEST]\nWhat was the most recent answer?",
        },
    ]
