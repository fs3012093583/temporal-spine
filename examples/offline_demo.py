from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

from temporal_spine import HistoricalTurn, ModelResponse, RunConfig, TemporalSpine, ToolCall


class ScriptedModel:
    """Offline stand-in that makes post-compression loading visible."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = deque(responses)

    @property
    def fixed_parameters(self) -> Mapping[str, Any]:
        return {"model": "offline-scripted", "temperature": 0, "max_tokens": 256}

    def complete(self, messages, tools) -> ModelResponse:
        return self.responses.popleft()


history = [
    HistoricalTurn(0, "I am allergic to peanuts.", "Understood. I will avoid peanuts."),
    HistoricalTurn(1, "Suggest a quick breakfast.", "Try peanut-butter toast or oatmeal."),
]
model = ScriptedModel(
    [
        ModelResponse(tool_calls=(ToolCall("c1", "load_components", {
            "selection": "selected", "action_ids": ["result:1"]
        }),)),
        ModelResponse(
            content="Choose oatmeal and avoid the peanut-butter toast because of your allergy."
        ),
    ]
)

result = TemporalSpine(model, RunConfig(max_load_rounds=2)).run(
    history,
    "Which earlier breakfast suggestion is safe for me?",
)
print(result.answer)
print("loaded:", result.loaded_action_ids)
print("fixed request hashes:", [call.fixed_request_hash for call in result.calls])
