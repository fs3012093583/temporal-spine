from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ToolCall:
    """One native function call returned by a chat model."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    """Normalized response used by the runtime."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    reasoning_content: str | None = None

    def as_assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": _json_arguments(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        return message


def _json_arguments(arguments: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(arguments), ensure_ascii=False, separators=(",", ":"))


class ChatModel(Protocol):
    """Minimal interface required by :class:`TemporalSpine`."""

    @property
    def fixed_parameters(self) -> Mapping[str, Any]: ...

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelResponse: ...
