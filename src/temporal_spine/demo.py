from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from .memory import HistoricalTurn
from .model import ChatModel, ModelResponse
from .runner import RunResult


class RecordingChatModel:
    """Record the exact JSON bodies passed to an OpenAI-compatible model."""

    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.request_bodies: list[dict[str, Any]] = []

    @property
    def fixed_parameters(self) -> Mapping[str, Any]:
        return self.model.fixed_parameters

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelResponse:
        body = {
            **dict(self.fixed_parameters),
            "messages": copy.deepcopy(list(messages)),
            "tools": copy.deepcopy(list(tools)),
        }
        self.request_bodies.append(body)
        return self.model.complete(messages, tools)

    def numbered_requests(self) -> list[dict[str, Any]]:
        return [
            {"call_index": index, "request_body": body}
            for index, body in enumerate(self.request_bodies)
        ]


def parse_history_json(raw_history: str) -> list[HistoricalTurn]:
    """Parse the editable compressed-history format used by the local demo."""

    try:
        payload = json.loads(raw_history)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"history JSON is invalid at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError("history JSON must be a list of turn objects")

    turns: list[HistoricalTurn] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"history item {index} must be an object")

        turn_id = item.get("turn_id")
        user_request = item.get("user_request")
        assistant_result = item.get("assistant_result", "")
        execution_records = item.get("execution_records", [])
        timestamp = item.get("timestamp")

        if isinstance(turn_id, bool) or not isinstance(turn_id, int):
            raise ValueError(f"history item {index}.turn_id must be an integer")
        if not isinstance(user_request, str) or not user_request.strip():
            raise ValueError(
                f"history item {index}.user_request must be a non-empty string"
            )
        if not isinstance(assistant_result, str):
            raise ValueError(f"history item {index}.assistant_result must be a string")
        if not isinstance(execution_records, list) or not all(
            isinstance(record, str) for record in execution_records
        ):
            raise ValueError(
                f"history item {index}.execution_records must be a list of strings"
            )
        if timestamp is not None and not isinstance(timestamp, str):
            raise ValueError(f"history item {index}.timestamp must be a string or null")

        turns.append(
            HistoricalTurn(
                turn_id=turn_id,
                user_request=user_request,
                assistant_result=assistant_result,
                execution_records=tuple(execution_records),
                timestamp=timestamp,
            )
        )

    if len({turn.turn_id for turn in turns}) != len(turns):
        raise ValueError("turn_id values must be unique")
    return turns


def result_metrics(result: RunResult) -> dict[str, Any]:
    """Build a JSON-serializable audit summary for the local demo."""

    total_usage: dict[str, int] = {}
    calls: list[dict[str, Any]] = []
    for call in result.calls:
        usage = dict(call.usage)
        for key, value in usage.items():
            total_usage[key] = total_usage.get(key, 0) + value
        calls.append(
            {
                "index": call.index,
                "phase": call.phase,
                "append_only_messages": call.append_only_messages,
                "usage": usage,
            }
        )
    return {
        "model_calls": len(result.calls),
        "fallback_used": result.fallback_used,
        "total_usage": total_usage,
        "calls": calls,
    }
