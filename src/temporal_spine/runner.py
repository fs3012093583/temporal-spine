from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .memory import ComponentStore, HistoricalTurn, SpinePolicy
from .model import ChatModel, ModelResponse, ToolCall
from .prompt import TOOLS, base_messages


class ProtocolError(RuntimeError):
    """The model or runtime violated the Temporal-Spine memory protocol."""


@dataclass(frozen=True)
class RunConfig:
    max_load_rounds: int = 1
    full_fallback_enabled: bool = True
    spine_policy: SpinePolicy = field(default_factory=SpinePolicy)

    def __post_init__(self) -> None:
        if self.max_load_rounds <= 0:
            raise ValueError("max_load_rounds must be positive")


@dataclass(frozen=True)
class CallAudit:
    index: int
    phase: str
    fixed_request_hash: str
    append_only_messages: bool
    usage: Mapping[str, int]


@dataclass(frozen=True)
class RunResult:
    answer: str
    loaded_action_ids: tuple[str, ...]
    fallback_used: bool
    calls: tuple[CallAudit, ...]
    trace: tuple[Mapping[str, Any], ...]


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tool_message(call: ToolCall, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.call_id,
        "name": call.name,
        "content": content,
    }


class TemporalSpine:
    """Post-compression memory loop with natural assistant completion."""

    def __init__(self, model: ChatModel, config: RunConfig | None = None) -> None:
        self.model = model
        self.config = config or RunConfig()
        self._tools = copy.deepcopy(TOOLS)

    def run(
        self,
        turns: Sequence[HistoricalTurn],
        current_request: str,
        *,
        current_date: str | None = None,
    ) -> RunResult:
        store = ComponentStore(turns, self.config.spine_policy)
        messages = base_messages(store, current_request, current_date)
        fixed_hash = _canonical_hash(
            {
                "initial_messages": messages,
                "tools": self._tools,
                "model_parameters": dict(self.model.fixed_parameters),
            }
        )
        previous_messages: list[dict[str, Any]] | None = None
        loaded: list[str] = []
        fallback_used = False
        calls: list[CallAudit] = []
        trace: list[Mapping[str, Any]] = []

        def request(phase: str) -> ModelResponse:
            nonlocal previous_messages
            append_only = previous_messages is None or (
                len(messages) >= len(previous_messages)
                and messages[: len(previous_messages)] == previous_messages
            )
            if not append_only:
                raise ProtocolError("messages are not append-only")
            response = self.model.complete(copy.deepcopy(messages), copy.deepcopy(self._tools))
            calls.append(
                CallAudit(
                    index=len(calls),
                    phase=phase,
                    fixed_request_hash=fixed_hash,
                    append_only_messages=append_only,
                    usage=dict(response.usage),
                )
            )
            previous_messages = copy.deepcopy(messages)
            return response

        def finish(response: ModelResponse, phase: str) -> RunResult:
            answer = response.content.strip()
            if response.tool_calls or not answer:
                raise ProtocolError(f"{phase} requires a non-empty ordinary assistant response")
            trace.append({"event": "answer_returned", "phase": phase})
            return RunResult(
                answer=answer,
                loaded_action_ids=tuple(loaded),
                fallback_used=fallback_used,
                calls=tuple(calls),
                trace=tuple(trace),
            )

        for load_round in range(self.config.max_load_rounds + 1):
            response = request("agent")
            if not response.tool_calls:
                return finish(response, "agent")
            if load_round == self.config.max_load_rounds:
                raise ProtocolError("model requested memory after the loading budget was closed")
            if len(response.tool_calls) != 1:
                raise ProtocolError("memory access requires exactly one tool call")
            call = response.tool_calls[0]
            if call.name != "load_components":
                raise ProtocolError(f"unknown memory tool: {call.name}")
            messages.append(response.as_assistant_message())

            selection = call.arguments.get("selection")
            raw_ids = call.arguments.get("action_ids")
            if not isinstance(raw_ids, list) or not all(
                isinstance(action_id, str) for action_id in raw_ids
            ):
                raise ProtocolError("action_ids must be a list of strings")
            if selection == "selected":
                if not raw_ids:
                    raise ProtocolError("selected loading requires at least one action ID")
                requested = raw_ids
            elif selection == "remaining":
                if raw_ids:
                    raise ProtocolError("remaining loading requires an empty action_ids list")
                if not self.config.full_fallback_enabled:
                    messages.append(_tool_message(call, "remaining loading is disabled"))
                    trace.append({"event": "remaining_rejected"})
                    continue
                requested = [
                    action_id for action_id in store.action_ids if action_id not in loaded
                ]
                fallback_used = True
            else:
                raise ProtocolError("selection must be selected or remaining")

            repeated = [action_id for action_id in requested if action_id in loaded]
            if repeated:
                raise ProtocolError(f"components already loaded: {repeated}")
            try:
                batch_ids = store.chronological(requested)
            except ValueError as exc:
                raise ProtocolError(str(exc)) from exc
            loaded.extend(batch_ids)
            result = store.render_components(batch_ids) or "(no remaining components)"
            messages.append(_tool_message(call, result))
            trace.append({"event": "components_loaded", "action_ids": batch_ids})

            if load_round + 1 < self.config.max_load_rounds:
                continue

            if self.config.full_fallback_enabled:
                remaining = [
                    action_id for action_id in store.action_ids if action_id not in loaded
                ]
                if remaining:
                    batch_ids = store.chronological(remaining)
                    loaded.extend(batch_ids)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[RUNTIME EVENT]\nmemory_load_budget_exhausted; "
                                "automatic_remaining_components_loaded\n\n"
                                + store.render_components(batch_ids)
                            ),
                        }
                    )
                    trace.append(
                        {"event": "components_loaded", "action_ids": batch_ids, "runtime": True}
                    )
                    fallback_used = True
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[RUNTIME EVENT]\nmemory_loading_closed; "
                        "answer using the currently visible evidence"
                    ),
                }
            )
            return finish(request("forced_answer"), "forced_answer")

        raise ProtocolError("memory loop terminated without an answer")
