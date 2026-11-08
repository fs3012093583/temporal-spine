from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .memory import ComponentStore, HistoricalTurn, SpinePolicy
from .model import ChatModel, ModelResponse, ToolCall
from .prompt import (
    DEFAULT_COMPONENT_BATCH_LIMIT,
    TOOLS,
    base_messages,
    history_access_state,
)


class ProtocolError(RuntimeError):
    """The model or runtime violated the fixed Temporal-Spine protocol."""


@dataclass(frozen=True)
class RunConfig:
    """Bound valid access decisions separately from total model calls."""

    max_components_per_load: int = DEFAULT_COMPONENT_BATCH_LIMIT
    max_invalid_actions: int = 1
    capture_natural_final: bool = True
    spine_policy: SpinePolicy = field(default_factory=SpinePolicy)
    max_access_calls: int = 1
    max_model_calls: int = 3

    def __post_init__(self) -> None:
        for name in ("max_components_per_load", "max_invalid_actions", "max_access_calls", "max_model_calls"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.max_components_per_load <= DEFAULT_COMPONENT_BATCH_LIMIT:
            raise ValueError(
                f"max_components_per_load must be between 1 and "
                f"{DEFAULT_COMPONENT_BATCH_LIMIT}"
            )
        if self.max_invalid_actions not in (0, 1):
            raise ValueError("max_invalid_actions must be 0 or 1")
        if not 0 <= self.max_access_calls <= 2:
            raise ValueError("max_access_calls must be 0, 1, or 2")
        if self.max_model_calls < self.max_access_calls + 1:
            raise ValueError("reserve at least one model call for final submission")


@dataclass(frozen=True)
class CallAudit:
    index: int
    phase: str
    fixed_request_hash: str
    append_only_messages: bool
    message_count: int
    usage: Mapping[str, int]


@dataclass(frozen=True)
class RunResult:
    answer: str
    loaded_action_ids: tuple[str, ...]
    dropped_action_ids: tuple[str, ...]
    invalid_action_count: int
    protocol_followed: bool
    submission_mode: str
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
    """MemAddr with K=1 (default) or K=2 append-only, deduplicated access.

    Invalid-action recovery shares the total model-call limit; it never adds access budget.
    There is no summarizer, relevance retriever, focus action, or Full History fallback.
    """

    def __init__(self, model: ChatModel, config: RunConfig | None = None) -> None:
        self.model = model
        self.config = config or RunConfig()
        self._tools = copy.deepcopy(TOOLS)
        load_schema = self._tools[0]["function"]["parameters"]["properties"][
            "action_ids"
        ]
        load_schema["maxItems"] = self.config.max_components_per_load

    def run(
        self,
        turns: Sequence[HistoricalTurn],
        current_request: str,
        *,
        current_date: str | None = None,
    ) -> RunResult:
        if not isinstance(current_request, str) or not current_request.strip():
            raise ValueError("current_request must be a non-empty string")
        store = ComponentStore(turns, self.config.spine_policy)
        messages = base_messages(
            store,
            current_request,
            current_date,
            component_batch_limit=self.config.max_components_per_load,
            max_access_calls=self.config.max_access_calls,
            max_model_calls=self.config.max_model_calls,
        )
        fixed_hash = _canonical_hash(
            {
                "system": messages[0],
                "tools": self._tools,
                "model_parameters": dict(self.model.fixed_parameters),
            }
        )
        previous_messages: list[dict[str, Any]] | None = None
        loaded: list[str] = []
        dropped: list[str] = []
        calls: list[CallAudit] = []
        trace: list[Mapping[str, Any]] = []

        def request(phase: str) -> ModelResponse:
            nonlocal previous_messages
            current_hash = _canonical_hash({
                "system": messages[0], "tools": self._tools,
                "model_parameters": dict(self.model.fixed_parameters),
            })
            if current_hash != fixed_hash:
                raise ProtocolError("fixed model parameters changed during the trajectory")
            append_only = previous_messages is None or (
                len(messages) >= len(previous_messages)
                and messages[: len(previous_messages)] == previous_messages
            )
            if not append_only:
                raise ProtocolError("messages are not append-only")
            response = self.model.complete(
                copy.deepcopy(messages), copy.deepcopy(self._tools)
            )
            if response.finish_reason in {"length", "content_filter"}:
                raise ProtocolError(
                    f"model output is incomplete ({response.finish_reason}); "
                    "increase the output budget or inspect provider settings"
                )
            calls.append(
                CallAudit(
                    index=len(calls),
                    phase=phase,
                    fixed_request_hash=fixed_hash,
                    append_only_messages=append_only,
                    message_count=len(messages),
                    usage=dict(response.usage),
                )
            )
            previous_messages = copy.deepcopy(messages)
            return response

        history_access_calls = 0
        invalid_actions = 0
        def access_state(**extra: Any) -> dict[str, Any]:
            return history_access_state(
                model_calls_used=len(calls),
                history_access_calls_used=history_access_calls,
                loaded_components_total=len(loaded),
                component_batch_limit=self.config.max_components_per_load,
                max_access_calls=self.config.max_access_calls,
                max_model_calls=self.config.max_model_calls,
                **extra,
            )

        for _ in range(self.config.max_model_calls):
            phase = (
                "access_or_submit"
                if not access_state()["history_access_closed"]
                else "final_submit"
            )
            if calls:
                phase += "_retry" if trace and trace[-1]["event"] == "action_rejected" else ""
            response = request(phase)
            try:
                resolved_final = self._resolve_final(response)
            except ProtocolError as exc:
                resolved_final = None
                rejection_reason = str(exc)
            else:
                rejection_reason = ""

            if resolved_final is not None:
                answer, submitted_with_tool, mode = resolved_final
                protocol_followed = submitted_with_tool and invalid_actions == 0
                trace.append(
                    {
                        "event": "final_result_submitted",
                        "model_call": len(calls),
                        "submission_mode": mode,
                        "protocol_followed": protocol_followed,
                    }
                )
                return RunResult(
                    answer=answer,
                    loaded_action_ids=tuple(loaded),
                    dropped_action_ids=tuple(dropped),
                    invalid_action_count=invalid_actions,
                    protocol_followed=protocol_followed,
                    submission_mode=mode,
                    calls=tuple(calls),
                    trace=tuple(trace),
                )

            load_call = response.tool_calls[0] if len(response.tool_calls) == 1 else None
            if not rejection_reason:
                if load_call is None:
                    rejection_reason = "call exactly one currently authorized function"
                elif load_call.name != "load_components":
                    rejection_reason = f"unauthorized tool: {load_call.name}"
                elif history_access_calls >= self.config.max_access_calls or len(calls) >= self.config.max_model_calls:
                    rejection_reason = "history access is closed; submit the final result"
                elif not isinstance(load_call.arguments, Mapping) or set(load_call.arguments) != {"selection", "action_ids"}:
                    rejection_reason = "load_components requires only selection and action_ids"
                else:
                    selection = load_call.arguments.get("selection")
                    requested = load_call.arguments.get("action_ids")
                    if selection != "selected":
                        rejection_reason = "load_components selection must be selected"
                    elif not isinstance(requested, list) or not requested or not all(
                        isinstance(action_id, str) for action_id in requested
                    ):
                        rejection_reason = (
                            "action_ids must be a non-empty list of strings"
                        )
                    else:
                        retained = requested[: self.config.max_components_per_load]
                        overflow = requested[self.config.max_components_per_load :]
                        dropped.extend(overflow)
                        try:
                            if len(retained) != len(set(retained)):
                                raise ValueError("retained action_ids must be distinct")
                            batch_ids = store.chronological(retained)
                        except ValueError as exc:
                            rejection_reason = str(exc)
                        else:
                            repeated = [action_id for action_id in batch_ids if action_id in loaded]
                            batch_ids = [action_id for action_id in batch_ids if action_id not in loaded]
                            loaded.extend(batch_ids)
                            history_access_calls += 1
                            state = access_state(
                                last_requested_component_count=len(requested),
                                last_loaded_component_count=len(batch_ids),
                                request_truncated=bool(overflow),
                            )
                            tool_payload: dict[str, Any] = {
                                "event": "loaded_components",
                                "status": "success",
                                "batch_index": history_access_calls,
                                "scope": "selected",
                                "already_loaded_action_ids": repeated,
                                "components": [
                                    store.loaded_component_payload(action_id)
                                    for action_id in batch_ids
                                ],
                            }
                            if overflow:
                                tool_payload["load_limit_notice"] = {
                                    "status": "request_truncated",
                                    "requested_count": len(requested),
                                    "maximum_per_load": (
                                        self.config.max_components_per_load
                                    ),
                                    "retained_requested_action_ids": retained,
                                    "dropped_action_ids": overflow,
                                    "message": (
                                        "Only the first requested action IDs up to the "
                                        "per-load limit were processed; the remaining IDs "
                                        "were not loaded."
                                    ),
                                }
                            tool_payload["history_access_state"] = state
                            messages.append(response.as_assistant_message())
                            messages.append(
                                _tool_message(
                                    load_call,
                                    json.dumps(
                                        tool_payload, ensure_ascii=False, indent=2
                                    ),
                                )
                            )
                            trace.append(
                                {
                                    "event": "components_loaded",
                                    "requested_action_ids": list(requested),
                                    "loaded_action_ids": list(batch_ids),
                                    "already_loaded_action_ids": repeated,
                                    "dropped_action_ids": list(overflow),
                                    "tool_result": tool_payload,
                                }
                            )
                            continue

            invalid_actions += 1
            trace.append(
                {
                    "event": "action_rejected",
                    "model_call": len(calls),
                    "reason": rejection_reason,
                    "invalid_action_count": invalid_actions,
                }
            )
            if invalid_actions > self.config.max_invalid_actions or len(calls) >= self.config.max_model_calls:
                raise ProtocolError(rejection_reason)
            state = access_state()
            rejection_payload = json.dumps(
                {
                    "event": "protocol_action_rejected",
                    "status": "invalid",
                    "reason": rejection_reason,
                    "history_access_state": state,
                },
                ensure_ascii=False,
                indent=2,
            )
            messages.append(response.as_assistant_message())
            if response.tool_calls:
                for rejected_call in response.tool_calls:
                    messages.append(_tool_message(rejected_call, rejection_payload))
            else:
                messages.append({"role": "user", "content": rejection_payload})

        raise ProtocolError("model call limit exhausted without a final result")

    def _resolve_final(
        self, response: ModelResponse
    ) -> tuple[str, bool, str] | None:
        if len(response.tool_calls) == 1:
            call = response.tool_calls[0]
            if call.name != "submit_final_result":
                return None
            if not isinstance(call.arguments, Mapping) or set(call.arguments) != {"result"}:
                raise ProtocolError("submit_final_result requires only result")
            result = call.arguments.get("result")
            if not isinstance(result, str) or not result.strip():
                raise ProtocolError("submit_final_result requires a non-empty result")
            return result.strip(), True, "native_submit_final_result"
        if response.tool_calls:
            return None
        if response.content.strip() and self.config.capture_natural_final:
            return (
                response.content.strip(),
                False,
                "runtime_captured_natural_result",
            )
        return None
