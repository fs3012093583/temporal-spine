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
    """The model or runtime violated the fixed Temporal-Spine protocol."""


@dataclass(frozen=True)
class RunConfig:
    max_control_rounds: int = 1
    max_final_retries: int = 2
    full_fallback_enabled: bool = True
    spine_policy: SpinePolicy = field(default_factory=SpinePolicy)

    def __post_init__(self) -> None:
        if self.max_control_rounds <= 0:
            raise ValueError("max_control_rounds must be positive")
        if self.max_final_retries < 0:
            raise ValueError("max_final_retries must be non-negative")


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
    focused_turn_ids: tuple[int, ...]
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
    """Deterministic runtime around an LLM-guided history access policy."""

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
        focused: list[int] = []
        fallback_used = False
        authorized = False
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

        for _round in range(self.config.max_control_rounds):
            response = request("control")
            if len(response.tool_calls) != 1:
                raise ProtocolError("control phase requires exactly one tool call")
            call = response.tool_calls[0]
            if call.name == "submit_final_result":
                raise ProtocolError("submit_final_result is forbidden before authorization")
            messages.append(response.as_assistant_message())

            if call.name == "load_components":
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
                    requested = [action_id for action_id in store.action_ids if action_id not in loaded]
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
                continue

            if call.name == "focus_user_turns":
                raw_turn_ids = call.arguments.get("turn_ids")
                if not isinstance(raw_turn_ids, list) or not all(
                    isinstance(turn_id, int) for turn_id in raw_turn_ids
                ):
                    raise ProtocolError("turn_ids must be a list of integers")
                repeated = [turn_id for turn_id in raw_turn_ids if turn_id in focused]
                if repeated:
                    raise ProtocolError(f"user turns already focused: {repeated}")
                try:
                    result = store.exact_user_requests(raw_turn_ids)
                except ValueError as exc:
                    raise ProtocolError(str(exc)) from exc
                focused.extend(sorted(dict.fromkeys(raw_turn_ids)))
                messages.append(_tool_message(call, result or "(no focused turns)"))
                trace.append({"event": "user_turns_focused", "turn_ids": raw_turn_ids})
                continue

            if call.name == "commit_final_result":
                raw_focus_ids = call.arguments.get("focus_turn_ids", [])
                if not isinstance(raw_focus_ids, list) or not all(
                    isinstance(turn_id, int) for turn_id in raw_focus_ids
                ):
                    raise ProtocolError("focus_turn_ids must be a list of integers")
                repeated = [turn_id for turn_id in raw_focus_ids if turn_id in focused]
                if repeated:
                    raise ProtocolError(f"user turns already focused: {repeated}")
                try:
                    focused_text = store.exact_user_requests(raw_focus_ids)
                except ValueError as exc:
                    raise ProtocolError(str(exc)) from exc
                focused.extend(sorted(dict.fromkeys(raw_focus_ids)))
                authorization = "[RUNTIME EVENT]\nfinal_result_authorized"
                if focused_text:
                    authorization = focused_text + "\n\n" + authorization
                messages.append(_tool_message(call, authorization))
                authorized = True
                trace.append(
                    {
                        "event": "final_result_authorized",
                        "reason": "model_commit",
                        "focus_turn_ids": raw_focus_ids,
                    }
                )
                break

            raise ProtocolError(f"unknown control tool: {call.name}")

        if not authorized:
            if self.config.full_fallback_enabled:
                remaining = [action_id for action_id in store.action_ids if action_id not in loaded]
                if remaining:
                    batch_ids = store.chronological(remaining)
                    loaded.extend(batch_ids)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[RUNTIME EVENT]\ncontrol_budget_exhausted; "
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
                    "content": "[RUNTIME EVENT]\nfinal_result_authorized",
                }
            )
            authorized = True
            trace.append({"event": "final_result_authorized", "reason": "round_limit"})

        for retry in range(self.config.max_final_retries + 1):
            response = request("final")
            if len(response.tool_calls) == 1 and response.tool_calls[0].name == "submit_final_result":
                result = response.tool_calls[0].arguments.get("result")
                if isinstance(result, str) and result.strip():
                    trace.append({"event": "final_result_submitted", "retry": retry})
                    return RunResult(
                        answer=result.strip(),
                        loaded_action_ids=tuple(loaded),
                        focused_turn_ids=tuple(focused),
                        fallback_used=fallback_used,
                        calls=tuple(calls),
                        trace=tuple(trace),
                    )
            messages.append(response.as_assistant_message())
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    rejection = (
                        "submit_final_result requires a non-empty result"
                        if tool_call.name == "submit_final_result"
                        else "memory/control tools are revoked; call submit_final_result"
                    )
                    messages.append(_tool_message(tool_call, rejection))
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[RUNTIME EVENT]\nprotocol_not_followed: call "
                            "submit_final_result with a non-empty result"
                        ),
                    }
                )
            trace.append({"event": "final_protocol_retry", "retry": retry})

        raise ProtocolError("model did not submit a valid final result")
