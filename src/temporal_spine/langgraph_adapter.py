from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
    raise ModuleNotFoundError(
        "LangGraph integration requires: python -m pip install -e '.[agent]'"
    ) from exc

from .memory import ComponentStore, HistoricalTurn
from .model import ChatModel, ToolCall
from .prompt import TOOLS, base_messages
from .runner import (
    CallAudit,
    ProtocolError,
    RunConfig,
    RunResult,
    _canonical_hash,
    _tool_message,
)


class _GraphState(TypedDict):
    messages: list[dict[str, Any]]
    previous_messages: list[dict[str, Any]] | None
    loaded_action_ids: list[str]
    fallback_used: bool
    calls: list[CallAudit]
    trace: list[Mapping[str, Any]]
    pending_call: ToolCall | None
    load_rounds: int
    loading_closed: bool
    answer: str


class LangGraphTemporalSpine:
    """LangGraph loop for post-compression Temporal-Spine memory access."""

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
        recent_messages: Sequence[Mapping[str, Any]] = (),
    ) -> RunResult:
        store = ComponentStore(turns, self.config.spine_policy)
        initial_messages = base_messages(
            store,
            current_request,
            current_date,
            recent_messages=recent_messages,
        )
        fixed_hash = _canonical_hash(
            {
                "initial_messages": initial_messages,
                "tools": self._tools,
                "model_parameters": dict(self.model.fixed_parameters),
            }
        )

        def model_node(state: _GraphState) -> dict[str, Any]:
            messages = state["messages"]
            previous = state["previous_messages"]
            append_only = previous is None or (
                len(messages) >= len(previous) and messages[: len(previous)] == previous
            )
            if not append_only:
                raise ProtocolError("messages are not append-only")
            response = self.model.complete(
                copy.deepcopy(messages),
                copy.deepcopy(self._tools),
            )
            phase = "forced_answer" if state["loading_closed"] else "agent"
            calls = state["calls"] + [
                CallAudit(
                    index=len(state["calls"]),
                    phase=phase,
                    fixed_request_hash=fixed_hash,
                    append_only_messages=append_only,
                    usage=dict(response.usage),
                )
            ]
            updates: dict[str, Any] = {
                "previous_messages": copy.deepcopy(messages),
                "calls": calls,
            }
            if response.tool_calls:
                if state["loading_closed"]:
                    raise ProtocolError(
                        "model requested memory after the loading budget was closed"
                    )
                if len(response.tool_calls) != 1:
                    raise ProtocolError("memory access requires exactly one tool call")
                call = response.tool_calls[0]
                if call.name != "load_components":
                    raise ProtocolError(f"unknown memory tool: {call.name}")
                updates["messages"] = messages + [response.as_assistant_message()]
                updates["pending_call"] = call
                return updates

            answer = response.content.strip()
            if not answer:
                raise ProtocolError("model returned neither a memory call nor an answer")
            updates["answer"] = answer
            updates["trace"] = state["trace"] + [
                {"event": "answer_returned", "phase": phase}
            ]
            return updates

        def memory_node(state: _GraphState) -> dict[str, Any]:
            call = state["pending_call"]
            if call is None:
                raise ProtocolError("memory node requires a pending tool call")
            selection = call.arguments.get("selection")
            raw_ids = call.arguments.get("action_ids")
            if not isinstance(raw_ids, list) or not all(
                isinstance(action_id, str) for action_id in raw_ids
            ):
                raise ProtocolError("action_ids must be a list of strings")

            loaded = list(state["loaded_action_ids"])
            trace = list(state["trace"])
            fallback_used = state["fallback_used"]
            if selection == "selected":
                if not raw_ids:
                    raise ProtocolError("selected loading requires at least one action ID")
                requested = raw_ids
            elif selection == "remaining":
                if raw_ids:
                    raise ProtocolError("remaining loading requires an empty action_ids list")
                if not self.config.full_fallback_enabled:
                    return {
                        "messages": state["messages"]
                        + [_tool_message(call, "remaining loading is disabled")],
                        "pending_call": None,
                        "load_rounds": state["load_rounds"] + 1,
                        "trace": trace + [{"event": "remaining_rejected"}],
                    }
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
            trace.append({"event": "components_loaded", "action_ids": batch_ids})
            return {
                "messages": state["messages"]
                + [
                    _tool_message(
                        call,
                        store.render_components(batch_ids) or "(no remaining components)",
                    )
                ],
                "loaded_action_ids": loaded,
                "fallback_used": fallback_used,
                "pending_call": None,
                "load_rounds": state["load_rounds"] + 1,
                "trace": trace,
            }

        def close_memory_node(state: _GraphState) -> dict[str, Any]:
            loaded = list(state["loaded_action_ids"])
            messages = list(state["messages"])
            trace = list(state["trace"])
            fallback_used = state["fallback_used"]
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
            return {
                "messages": messages,
                "loaded_action_ids": loaded,
                "fallback_used": fallback_used,
                "trace": trace,
                "loading_closed": True,
            }

        def route_after_model(state: _GraphState) -> str:
            return "done" if state["answer"] else "memory"

        def route_after_memory(state: _GraphState) -> str:
            if state["load_rounds"] >= self.config.max_load_rounds:
                return "close_memory"
            return "model"

        builder = StateGraph(_GraphState)
        builder.add_node("model", model_node)
        builder.add_node("memory", memory_node)
        builder.add_node("close_memory", close_memory_node)
        builder.add_edge(START, "model")
        builder.add_conditional_edges(
            "model",
            route_after_model,
            {"memory": "memory", "done": END},
        )
        builder.add_conditional_edges(
            "memory",
            route_after_memory,
            {"model": "model", "close_memory": "close_memory"},
        )
        builder.add_edge("close_memory", "model")
        graph = builder.compile()

        state = graph.invoke(
            {
                "messages": initial_messages,
                "previous_messages": None,
                "loaded_action_ids": [],
                "fallback_used": False,
                "calls": [],
                "trace": [],
                "pending_call": None,
                "load_rounds": 0,
                "loading_closed": False,
                "answer": "",
            },
            {"recursion_limit": self.config.max_load_rounds * 3 + 6},
        )
        return RunResult(
            answer=state["answer"],
            loaded_action_ids=tuple(state["loaded_action_ids"]),
            fallback_used=state["fallback_used"],
            calls=tuple(state["calls"]),
            trace=tuple(state["trace"]),
        )
