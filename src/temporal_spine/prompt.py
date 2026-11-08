from __future__ import annotations

import json
from typing import Any

from .memory import ComponentStore


PROTOCOL_VERSION = "memaddr-bounded-access-v2-keep-first-no-fallback"
DEFAULT_COMPONENT_BATCH_LIMIT = 5
MODEL_CALL_LIMIT = 3


SYSTEM_PROMPT = """You are a general-purpose assistant with controlled
access to long conversation history. Complete the current request correctly and completely while
accessing as little additional history as is safely possible.

The visible context is append-only. It initially contains a chronological User-Spine and the action
IDs of hidden historical components. Later calls also contain all previously loaded original
components, preceding tool actions, and the latest history_access_state. An action ID is only an
address, not its content.

Before acting, read the latest history_access_state. Call a history-access tool only when the
visible history is insufficient, history_access_closed is false, history_access_calls_remaining is
greater than zero, and that tool is listed in allowed_next_tools. Otherwise, call
submit_final_result with the best result supported by the visible history.

Respect component_batch_limit and never reload an already visible component. Later facts may
update earlier ones; perform the requested reasoning rather than merely copying evidence.

When history access is permitted and
required, call load_components with the smallest likely sufficient set of still-hidden action IDs,
without exceeding component_batch_limit.

Call exactly one currently authorized function and return no ordinary text."""


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "load_components",
            "description": (
                "Load one selected set of hidden history components. Full-history fallback is "
                "disabled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selection": {"type": "string", "enum": ["selected"]},
                    "action_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "uniqueItems": True,
                        "maxItems": DEFAULT_COMPONENT_BATCH_LIMIT,
                    }
                },
                "required": ["selection", "action_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final_result",
            "description": (
                "Submit the complete user-facing result when the currently visible history is "
                "sufficient or further history access is not permitted. This action is terminal."
            ),
            "parameters": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
        },
    },
]


def history_access_state(
    *,
    model_calls_used: int,
    history_access_calls_used: int,
    loaded_components_total: int,
    component_batch_limit: int,
    max_access_calls: int = 1,
    max_model_calls: int = MODEL_CALL_LIMIT,
    last_requested_component_count: int | None = None,
    last_loaded_component_count: int | None = None,
    request_truncated: bool | None = None,
) -> dict[str, Any]:
    """Return the authoritative state appended before the next model call."""

    access_remaining = max(0, max_access_calls - history_access_calls_used)
    model_calls_remaining = max(0, max_model_calls - model_calls_used)
    access_remaining = min(access_remaining, max(0, model_calls_remaining - 1))
    access_closed = access_remaining == 0
    state: dict[str, Any] = {
        "event": "history_access_state",
        "access_policy": "one_shot" if max_access_calls == 1 else "bounded",
        "history_access_tool": "load_components",
        "model_call_limit": max_model_calls,
        "model_calls_used": model_calls_used,
        "model_calls_remaining": model_calls_remaining,
        "history_access_call_limit": max_access_calls,
        "history_access_calls_used": history_access_calls_used,
        "history_access_calls_remaining": access_remaining,
        "history_access_closed": access_closed,
        "component_batch_limit": component_batch_limit,
        "loaded_batches": history_access_calls_used,
        "loaded_components_total": loaded_components_total,
        "allowed_next_tools": (
            ["submit_final_result"]
            if access_closed
            else ["load_components", "submit_final_result"]
        ),
    }
    if last_requested_component_count is not None:
        state["last_requested_component_count"] = last_requested_component_count
    if last_loaded_component_count is not None:
        state["last_loaded_component_count"] = last_loaded_component_count
    if request_truncated is not None:
        state["request_truncated"] = request_truncated
    return state


def base_messages(
    store: ComponentStore,
    current_request: str,
    current_date: str | None,
    *,
    component_batch_limit: int,
    max_access_calls: int = 1,
    max_model_calls: int = MODEL_CALL_LIMIT,
) -> list[dict[str, Any]]:
    initial_state = history_access_state(
        model_calls_used=0,
        history_access_calls_used=0,
        loaded_components_total=0,
        component_batch_limit=component_batch_limit,
        max_access_calls=max_access_calls,
        max_model_calls=max_model_calls,
    )
    date = current_date or "(not supplied)"
    content = f"""<runtime_contract>
The latest history_access_state is authoritative for the current access budget, component batch
limit, and allowed tools. A non-empty submit_final_result is terminal; answer length is recorded
for audit but is not a validity condition.
</runtime_contract>

{store.render_spine()}

<current_date>
{date}
</current_date>

<current_request>
{current_request}
</current_request>

<history_access_state>
{json.dumps(initial_state, ensure_ascii=False, indent=2)}
</history_access_state>"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
