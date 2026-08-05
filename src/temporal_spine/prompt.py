from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from .memory import ComponentStore


SYSTEM_PROMPT = """You are an agent working after older conversation history has been compressed.
The visible User-Spine preserves historical user requests in chronological order. Historical
assistant results, execution records, and exact originals of unusually long user requests are
source-faithful components that remain hidden until loaded. Each User-Spine turn carries a
loadable_components object that maps available component kinds to their action IDs.

If the currently visible evidence is sufficient, respond normally with the best answer. If exact
historical assistant or execution evidence is needed, call load_components with the smallest likely
sufficient non-repeated set. You may inspect a loaded result and request another batch. Select all
remaining components only when global coverage is genuinely required. Later turns may update
earlier facts. Do not conclude that information is unavailable merely because a relevant component
is still hidden. Never expose internal action IDs or memory-management details in the final answer."""


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "load_components",
            "description": "Load selected historical components, or every still-unloaded component.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection": {"type": "string", "enum": ["selected", "remaining"]},
                    "action_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["selection", "action_ids"],
                "additionalProperties": False,
            },
        },
    },
]


def base_messages(
    store: ComponentStore,
    current_request: str,
    current_date: str | None,
    recent_messages: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    date = current_date or "(not supplied)"
    content = json.dumps(
        {
            "context_type": "post_compression_memory",
            "current_date": date,
            "user_spine": store.spine_payload(),
        },
        ensure_ascii=False,
        indent=2,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    for index, message in enumerate(recent_messages):
        role = message.get("role")
        message_content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(message_content, str):
            raise ValueError(
                f"recent message {index} must contain a user/assistant role and string content"
            )
        messages.append(copy.deepcopy(dict(message)))
    messages.append(
        {"role": "user", "content": f"[CURRENT REQUEST]\n{current_request}"}
    )
    return messages
