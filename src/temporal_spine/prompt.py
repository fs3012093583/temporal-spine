from __future__ import annotations

import json
from typing import Any

from .memory import ComponentStore


SYSTEM_PROMPT = """You are completing the user's current task with access to a structured long-term
conversation memory. The visible User-Spine preserves historical user requests in chronological
order. Historical assistant results, execution records, and exact originals of unusually long user
requests are source-faithful components that remain hidden until loaded.

Before the runtime appends a final_result_authorized event, call exactly one provided function and
do not answer in ordinary text. Load the smallest likely sufficient set of components. You may load
another non-repeated batch after inspecting a result. Select all remaining components only when
global coverage is genuinely required. Later turns may update earlier facts. Do not conclude that
information is unavailable merely because a relevant assistant result is still hidden.

After final_result_authorized, memory functions are revoked by the runtime. Use the visible history
to produce the best correct and complete result for the current request, then call
submit_final_result exactly once. Never expose internal action IDs or memory-management details in
the submitted result."""


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "load_components",
            "description": "Load selected components, or every still-unloaded component.",
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
    {
        "type": "function",
        "function": {
            "name": "focus_user_turns",
            "description": "Copy exact already-visible user requests to the append-only tail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "turn_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["turn_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_final_result",
            "description": (
                "Declare that the visible evidence is sufficient, optionally copying a small "
                "set of already-visible user requests to the append-only tail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus_turn_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final_result",
            "description": "Submit the final user-facing result after runtime authorization.",
            "parameters": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
        },
    },
]


def base_messages(
    store: ComponentStore,
    current_request: str,
    current_date: str | None,
) -> list[dict[str, Any]]:
    catalog = json.dumps(store.catalog(), ensure_ascii=False, separators=(",", ":"))
    date = current_date or "(not supplied)"
    content = f"""[CURRENT DATE]
{date}

[USER-SPINE]
{store.render_spine()}

[LOADABLE COMPONENT CATALOG]
{catalog}

[CURRENT REQUEST]
{current_request}

[RUNTIME EVENT]
control_phase_started"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
