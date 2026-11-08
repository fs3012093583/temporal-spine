from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence


_TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)


@dataclass(frozen=True)
class HistoricalTurn:
    """One historical user boundary and everything produced before the next one."""

    turn_id: int
    user_request: str
    assistant_result: str = ""
    execution_records: tuple[str, ...] = ()
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if type(self.turn_id) is not int or self.turn_id < 0:
            raise ValueError("turn_id must be a non-negative integer")
        if not isinstance(self.user_request, str) or not isinstance(self.assistant_result, str):
            raise ValueError("user_request and assistant_result must be strings")
        if not isinstance(self.execution_records, (tuple, list)) or not all(
            isinstance(record, str) for record in self.execution_records
        ):
            raise ValueError("execution_records must be a sequence of strings")
        object.__setattr__(self, "execution_records", tuple(self.execution_records))
        if self.timestamp is not None and not isinstance(self.timestamp, str):
            raise ValueError("timestamp must be a string or None")


@dataclass(frozen=True)
class SpinePolicy:
    """Query-independent preview limits in deterministic spans, not API tokens.

    The ``*_tokens`` names are retained for compatibility with the original release.
    """

    max_user_tokens: int = 512
    head_tokens: int = 256
    tail_tokens: int = 256

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (
            self.max_user_tokens, self.head_tokens, self.tail_tokens
        )):
            raise ValueError("preview limits must be integers")
        if self.max_user_tokens <= 0:
            raise ValueError("max_user_tokens must be positive")
        if self.head_tokens < 0 or self.tail_tokens < 0:
            raise ValueError("head_tokens and tail_tokens must be non-negative")
        if self.head_tokens + self.tail_tokens > self.max_user_tokens:
            raise ValueError("head_tokens + tail_tokens cannot exceed max_user_tokens")
        if self.head_tokens + self.tail_tokens == 0:
            raise ValueError("the spine must retain at least one token")


class ComponentKind(str, Enum):
    USER_ORIGINAL = "user_request_original"
    ASSISTANT_RESULT = "assistant_result"
    EXECUTION_RECORDS = "execution_records"


@dataclass(frozen=True)
class Component:
    action_id: str
    kind: ComponentKind
    turn_id: int
    content: str


@dataclass(frozen=True)
class SpineTurn:
    turn_id: int
    visible_user_request: str
    timestamp: str | None
    user_original_action_id: str | None
    result_action_id: str | None
    trace_action_id: str | None


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in _TOKEN_RE.finditer(text)]


def head_tail_preview(text: str, policy: SpinePolicy) -> tuple[str, bool]:
    """Return a deterministic preview without consulting the current query."""

    spans = _token_spans(text)
    if len(spans) <= policy.max_user_tokens:
        return text, False
    head_end = spans[policy.head_tokens - 1][1] if policy.head_tokens else 0
    tail_start = spans[-policy.tail_tokens][0] if policy.tail_tokens else len(text)
    omitted = len(spans) - policy.head_tokens - policy.tail_tokens
    marker = f"\n... <{omitted} spans omitted; original is loadable> ...\n"
    return text[:head_end] + marker + text[tail_start:], True


class ComponentStore:
    """Immutable source-faithful history and its always-visible User-Spine."""

    def __init__(
        self,
        turns: Sequence[HistoricalTurn],
        policy: SpinePolicy | None = None,
    ) -> None:
        self.policy = policy or SpinePolicy()
        self.turns = tuple(sorted(turns, key=lambda turn: turn.turn_id))
        turn_ids = [turn.turn_id for turn in self.turns]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("turn_id values must be unique")

        self.components: dict[str, Component] = {}
        spine: list[SpineTurn] = []
        for turn in self.turns:
            preview, truncated = head_tail_preview(turn.user_request, self.policy)
            user_id = f"user:{turn.turn_id}" if truncated else None
            result_id = f"result:{turn.turn_id}" if turn.assistant_result else None
            trace_id = f"trace:{turn.turn_id}" if turn.execution_records else None
            if user_id:
                self.components[user_id] = Component(
                    user_id, ComponentKind.USER_ORIGINAL, turn.turn_id, turn.user_request
                )
            if result_id:
                self.components[result_id] = Component(
                    result_id,
                    ComponentKind.ASSISTANT_RESULT,
                    turn.turn_id,
                    turn.assistant_result,
                )
            if trace_id:
                trace_text = "\n\n".join(
                    f"[event {index}]\n{record}"
                    for index, record in enumerate(turn.execution_records)
                )
                self.components[trace_id] = Component(
                    trace_id, ComponentKind.EXECUTION_RECORDS, turn.turn_id, trace_text
                )
            spine.append(
                SpineTurn(
                    turn_id=turn.turn_id,
                    visible_user_request=preview,
                    timestamp=turn.timestamp,
                    user_original_action_id=user_id,
                    result_action_id=result_id,
                    trace_action_id=trace_id,
                )
            )
        self.spine_turns = tuple(spine)
        self._spine_by_id = {turn.turn_id: turn for turn in self.spine_turns}

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(self.components)

    def catalog(self) -> list[dict[str, object]]:
        return [
            {
                "action_id": component.action_id,
                "historical_turn_id": component.turn_id,
                "component_kind": component.kind.value,
                "source_role": self.source_role(component),
            }
            for component in self.components.values()
        ]

    @staticmethod
    def source_role(component: Component) -> str:
        if component.kind is ComponentKind.USER_ORIGINAL:
            return "user"
        if component.kind is ComponentKind.ASSISTANT_RESULT:
            return "assistant"
        return "observable_execution"

    def chronological(self, action_ids: Iterable[str]) -> list[str]:
        unique = list(dict.fromkeys(action_ids))
        missing = [action_id for action_id in unique if action_id not in self.components]
        if missing:
            raise ValueError(f"unknown action IDs: {missing}")
        kind_order = {
            ComponentKind.USER_ORIGINAL: 0,
            ComponentKind.ASSISTANT_RESULT: 1,
            ComponentKind.EXECUTION_RECORDS: 2,
        }
        return sorted(
            unique,
            key=lambda action_id: (
                self.components[action_id].turn_id,
                kind_order[self.components[action_id].kind],
            ),
        )

    def render_spine(self) -> str:
        rows: list[str] = []
        for turn in self.spine_turns:
            metadata = [f"turn_id={turn.turn_id}"]
            if turn.timestamp:
                metadata.append(f"timestamp={turn.timestamp}")
            if turn.user_original_action_id:
                metadata.append(f"user_original_action_id={turn.user_original_action_id}")
            if turn.result_action_id:
                metadata.append(f"result_action_id={turn.result_action_id}")
            if turn.trace_action_id:
                metadata.append(f"trace_action_id={turn.trace_action_id}")
            rows.append(
                "<user-turn " + " ".join(metadata) + ">\n"
                + turn.visible_user_request
                + "\n</user-turn>"
            )
        return "\n\n".join(rows) if rows else "(empty history)"

    def loaded_component_payload(self, action_id: str) -> dict[str, Any]:
        """Materialize one component with explicit, source-faithful provenance."""

        if action_id not in self.components:
            raise ValueError(f"unknown action ID: {action_id}")
        component = self.components[action_id]
        row: dict[str, Any] = {
            "action_id": action_id,
            "historical_turn_id": component.turn_id,
            "component_kind": component.kind.value,
            "source_role": self.source_role(component),
        }
        if component.kind in {
            ComponentKind.ASSISTANT_RESULT,
            ComponentKind.EXECUTION_RECORDS,
        }:
            spine_turn = self._spine_by_id[component.turn_id]
            paired_request: dict[str, Any] = {
                "source_role": "user",
                "completeness": (
                    "head_tail_preview"
                    if spine_turn.user_original_action_id is not None
                    else "full"
                ),
                "authoritative_content": spine_turn.visible_user_request,
            }
            if spine_turn.user_original_action_id is not None:
                paired_request["user_original_action_id"] = (
                    spine_turn.user_original_action_id
                )
            # Keep the paired user request before the hidden result in serialized JSON.
            row["paired_visible_user_request"] = paired_request
        row["authoritative_content"] = component.content
        return row
