from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


_TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)


@dataclass(frozen=True)
class HistoricalTurn:
    """One historical user boundary and everything produced before the next one."""

    turn_id: int
    user_request: str
    assistant_result: str = ""
    execution_records: tuple[str, ...] = ()
    timestamp: str | None = None


@dataclass(frozen=True)
class SpinePolicy:
    """Query-independent policy for unusually long historical user requests."""

    max_user_tokens: int = 512
    head_tokens: int = 256
    tail_tokens: int = 256

    def __post_init__(self) -> None:
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
    marker = f"\n... <{omitted} approximate tokens omitted; original is loadable> ...\n"
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

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(self.components)

    def catalog(self) -> list[dict[str, object]]:
        return [
            {
                "action_id": component.action_id,
                "kind": component.kind.value,
                "turn_id": component.turn_id,
            }
            for component in self.components.values()
        ]

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

    def spine_payload(self) -> list[dict[str, object]]:
        """Return the always-visible user history as structured JSON-ready data."""

        rows: list[dict[str, object]] = []
        for turn in self.spine_turns:
            row: dict[str, object] = {
                "turn_id": turn.turn_id,
                "user_request": turn.visible_user_request,
            }
            if turn.timestamp:
                row["timestamp"] = turn.timestamp
            loadable_components: dict[str, str] = {}
            if turn.user_original_action_id:
                loadable_components["user_request_original"] = (
                    turn.user_original_action_id
                )
            if turn.result_action_id:
                loadable_components["assistant_result"] = turn.result_action_id
            if turn.trace_action_id:
                loadable_components["execution_records"] = turn.trace_action_id
            if loadable_components:
                row["loadable_components"] = loadable_components
            rows.append(row)
        return rows

    def render_spine(self) -> str:
        return json.dumps(
            self.spine_payload(),
            ensure_ascii=False,
            indent=2,
        )

    def render_components(self, action_ids: Iterable[str]) -> str:
        rows: list[str] = []
        for action_id in self.chronological(action_ids):
            component = self.components[action_id]
            rows.append(
                f'<memory-component action_id="{action_id}" '
                f'kind="{component.kind.value}" turn_id="{component.turn_id}">\n'
                f"{component.content}\n"
                "</memory-component>"
            )
        return "\n\n".join(rows)
