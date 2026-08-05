"""OpenHands SDK integration for the Temporal-Spine memory runtime.

This module deliberately keeps the OpenHands dependency optional.  The offline
test mode exercises the memory projection and produces the same kind of raw
context files that the live SDK callback writes, while the live adapter adds a
Temporal-Spine condenser and a ``load_components`` tool to an OpenHands
conversation.

The live path follows the OpenHands SDK's public extension points:

* ``CondenserBase`` returns a ``Condensation`` event whose summary is the
  source-faithful JSON User-Spine.
* the event callback records the append-only event stream and the current
  model-visible view;
* a registered OpenHands tool loads hidden components from the archive.

No API key or authorization header is ever written to the debug directory.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .memory import ComponentStore, HistoricalTurn, SpinePolicy


class OpenHandsUnavailableError(RuntimeError):
    """Raised when the live adapter is used without the optional SDK extra."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "client_secret",
    "password",
    "refresh_token",
    "secret_key",
}


def _jsonable(value: Any) -> Any:
    """Convert SDK/Pydantic objects into JSON without serializing secrets."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if str(key).lower() in _SENSITIVE_KEYS
            else _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json", exclude_none=True))
        except TypeError:
            return _jsonable(model_dump(exclude_none=True))
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return str(value)


def _message_dump(message: Any) -> dict[str, Any]:
    dumped = _jsonable(message)
    if isinstance(dumped, dict):
        return dumped
    return {"value": dumped}


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        if isinstance(item, Mapping) and isinstance(item.get("text"), str):
            parts.append(item["text"])
            continue
        parts.append(json.dumps(_jsonable(item), ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def _event_message(event: Any) -> Any | None:
    converter = getattr(event, "to_llm_message", None)
    if not callable(converter):
        return None
    try:
        return converter()
    except Exception:
        return None


def _event_role(event: Any) -> str | None:
    message = _event_message(event)
    role = getattr(message, "role", None)
    if role is None and isinstance(message, Mapping):
        role = message.get("role")
    return str(getattr(role, "value", role)) if role is not None else None


def _event_source(event: Any) -> str | None:
    source = getattr(event, "source", None)
    if source is None:
        return None
    return str(getattr(source, "value", source))


def _event_id(event: Any) -> str:
    event_id = getattr(event, "id", None)
    return str(event_id) if event_id is not None else f"event-{id(event)}"


def _event_record(event: Any) -> str:
    """Serialize an event as source-faithful text for a memory component."""

    payload = {
        "event_type": type(event).__name__,
        "event_id": _event_id(event),
        "source": _event_source(event),
        "role": _event_role(event),
        "event": _jsonable(event),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def historical_turns_from_events(events: Sequence[Any]) -> tuple[HistoricalTurn, ...]:
    """Project OpenHands events into Temporal-Spine's user-boundary turns.

    Real user messages start a turn.  Agent text is kept as an assistant
    result; tool calls and observations are kept as execution records.  The
    original event JSON is retained inside those fields so loading a component
    does not ask an LLM to regenerate it.
    """

    turns: list[HistoricalTurn] = []
    current_user: str | None = None
    assistant_records: list[str] = []
    execution_records: list[str] = []

    def flush() -> None:
        nonlocal current_user, assistant_records, execution_records
        if current_user is None:
            return
        turns.append(
            HistoricalTurn(
                turn_id=len(turns),
                user_request=current_user,
                assistant_result="\n\n".join(assistant_records),
                execution_records=tuple(execution_records),
            )
        )
        current_user = None
        assistant_records = []
        execution_records = []

    for event in events:
        role = _event_role(event)
        source = _event_source(event)
        # OpenHands distinguishes event source from LLM role.  Only a genuine
        # user-source event starts a new historical turn; condensation summary
        # events are environment-source user messages and must not do so.
        if source == "user" and role == "user":
            flush()
            message = _event_message(event)
            current_user = _message_text(message) if message is not None else _event_record(event)
            continue

        if current_user is None:
            continue
        if role == "assistant" and not getattr(_event_message(event), "tool_calls", None):
            assistant_records.append(_event_record(event))
        else:
            execution_records.append(_event_record(event))

    flush()
    return tuple(turns)


class TemporalSpineMemory:
    """Archive and progressive loader shared by the condenser and tool."""

    def __init__(self, policy: SpinePolicy | None = None) -> None:
        self.policy = policy or SpinePolicy()
        self._archive_events: list[Any] = []
        self._archive_ids: set[str] = set()
        self._manual_turns: tuple[HistoricalTurn, ...] | None = None
        self.store: ComponentStore | None = None
        self.loaded_action_ids: list[str] = []
        self.last_compression: dict[str, Any] | None = None

    def observe_event(self, event: Any) -> None:
        """Add one append-only SDK event to the source archive."""

        event_id = _event_id(event)
        if event_id not in self._archive_ids:
            self._archive_ids.add(event_id)
            self._archive_events.append(event)
        self._rebuild_store()

    def observe_events(self, events: Sequence[Any]) -> None:
        for event in events:
            self.observe_event(event)

    def set_historical_turns(self, turns: Sequence[HistoricalTurn]) -> None:
        """Set deterministic turns for offline tests."""

        self._manual_turns = tuple(turns)
        self._archive_events = []
        self._archive_ids = set()
        self._rebuild_store()

    def _rebuild_store(self) -> ComponentStore:
        turns = (
            self._manual_turns
            if self._manual_turns is not None
            else historical_turns_from_events(self._archive_events)
        )
        self.store = ComponentStore(turns, self.policy)
        self.loaded_action_ids = [
            action_id
            for action_id in self.loaded_action_ids
            if action_id in self.store.components
        ]
        return self.store

    def _require_store(self) -> ComponentStore:
        return self.store or self._rebuild_store()

    def spine_payload(self) -> list[dict[str, object]]:
        return self._require_store().spine_payload()

    def render_summary(self) -> str:
        payload = {
            "context_type": "openhands_temporal_spine_memory",
            "memory_policy": "historical user requests stay visible; archived assistant and execution evidence is loadable",
            "user_spine": self.spine_payload(),
            "loading_protocol": {
                "tool": "load_components",
                "selection": ["selected", "remaining"],
                "rule": "load the smallest non-repeated set needed for the current task",
            },
        }
        return (
            "[TEMPORAL-SPINE COMPRESSED HISTORY]\n"
            "The JSON below is the source-faithful User-Spine. Historical assistant "
            "results and tool records are hidden but recoverable through the "
            "load_components tool.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def load_components(
        self,
        selection: Literal["selected", "remaining"],
        action_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], str]:
        store = self._require_store()
        loaded = set(self.loaded_action_ids)
        if selection == "selected":
            if not action_ids:
                raise ValueError("selected loading requires at least one action ID")
            requested = list(action_ids)
        elif selection == "remaining":
            if action_ids:
                raise ValueError("remaining loading requires an empty action_ids list")
            requested = [action_id for action_id in store.action_ids if action_id not in loaded]
        else:
            raise ValueError("selection must be selected or remaining")

        repeated = [action_id for action_id in requested if action_id in loaded]
        if repeated:
            raise ValueError(f"components already loaded: {repeated}")
        ordered = store.chronological(requested)
        self.loaded_action_ids.extend(ordered)
        return tuple(ordered), store.render_components(ordered) or "(no remaining components)"

    def record_compression(self, *, visible_events: int, forgotten_events: int) -> None:
        self.last_compression = {
            "visible_events_before": visible_events,
            "forgotten_events": forgotten_events,
            "archive_events": len(self._archive_events),
            "user_turns": len(self._require_store().spine_payload()),
            "timestamp": _now(),
        }

    def snapshot(self) -> dict[str, Any]:
        store = self._require_store()
        return {
            "archive_event_count": len(self._archive_events),
            "historical_turn_count": len(store.spine_payload()),
            "available_action_ids": list(store.action_ids),
            "loaded_action_ids": list(self.loaded_action_ids),
            "last_compression": self.last_compression,
        }


class OpenHandsDebugRecorder:
    """Persist event, model-message, and latest-view diagnostics as JSON."""

    def __init__(
        self,
        log_dir: str | Path,
        memory: TemporalSpineMemory | None = None,
        tools: Sequence[Any] = (),
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.memory = memory
        self._tools = list(tools)
        self._events: list[Any] = []
        self._context_index = 0
        self._request_index = 0
        self._write_json(
            self.log_dir / "metadata.json",
            {
                "created_at": _now(),
                "purpose": "OpenHands SDK Temporal-Spine debugging",
                "secrets": "API keys and authorization headers are excluded",
            },
        )

    def set_tools(self, tools: Sequence[Any]) -> None:
        """Set the tool specs to include in every subsequent raw-context file."""

        self._tools = list(tools)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _append_jsonl(self, path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")

    def record_context(
        self,
        messages: Sequence[Any],
        *,
        phase: str,
        tools: Sequence[Any] | None = None,
        trigger: str | None = None,
    ) -> Path:
        self._context_index += 1
        payload = {
            "context_index": self._context_index,
            "timestamp": _now(),
            "phase": phase,
            "trigger": trigger,
            "messages": [_message_dump(message) for message in messages],
            "tools": [_jsonable(tool) for tool in (self._tools if tools is None else tools)],
            "memory": self.memory.snapshot() if self.memory is not None else None,
        }
        path = self.log_dir / f"raw_context_{self._context_index:04d}.json"
        self._write_json(path, payload)
        self._write_json(self.log_dir / "latest_raw_context.json", payload)
        return path

    def record_model_request(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
        request_params: Mapping[str, Any] | None = None,
        phase: str = "provider_request",
    ) -> Path:
        """Persist the provider-ready request before it is sent to the model."""

        self._request_index += 1
        payload = {
            "request_index": self._request_index,
            "timestamp": _now(),
            "phase": phase,
            "messages": [_message_dump(message) for message in messages],
            "tools": [_jsonable(tool) for tool in tools],
            "request_params": _jsonable(request_params or {}),
        }
        path = self.log_dir / f"model_request_{self._request_index:04d}.json"
        self._write_json(path, payload)
        self._write_json(self.log_dir / "latest_model_request.json", payload)
        return path

    def callback(self, event: Any) -> None:
        """OpenHands Conversation callback suitable for ``callbacks=[...]``."""

        self._events.append(event)
        if self.memory is not None:
            self.memory.observe_event(event)
        message = _event_message(event)
        row = {
            "timestamp": _now(),
            "event_index": len(self._events) - 1,
            "event_type": type(event).__name__,
            "event": _jsonable(event),
            "llm_message": _message_dump(message) if message is not None else None,
        }
        self._append_jsonl(self.log_dir / "events.jsonl", row)
        if message is not None:
            self._append_jsonl(
                self.log_dir / "llm_messages.jsonl",
                {
                    "timestamp": row["timestamp"],
                    "event_index": row["event_index"],
                    "event_type": row["event_type"],
                    "message": _message_dump(message),
                },
            )
        self._snapshot_view(trigger=type(event).__name__)

    def _snapshot_view(self, *, trigger: str) -> None:
        messages: list[Any] = []
        try:
            from openhands.sdk.context.view import View

            view = View.from_events(self._events)
            messages = [event.to_llm_message() for event in view.events]
        except Exception:
            messages = [
                message
                for event in self._events
                if (message := _event_message(event)) is not None
            ]
        if messages:
            self.record_context(messages, phase="event_callback_view", trigger=trigger)

    @property
    def latest_context_path(self) -> Path:
        return self.log_dir / "latest_raw_context.json"


def load_components_tool_schema() -> dict[str, Any]:
    """Return an OpenAI-compatible schema for offline inspection."""

    return {
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
    }


@dataclass(frozen=True)
class OfflineTestResult:
    log_dir: Path
    latest_context: Path
    loaded_action_ids: tuple[str, ...]
    available_action_ids: tuple[str, ...]


def run_offline_test(
    log_dir: str | Path = ".openhands-temporal-spine/test-run",
    *,
    print_context: bool = True,
) -> OfflineTestResult:
    """Run a deterministic compression/load test without an SDK or API call."""

    memory = TemporalSpineMemory()
    memory.set_historical_turns(
        [
            HistoricalTurn(
                0,
                "Create a Python parser for the project configuration file.",
                "Implemented parser.py with a ConfigParser class.",
                ("pytest tests/test_parser.py -q -> 4 passed",),
            ),
            HistoricalTurn(
                1,
                "The parser must preserve comments and report line numbers for errors.",
                "Updated the parser to retain comments and attach line numbers to ParseError.",
                ("pytest tests/test_parser.py -q -> 7 passed",),
            ),
        ]
    )
    recorder = OpenHandsDebugRecorder(log_dir, memory)
    system = {
        "role": "system",
        "content": "You are an OpenHands coding agent with Temporal-Spine memory.",
    }
    summary = {"role": "user", "content": memory.render_summary()}
    tools = [load_components_tool_schema()]
    before = recorder.record_context(
        [system, summary],
        phase="after_compression_before_load",
        tools=tools,
    )
    recorder.record_model_request(
        [system, summary],
        tools=tools,
        request_params={"mode": "offline_test", "phase": "after_compression_before_load"},
    )

    result_id = next(
        action_id for action_id in memory._require_store().action_ids if action_id.startswith("result:")
    )
    loaded_ids, loaded_text = memory.load_components("selected", [result_id])
    tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "offline-load-1",
                "type": "function",
                "function": {
                    "name": "load_components",
                    "arguments": json.dumps(
                        {"selection": "selected", "action_ids": [result_id]},
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }
    after = recorder.record_context(
        [
            system,
            summary,
            tool_call,
            {"role": "tool", "tool_call_id": "offline-load-1", "content": loaded_text},
        ],
        phase="after_load_components",
        tools=tools,
    )
    recorder.record_model_request(
        [
            system,
            summary,
            tool_call,
            {"role": "tool", "tool_call_id": "offline-load-1", "content": loaded_text},
        ],
        tools=tools,
        request_params={"mode": "offline_test", "phase": "after_load_components"},
    )
    recorder._write_json(
        Path(log_dir) / "summary.json",
        {
            "mode": "offline_test",
            "compression_verified": True,
            "loading_verified": True,
            "before_context": str(before),
            "after_context": str(after),
            "memory": memory.snapshot(),
        },
    )
    if print_context:
        print("=== Temporal-Spine offline test: raw context after compression ===")
        print(before.read_text(encoding="utf-8"))
        print("=== Temporal-Spine offline test: raw context after load_components ===")
        print(after.read_text(encoding="utf-8"))
    store = memory._require_store()
    return OfflineTestResult(
        log_dir=Path(log_dir),
        latest_context=recorder.latest_context_path,
        loaded_action_ids=loaded_ids,
        available_action_ids=store.action_ids,
    )


try:  # Keep the core package importable without the optional OpenHands extra.
    from pydantic import Field, PrivateAttr

    from openhands.sdk import (  # type: ignore[import-not-found]
        Action,
        Agent,
        Conversation,
        Event,
        LLM,
        Observation,
        TextContent,
        ToolDefinition,
    )
    from openhands.sdk.context.condenser import CondenserBase  # type: ignore[import-not-found]
    from openhands.sdk.context.view import View  # type: ignore[import-not-found]
    from openhands.sdk.event.condenser import Condensation  # type: ignore[import-not-found]
    from openhands.sdk.tool import Tool, ToolExecutor, register_tool  # type: ignore[import-not-found]
    from rich.text import Text

    _OPENHANDS_IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - environment dependent
    _OPENHANDS_IMPORT_ERROR = exc


if _OPENHANDS_IMPORT_ERROR is None:
    _MEMORY_REGISTRY: dict[str, TemporalSpineMemory] = {}

    class RecordingLLM(LLM):
        """OpenHands LLM that records provider-ready chat requests."""

        _debug_recorder: OpenHandsDebugRecorder | None = PrivateAttr(default=None)

        def attach_debug_recorder(self, recorder: OpenHandsDebugRecorder) -> "RecordingLLM":
            self._debug_recorder = recorder
            return self

        def _finalize_completion_params(
            self,
            formatted_messages: list[dict[str, Any]],
            tools: Sequence[Any] | None,
            add_security_risk_prediction: bool,
            kwargs: dict[str, Any],
            call_context: Any = None,
        ) -> tuple[Any, ...]:
            result = super()._finalize_completion_params(
                formatted_messages,
                tools,
                add_security_risk_prediction,
                kwargs,
                call_context=call_context,
            )
            if self._debug_recorder is not None:
                _, provider_tools, _, request_params, _ = result
                self._debug_recorder.record_model_request(
                    formatted_messages,
                    tools=provider_tools,
                    request_params=request_params,
                )
            return result

    class TemporalSpineCondenser(CondenserBase):
        """OpenHands condenser that emits a JSON User-Spine, not a summary."""

        max_events: int = Field(default=24, ge=2)
        keep_recent_events: int = Field(default=6, ge=0)
        _memory: TemporalSpineMemory | None = PrivateAttr(default=None)

        def attach(self, memory: TemporalSpineMemory) -> "TemporalSpineCondenser":
            self._memory = memory
            return self

        def handles_condensation_requests(self) -> bool:
            return True

        def condense(self, view: View, agent_llm: Any = None) -> View | Condensation:
            del agent_llm
            if self._memory is None:
                raise RuntimeError("TemporalSpineCondenser is not attached to memory")
            self._memory.observe_events(view.events)
            if len(view.events) <= self.max_events and not view.unhandled_condensation_request:
                return view

            events = list(view.events)
            if len(events) <= 1:
                return view
            tail_start = max(1, len(events) - self.keep_recent_events)
            keep_indices = set(range(tail_start, len(events)))
            # Preserve all leading system messages.  The summary is inserted
            # immediately after them so OpenHands' message ordering remains valid.
            for index, event in enumerate(events):
                if index < tail_start and _event_role(event) == "system":
                    keep_indices.add(index)
            forgotten = {
                _event_id(event)
                for index, event in enumerate(events)
                if index not in keep_indices
            }
            if not forgotten:
                return view

            retained = [event for index, event in enumerate(events) if index in keep_indices]
            summary_offset = 0
            for event in retained:
                if _event_role(event) == "system":
                    summary_offset += 1
                else:
                    break
            response_ids = [
                str(getattr(event, "llm_response_id"))
                for event in events
                if getattr(event, "llm_response_id", None) is not None
            ]
            llm_response_id = response_ids[-1] if response_ids else f"temporal-spine-{uuid.uuid4().hex}"
            self._memory.record_compression(
                visible_events=len(events),
                forgotten_events=len(forgotten),
            )
            return Condensation(
                forgotten_event_ids=forgotten,
                summary=self._memory.render_summary(),
                summary_offset=summary_offset,
                llm_response_id=llm_response_id,
            )


    class TemporalSpineLoadAction(Action):
        selection: Literal["selected", "remaining"] = "selected"
        action_ids: list[str] = Field(default_factory=list)

        @property
        def visualize(self) -> Text:
            return Text(f"load_components({self.selection}, {self.action_ids})")


    class TemporalSpineLoadObservation(Observation):
        loaded_action_ids: list[str] = Field(default_factory=list)


    class TemporalSpineLoadExecutor(ToolExecutor[TemporalSpineLoadAction, TemporalSpineLoadObservation]):
        def __init__(self, memory: TemporalSpineMemory) -> None:
            self.memory = memory

        def __call__(
            self,
            action: TemporalSpineLoadAction,
            conversation: Any = None,
        ) -> TemporalSpineLoadObservation:
            del conversation
            loaded_ids, text = self.memory.load_components(action.selection, action.action_ids)
            return TemporalSpineLoadObservation(
                loaded_action_ids=list(loaded_ids),
                content=[TextContent(text=text)],
            )


    _TEMPORAL_SPINE_LOAD_DESCRIPTION = (
        "Load selected historical assistant, execution, or exact user components "
        "from Temporal-Spine compressed conversation memory."
    )


    class TemporalSpineLoadTool(ToolDefinition[TemporalSpineLoadAction, TemporalSpineLoadObservation]):
        name = "load_components"

        @classmethod
        def create(cls, conv_state: Any = None, memory_id: str | None = None, **params: Any) -> Sequence[Any]:
            del conv_state
            key = memory_id or params.get("memory_id")
            memory = _MEMORY_REGISTRY.get(str(key)) if key is not None else None
            if memory is None:
                raise RuntimeError("Temporal-Spine memory registry entry is missing")
            return [
                cls(
                    description=_TEMPORAL_SPINE_LOAD_DESCRIPTION,
                    action_type=TemporalSpineLoadAction,
                    observation_type=TemporalSpineLoadObservation,
                    executor=TemporalSpineLoadExecutor(memory),
                )
            ]


    @dataclass
    class OpenHandsSession:
        conversation: Any
        memory: TemporalSpineMemory
        recorder: OpenHandsDebugRecorder

        def send(self, message: str) -> Any:
            self.conversation.send_message(message)
            return self.conversation.run()

        def force_compress(self) -> Any:
            return self.conversation.condense()

        @property
        def latest_raw_context(self) -> Path:
            return self.recorder.latest_context_path

        def close(self) -> None:
            self.conversation.close()


    class OpenHandsTemporalSpine:
        """Factory for a real OpenHands coding conversation with Temporal-Spine."""

        def __init__(
            self,
            *,
            log_dir: str | Path = ".openhands-temporal-spine/live-run",
            policy: SpinePolicy | None = None,
            max_events: int = 24,
            keep_recent_events: int = 6,
        ) -> None:
            self.memory = TemporalSpineMemory(policy)
            self.recorder = OpenHandsDebugRecorder(log_dir, self.memory)
            self.max_events = max_events
            self.keep_recent_events = keep_recent_events
            self._memory_id = uuid.uuid4().hex
            _MEMORY_REGISTRY[self._memory_id] = self.memory
            register_tool(TemporalSpineLoadTool.name, TemporalSpineLoadTool)

        def create_conversation(
            self,
            *,
            model: str,
            api_key_env: str = "LLM_API_KEY",
            base_url: str | None = None,
            workspace: str | Path = ".",
            persistence_dir: str | Path | None = ".openhands-temporal-spine/conversations",
            conversation_id: Any | None = None,
        ) -> OpenHandsSession:
            from pydantic import SecretStr
            from openhands.tools.file_editor import FileEditorTool
            from openhands.tools.terminal import TerminalTool

            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Set the model key through environment variable {api_key_env}; "
                    "the adapter never stores keys in logs or source files."
                )
            llm = RecordingLLM(
                usage_id="agent",
                model=model,
                base_url=base_url or os.getenv("LLM_BASE_URL"),
                api_key=SecretStr(api_key),
            ).attach_debug_recorder(self.recorder)
            condenser = TemporalSpineCondenser(
                max_events=self.max_events,
                keep_recent_events=self.keep_recent_events,
            ).attach(self.memory)
            tools = [
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
                Tool(name=TemporalSpineLoadTool.name, params={"memory_id": self._memory_id}),
            ]
            self.recorder.set_tools(tools)
            agent = Agent(llm=llm, tools=tools, condenser=condenser)
            conversation = Conversation(
                agent=agent,
                callbacks=[self.recorder.callback],
                workspace=str(workspace),
                persistence_dir=str(persistence_dir) if persistence_dir is not None else None,
                conversation_id=conversation_id,
            )
            return OpenHandsSession(conversation, self.memory, self.recorder)


else:

    class TemporalSpineCondenser:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise OpenHandsUnavailableError(
                "Install the optional OpenHands integration: "
                "python -m pip install -e '.[openhands]'"
            ) from _OPENHANDS_IMPORT_ERROR


    class OpenHandsTemporalSpine:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise OpenHandsUnavailableError(
                "Install the optional OpenHands integration: "
                "python -m pip install -e '.[openhands]'"
            ) from _OPENHANDS_IMPORT_ERROR


    @dataclass
    class OpenHandsSession:  # type: ignore[no-redef]
        """Import-compatible placeholder when the optional SDK is absent."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise OpenHandsUnavailableError(
                "Install the optional OpenHands integration: "
                "python -m pip install -e '.[openhands]'"
            ) from _OPENHANDS_IMPORT_ERROR


__all__ = [
    "OfflineTestResult",
    "OpenHandsDebugRecorder",
    "OpenHandsSession",
    "OpenHandsTemporalSpine",
    "OpenHandsUnavailableError",
    "TemporalSpineCondenser",
    "TemporalSpineMemory",
    "historical_turns_from_events",
    "load_components_tool_schema",
    "run_offline_test",
]
