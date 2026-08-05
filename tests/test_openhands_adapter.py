from __future__ import annotations

import json
from pathlib import Path

from temporal_spine import HistoricalTurn
from temporal_spine.openhands_adapter import (
    OpenHandsDebugRecorder,
    TemporalSpineMemory,
    load_components_tool_schema,
    run_offline_test,
)


def test_offline_openhands_mode_writes_raw_context(tmp_path: Path) -> None:
    result = run_offline_test(tmp_path, print_context=False)

    assert result.latest_context.exists()
    latest = json.loads(result.latest_context.read_text(encoding="utf-8"))
    assert latest["phase"] == "after_load_components"
    assert any(message["role"] == "tool" for message in latest["messages"])
    assert result.loaded_action_ids
    assert (tmp_path / "summary.json").exists()
    model_request = json.loads(
        (tmp_path / "latest_model_request.json").read_text(encoding="utf-8")
    )
    assert model_request["phase"] == "provider_request"
    assert model_request["tools"][0]["function"]["name"] == "load_components"


def test_memory_load_is_monotonic_and_source_faithful() -> None:
    memory = TemporalSpineMemory()
    memory.set_historical_turns(
        [HistoricalTurn(0, "Remember this requirement.", "The exact assistant result.")]
    )

    ids, content = memory.load_components("selected", ["result:0"])

    assert ids == ("result:0",)
    assert "The exact assistant result." in content
    assert memory.loaded_action_ids == ["result:0"]


def test_debug_recorder_keeps_event_and_context_logs(tmp_path: Path) -> None:
    memory = TemporalSpineMemory()
    recorder = OpenHandsDebugRecorder(tmp_path, memory)
    recorder.record_context(
        [{"role": "system", "content": "system"}],
        phase="test",
        tools=[load_components_tool_schema()],
    )

    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "latest_raw_context.json").exists()
    assert json.loads((tmp_path / "latest_raw_context.json").read_text())[
        "tools"
    ][0]["function"]["name"] == "load_components"


def test_debug_recorder_redacts_sensitive_request_parameters(tmp_path: Path) -> None:
    recorder = OpenHandsDebugRecorder(tmp_path)
    recorder.record_model_request(
        [{"role": "user", "content": "hello"}],
        request_params={"api_key": "do-not-write", "model": "test-model"},
    )

    payload = json.loads(
        (tmp_path / "latest_model_request.json").read_text(encoding="utf-8")
    )
    assert payload["request_params"] == {
        "api_key": "[REDACTED]",
        "model": "test-model",
    }
    assert "do-not-write" not in (tmp_path / "latest_model_request.json").read_text()
