import json

import pytest

from temporal_spine import ComponentStore, HistoricalTurn, ModelResponse, ProtocolError, RunConfig, SpinePolicy, TemporalSpine
from test_core import FakeModel, response, turns


def load(ids, call="load"):
    return response("load_components", {"selection": "selected", "action_ids": ids}, call)


def submit():
    return response("submit_final_result", {"result": "done"}, "final")


def test_progressive_batches_deduplicate_without_resorting_previous_messages():
    model = FakeModel([load(["result:2"], "l1"), load(["result:2", "result:0"], "l2"), submit()])
    result = TemporalSpine(model, RunConfig(max_access_calls=2)).run(turns(3), "question")
    assert result.loaded_action_ids == ("result:2", "result:0")
    assert len(result.calls) == 3
    assert len({call.fixed_request_hash for call in result.calls}) == 1
    first_payload = json.loads(model.requests[1][0][-1]["content"])
    last_payload = json.loads(model.requests[2][0][-1]["content"])
    assert first_payload["history_access_state"]["history_access_calls_remaining"] == 1
    assert last_payload["already_loaded_action_ids"] == ["result:2"]
    assert [row["action_id"] for row in last_payload["components"]] == ["result:0"]
    assert last_payload["history_access_state"]["allowed_next_tools"] == ["submit_final_result"]
    assert model.requests[2][0][:len(model.requests[1][0])] == model.requests[1][0]


def test_valid_repeated_only_batch_consumes_access_budget():
    model = FakeModel([load(["result:0"], "l1"), load(["result:0"], "l2"), submit()])
    result = TemporalSpine(model, RunConfig(max_access_calls=2)).run(turns(1), "question")
    assert result.loaded_action_ids == ("result:0",)
    payload = json.loads(model.requests[2][0][-1]["content"])
    assert payload["components"] == []
    assert payload["history_access_state"]["history_access_calls_used"] == 2
    assert result.invalid_action_count == 0


def test_overflow_is_cut_before_retained_id_validation():
    model = FakeModel([load(["result:0", "result:0", "unknown"]), submit()])
    result = TemporalSpine(model, RunConfig(max_components_per_load=1)).run(turns(1), "question")
    assert result.invalid_action_count == 0
    assert result.loaded_action_ids == ("result:0",)
    assert result.dropped_action_ids == ("result:0", "unknown")


def test_duplicate_retained_ids_rejected_before_loading():
    model = FakeModel([load(["result:0", "result:0"]), submit()])
    result = TemporalSpine(model).run(turns(1), "question")
    assert result.invalid_action_count == 1
    assert result.loaded_action_ids == ()


def test_invalid_repair_cannot_extend_total_call_limit():
    model = FakeModel([load(["result:0"], "l1"), load(["unknown"], "bad"), submit()])
    result = TemporalSpine(model, RunConfig(max_access_calls=2)).run(turns(1), "question")
    state = json.loads(model.requests[2][0][-1]["content"])["history_access_state"]
    assert state["history_access_closed"] is True
    assert len(result.calls) == 3
    assert result.invalid_action_count == 1


def test_zero_budget_never_loads_hidden_content():
    model = FakeModel([load(["result:0"]), submit()])
    result = TemporalSpine(model, RunConfig(max_access_calls=0)).run(turns(1), "question")
    assert result.loaded_action_ids == ()
    assert result.invalid_action_count == 1


def test_unrequested_originals_never_appear_in_initial_context():
    model = FakeModel([submit()])
    TemporalSpine(model).run([HistoricalTurn(5, "my request", "HIDDEN_ASSISTANT_ORIGINAL", timestamp="2026-01-01")], "question", current_date="2026-01-02")
    initial = json.dumps(model.requests[0][0])
    assert "HIDDEN_ASSISTANT_ORIGINAL" not in initial
    assert "2026-01-01" in initial and "2026-01-02" in initial
    assert "result:5" in initial


def test_original_text_is_returned_verbatim():
    original = "  First line.\n\nSecond line.  "
    store = ComponentStore([HistoricalTurn(0, "request", original)])
    assert store.loaded_component_payload("result:0")["authoritative_content"] == original


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_truncated_or_filtered_output_is_not_captured_as_final(reason):
    with pytest.raises(ProtocolError, match="incomplete"):
        TemporalSpine(FakeModel([ModelResponse(content="partial", finish_reason=reason)])).run([], "q")


@pytest.mark.parametrize("config", [
    {"max_access_calls": 3}, {"max_access_calls": 2, "max_model_calls": 2},
    {"max_access_calls": True}, {"max_components_per_load": 0},
    {"max_invalid_actions": 2},
])
def test_invalid_limits_fail_early(config):
    with pytest.raises(ValueError):
        RunConfig(**config)


@pytest.mark.parametrize("turn_id", [-1, True, "1"])
def test_invalid_turn_ids_rejected(turn_id):
    with pytest.raises(ValueError):
        HistoricalTurn(turn_id, "q")


def test_duplicate_turn_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        ComponentStore([HistoricalTurn(0, "a"), HistoricalTurn(0, "b")])


def test_cjk_span_preview_and_original_recovery():
    original = "甲乙丙丁戊己庚辛壬癸"
    store = ComponentStore([HistoricalTurn(0, original)], SpinePolicy(6, 3, 3))
    preview = store.spine_turns[0].visible_user_request
    assert preview.startswith("甲乙丙") and preview.endswith("辛壬癸")
    assert "4 spans omitted" in preview
    assert store.loaded_component_payload("user:0")["authoritative_content"] == original


def test_extra_tool_arguments_are_rejected():
    model = FakeModel([response("submit_final_result", {"result": "done", "extra": True}, "bad"), submit()])
    assert TemporalSpine(model).run([], "q").invalid_action_count == 1


def test_fixed_parameters_cannot_drift_between_calls():
    class DriftingModel(FakeModel):
        @property
        def fixed_parameters(self):
            return {"model": "changed" if self.requests else "initial"}

    model = DriftingModel([load(["result:0"]), submit()])
    with pytest.raises(ProtocolError, match="parameters changed"):
        TemporalSpine(model).run(turns(1), "q")
    assert len(model.requests) == 1
