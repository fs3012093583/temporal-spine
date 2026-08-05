from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("gradio")


def _load_app():
    path = Path(__file__).parents[1] / "examples" / "gradio_app.py"
    spec = importlib.util.spec_from_file_location("temporal_spine_gradio_app", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_compression_moves_hot_turns_into_archive() -> None:
    app = _load_app()
    state = app.initial_chat_state()
    state["hot_messages"] = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    compressed, status, preview = app.compress_history(state)

    assert compressed["compression_active"] is True
    assert compressed["hot_messages"] == []
    assert [turn["turn_id"] for turn in compressed["archived_turns"]] == [0, 1]
    assert preview["compressed_turns"] == 2
    assert preview["user_spine"][1]["loadable_components"] == {
        "assistant_result": "result:1"
    }
    assert "已压缩 2 个新轮次" in status


def test_repeated_compression_appends_new_turn_with_monotonic_id() -> None:
    app = _load_app()
    state = app.initial_chat_state()
    state["hot_messages"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]
    state, _, _ = app.compress_history(state)
    state["hot_messages"] = [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer two"},
    ]

    state, _, preview = app.compress_history(state)

    assert [turn["turn_id"] for turn in state["archived_turns"]] == [0, 1]
    assert state["next_turn_id"] == 2
    assert preview["compressed_turns"] == 2


def test_model_input_renderer_is_readable_json() -> None:
    app = _load_app()
    assert app._render_model_inputs(None) == "[]"
    assert app._render_raw_request([]) == ""
    assert app._render_raw_request(["你好".encode("utf-8")]) == "你好"


def test_pretty_request_expands_raw_json_without_changing_values() -> None:
    app = _load_app()
    raw = '{"messages":[{"role":"user","content":"你好"}],"tools":[]}'
    pretty = app._render_pretty_request([raw.encode("utf-8")])
    assert "\n" in pretty
    assert app.json.loads(pretty) == app.json.loads(raw)


def test_post_answer_context_appends_assistant_after_raw_user_request() -> None:
    app = _load_app()
    raw = '{"messages":[{"role":"user","content":"问题"}],"tools":[]}'
    snapshot = app.json.loads(
        app._render_post_answer_context([raw.encode("utf-8")], "回答")
    )

    assert snapshot["messages"][-2:] == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]
    assert app.json.loads(app._render_raw_request([raw.encode("utf-8")]))[
        "messages"
    ][-1]["role"] == "user"


def test_demo_conversation_sends_ten_messages_in_order(monkeypatch) -> None:
    app = _load_app()
    sent_messages: list[str] = []

    def fake_send_message(
        message,
        chat_history,
        chat_state,
        api_key,
        model_name,
        base_url,
        max_load_rounds,
        full_fallback_enabled,
    ):
        sent_messages.append(message)
        history = list(chat_history)
        history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"回答 {len(sent_messages)}"},
            ]
        )
        state = app.copy.deepcopy(chat_state)
        state["hot_messages"] = history.copy()
        return history, "", state, [], {}, [], "ok", "pretty", "raw", {}

    monkeypatch.setattr(app, "send_message", fake_send_message)

    updates = list(
        app.run_demo_conversation(
            [], app.initial_chat_state(), "key", "model", "https://example.test", 2, False
        )
    )

    assert sent_messages == app.DEMO_MESSAGES
    assert len(updates) == 10
    assert len(updates[-1][0]) == 20
    assert "Demo 10/10 已完成" in updates[-1][6]
