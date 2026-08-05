from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterator
from datetime import date
from typing import Any, Callable

try:
    import gradio as gr
except ModuleNotFoundError as exc:  # pragma: no cover - optional UI dependency
    raise ModuleNotFoundError(
        "The web demo requires: python -m pip install -e '.[ui]'"
    ) from exc

from temporal_spine import (
    ComponentStore,
    HistoricalTurn,
    OpenAICompatibleClient,
    RunConfig,
)
from temporal_spine.demo import RecordingChatModel, result_metrics
from temporal_spine.langgraph_adapter import LangGraphTemporalSpine


NORMAL_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Continue the conversation naturally and answer the "
    "user's latest message."
)

DEMO_MESSAGES = [
    "我叫林舟，正在学习强化学习。",
    "我每周能学习 8 小时，主要使用 Python。",
    "我对花生过敏，早餐建议需要避开花生。",
    "请给我制定一个四周的强化学习入门计划。",
    "第一周应该重点学习哪些内容？",
    "请介绍 Q-Learning 的核心更新公式。",
    "探索与利用之间有什么区别？",
    "给我设计一个 FrozenLake 练习任务。",
    "请提醒我之前说过的每周学习时间和饮食限制。",
    "总结我的学习目标，并给出下一步行动。",
]

ENTER_TO_SEND_JS = r"""
const installEnterHandler = () => {
  const root = document.querySelector("#chat-input");
  const input = root?.querySelector("textarea, input");
  const submit = root?.querySelector("button");
  if (!input || !submit) return false;
  if (input.dataset.temporalSpineEnterBound === "true") return true;
  input.dataset.temporalSpineEnterBound = "true";
  input.addEventListener("keydown", (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.isComposing
    ) {
      event.preventDefault();
      event.stopImmediatePropagation();
      submit.click();
    }
  }, true);
  return true;
};
if (!installEnterHandler()) {
  const observer = new MutationObserver(() => {
    if (installEnterHandler()) observer.disconnect();
  });
  observer.observe(document.body, {childList: true, subtree: true});
}
"""


def _render_model_inputs(recorder: RecordingChatModel | None) -> str:
    requests = recorder.numbered_requests() if recorder else []
    return json.dumps(requests, ensure_ascii=False, indent=2)


def _render_raw_request(raw_requests: list[bytes]) -> str:
    if not raw_requests:
        return ""
    return raw_requests[-1].decode("utf-8")


def _render_pretty_request(raw_requests: list[bytes]) -> str:
    raw_request = _render_raw_request(raw_requests)
    if not raw_request:
        return ""
    return json.dumps(json.loads(raw_request), ensure_ascii=False, indent=2)


def _render_post_answer_context(raw_requests: list[bytes], answer: str) -> str:
    raw_request = _render_raw_request(raw_requests)
    if not raw_request:
        return ""
    body = json.loads(raw_request)
    body["messages"] = [
        *body.get("messages", []),
        {"role": "assistant", "content": answer},
    ]
    return json.dumps(body, ensure_ascii=False, indent=2)


def initial_chat_state() -> dict[str, Any]:
    return {
        "compression_active": False,
        "archived_turns": [],
        "hot_messages": [],
        "next_turn_id": 0,
    }


def _historical_turns(state: dict[str, Any]) -> list[HistoricalTurn]:
    return [HistoricalTurn(**item) for item in state["archived_turns"]]


def _memory_preview(state: dict[str, Any]) -> dict[str, Any]:
    turns = _historical_turns(state)
    if not turns:
        return {
            "mode": "normal_full_history",
            "compressed_turns": 0,
            "hot_turns": len(state["hot_messages"]) // 2,
            "user_spine": [],
        }
    store = ComponentStore(turns)
    return {
        "mode": "temporal_spine",
        "compressed_turns": len(turns),
        "hot_turns": len(state["hot_messages"]) // 2,
        "user_spine": store.spine_payload(),
    }


def _client(
    api_key: str,
    model_name: str,
    base_url: str,
    request_observer: Callable[[bytes], None],
) -> OpenAICompatibleClient:
    resolved_api_key = api_key.strip() or os.getenv("DEEPSEEK_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "请设置 DEEPSEEK_API_KEY；当前页面启动进程没有可用的 API Key"
        )
    if not model_name.strip():
        raise ValueError("model name must not be empty")
    if not base_url.strip().startswith(("https://", "http://")):
        raise ValueError("API URL must start with https:// or http://")
    return OpenAICompatibleClient(
        model=model_name.strip(),
        base_url=base_url.strip(),
        api_key=resolved_api_key,
        max_tokens=512,
        request_observer=request_observer,
    )


def send_message(
    message: str,
    chat_history: list[dict[str, Any]] | None,
    chat_state: dict[str, Any] | None,
    api_key: str,
    model_name: str,
    base_url: str,
    max_load_rounds: float,
    full_fallback_enabled: bool,
) -> tuple[
    list[dict[str, Any]],
    str,
    dict[str, Any],
    list[str],
    dict[str, Any],
    list[dict[str, Any]],
    str,
    str,
    str,
    dict[str, Any],
]:
    """Send one chat turn in normal or post-compression mode."""

    history = copy.deepcopy(chat_history or [])
    state = copy.deepcopy(chat_state or initial_chat_state())
    user_text = message.strip()
    if not user_text:
        return (
            history,
            message,
            state,
            [],
            {},
            [],
            "请输入消息。",
            "",
            "",
            _memory_preview(state),
        )

    recorder: RecordingChatModel | None = None
    raw_requests: list[bytes] = []
    try:
        recorder = RecordingChatModel(
            _client(api_key, model_name, base_url, raw_requests.append)
        )
        if state["compression_active"]:
            result = LangGraphTemporalSpine(
                recorder,
                RunConfig(
                    max_load_rounds=int(max_load_rounds),
                    full_fallback_enabled=full_fallback_enabled,
                ),
            ).run(
                _historical_turns(state),
                user_text,
                current_date=date.today().isoformat(),
                recent_messages=state["hot_messages"],
            )
            answer = result.answer
            loaded_ids = list(result.loaded_action_ids)
            metrics = result_metrics(result)
            metrics["mode"] = "temporal_spine"
            trace = [dict(event) for event in result.trace]
            status = (
                f"Temporal-Spine 模式：{len(state['archived_turns'])} 个压缩轮次，"
                f"本轮加载 {len(loaded_ids)} 个组件。"
            )
        else:
            response = recorder.complete(
                [
                    {"role": "system", "content": NORMAL_CHAT_SYSTEM_PROMPT},
                    *copy.deepcopy(state["hot_messages"]),
                    {"role": "user", "content": user_text},
                ],
                [],
            )
            if response.tool_calls or not response.content.strip():
                raise RuntimeError("normal chat requires a plain assistant response")
            answer = response.content.strip()
            loaded_ids = []
            usage = dict(response.usage)
            metrics = {
                "mode": "normal_full_history",
                "model_calls": 1,
                "fallback_used": False,
                "total_usage": usage,
                "calls": [
                    {
                        "index": 0,
                        "phase": "normal_chat",
                        "usage": usage,
                    }
                ],
            }
            trace = [{"event": "normal_chat_answer_returned"}]
            status = (
                "普通聊天模式：完整未压缩对话直接进入模型。"
                "需要时点击“压缩历史”。"
            )
    except Exception as exc:
        return (
            history,
            message,
            state,
            [],
            {},
            [],
            f"运行失败：{type(exc).__name__}: {exc}",
            _render_pretty_request(raw_requests),
            _render_raw_request(raw_requests),
            _memory_preview(state),
        )

    history.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
    )
    state["hot_messages"].extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
    )
    return (
        history,
        "",
        state,
        loaded_ids,
        metrics,
        trace,
        status,
        _render_post_answer_context(raw_requests, answer),
        _render_raw_request(raw_requests),
        _memory_preview(state),
    )


def compress_history(
    chat_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Move completed hot turns into the source-faithful compressed store."""

    state = copy.deepcopy(chat_state or initial_chat_state())
    hot_messages = state["hot_messages"]
    if not hot_messages:
        return state, "没有新的完整对话轮次可以压缩。", _memory_preview(state)
    if len(hot_messages) % 2:
        return state, "当前存在未完成轮次，暂时不能压缩。", _memory_preview(state)

    new_turns = 0
    for index in range(0, len(hot_messages), 2):
        user_message = hot_messages[index]
        assistant_message = hot_messages[index + 1]
        if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
            return state, "热对话角色顺序无效，未执行压缩。", _memory_preview(state)
        turn_id = state["next_turn_id"]
        state["archived_turns"].append(
            {
                "turn_id": turn_id,
                "user_request": str(user_message["content"]),
                "assistant_result": str(assistant_message["content"]),
                "execution_records": (),
                "timestamp": None,
            }
        )
        state["next_turn_id"] = turn_id + 1
        new_turns += 1

    state["hot_messages"] = []
    state["compression_active"] = True
    status = (
        f"已压缩 {new_turns} 个新轮次；累计 {len(state['archived_turns'])} 个轮次。"
        "接下来的消息将使用 Temporal-Spine，之后仍可再次压缩新对话。"
    )
    return state, status, _memory_preview(state)


def reset_chat() -> tuple[Any, ...]:
    state = initial_chat_state()
    return (
        [],
        "",
        state,
        [],
        {},
        [],
        "会话已重置。",
        "",
        "",
        _memory_preview(state),
    )


def run_demo_conversation(
    chat_history: list[dict[str, Any]] | None,
    chat_state: dict[str, Any] | None,
    api_key: str,
    model_name: str,
    base_url: str,
    max_load_rounds: float,
    full_fallback_enabled: bool,
) -> Iterator[tuple[Any, ...]]:
    """Append a ten-turn demo by sending each message through the real agent."""

    history = copy.deepcopy(chat_history or [])
    state = copy.deepcopy(chat_state or initial_chat_state())
    total = len(DEMO_MESSAGES)
    for index, demo_message in enumerate(DEMO_MESSAGES, start=1):
        result = send_message(
            demo_message,
            history,
            state,
            api_key,
            model_name,
            base_url,
            max_load_rounds,
            full_fallback_enabled,
        )
        history = result[0]
        state = result[2]
        values = list(result)
        if result[1]:
            values[6] = f"Demo {index}/{total} 失败：{result[6]}"
            yield tuple(values)
            return
        values[6] = (
            f"Demo {index}/{total} 已完成。正在逐轮追加到当前会话；"
            "Demo 完成后可点击“压缩历史”。"
        )
        yield tuple(values)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Temporal-Spine Chat Agent") as demo:
        chat_state = gr.State(initial_chat_state())
        gr.Markdown(
            """
# Temporal-Spine × LangGraph Chat Agent

先像普通 Agent 一样连续聊天；点击**压缩历史**后，已完成轮次会转成 User-Spine 和
可加载原始组件。之后仍可继续聊天，新对话保留在 Hot Window，必要时可再次手动压缩。

> 点击发送后，当前模型输入会发送到配置的 OpenAI-compatible API。完整请求体会显示在
> 右侧，但 Authorization/API Key 不会显示或写入文件。
"""
        )

        with gr.Accordion("模型与运行设置", open=False):
            api_key = gr.Textbox(
                value="",
                type="password",
                label="API Key（当前进程已配置时可留空）",
            )
            model_name = gr.Textbox(value="deepseek-chat", label="模型名称")
            base_url = gr.Textbox(
                value="https://api.deepseek.com/chat/completions",
                label="Chat Completions URL",
            )
            max_load_rounds = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="最多记忆加载轮数",
            )
            full_fallback = gr.Checkbox(
                value=False,
                label="到达上限后自动加载全部剩余组件",
            )

        with gr.Row():
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    value=[],
                    label="连续对话",
                    height=540,
                    buttons=["copy", "copy_all"],
                )
                message = gr.Textbox(
                    label="输入消息",
                    placeholder="输入消息后按回车发送……",
                    lines=1,
                    max_lines=1,
                    autofocus=True,
                    submit_btn="发送",
                    elem_id="chat-input",
                )
                with gr.Row():
                    demo_button = gr.Button("一键运行 10 轮 Demo", variant="primary")
                    compress_button = gr.Button("压缩历史", variant="secondary")
                    reset_button = gr.Button("重置会话", variant="stop")
                status = gr.Textbox(
                    value="普通聊天模式：尚未压缩。",
                    label="运行状态",
                    interactive=False,
                )

            with gr.Column(scale=5):
                gr.Markdown(
                    "### 回答后的当前上下文快照\n"
                    "这里在本轮真实请求的 `messages` 后追加本次 `assistant` 输出，"
                    "所以最后一条是刚得到的回答。"
                )
                pretty_context = gr.Textbox(
                    value="",
                    label="回答后上下文（最后一条为 assistant）",
                    lines=38,
                    max_lines=50,
                    interactive=False,
                    buttons=["copy"],
                )
                memory_preview = gr.JSON(
                    value=_memory_preview(initial_chat_state()),
                    label="当前压缩记忆 / Hot Window",
                )
                loaded_ids = gr.JSON(label="本轮实际加载的组件 ID")
                metrics = gr.JSON(label="本轮调用与 token 审计")
                trace = gr.JSON(label="本轮运行 trace")
                with gr.Accordion("完全未改写的单行 RAW", open=False):
                    gr.Markdown(
                        "下面是与 HTTP `Request.data` 完全一致的 UTF-8 JSON 字符串。"
                        "它是生成回答前的真实输入，因此最后一条必然是 user 问题。"
                    )
                    raw_context = gr.Textbox(
                        value="",
                        label="最后一次调用的单行 RAW request body",
                        lines=8,
                        max_lines=12,
                        interactive=False,
                        buttons=["copy"],
                    )

        chat_inputs = [
            message,
            chatbot,
            chat_state,
            api_key,
            model_name,
            base_url,
            max_load_rounds,
            full_fallback,
        ]
        chat_outputs = [
            chatbot,
            message,
            chat_state,
            loaded_ids,
            metrics,
            trace,
            status,
            pretty_context,
            raw_context,
            memory_preview,
        ]
        message.submit(
            fn=send_message,
            inputs=chat_inputs,
            outputs=chat_outputs,
            api_name="chat",
            concurrency_id="chat-session",
            concurrency_limit=1,
        )
        demo_button.click(
            fn=run_demo_conversation,
            inputs=[
                chatbot,
                chat_state,
                api_key,
                model_name,
                base_url,
                max_load_rounds,
                full_fallback,
            ],
            outputs=chat_outputs,
            api_name="run_demo",
            concurrency_id="chat-session",
            concurrency_limit=1,
        )
        compress_button.click(
            fn=compress_history,
            inputs=[chat_state],
            outputs=[chat_state, status, memory_preview],
            api_name="compress",
        )
        reset_button.click(
            fn=reset_chat,
            inputs=[],
            outputs=chat_outputs,
            api_name="reset",
        )
    return demo


if __name__ == "__main__":
    build_demo().queue().launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=False,
        js=ENTER_TO_SEND_JS,
    )
