from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

from .client import OpenAICompatibleClient, ProviderError
from .memory import HistoricalTurn
from .model import ModelResponse, ToolCall
from .runner import ProtocolError, RunConfig, RunResult, TemporalSpine


def load_history(path: str | Path) -> list[HistoricalTurn]:
    """Read UTF-8 JSON; ids supplied by the application are never renumbered."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"turns"}:
        raise ValueError('history must be a JSON object with one "turns" array')
    if not isinstance(payload["turns"], list):
        raise ValueError('"turns" must be an array')
    turns = []
    allowed = {"turn_id", "user_request", "assistant_result", "timestamp", "execution_records"}
    for index, row in enumerate(payload["turns"]):
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError(f"turn {index} has unknown fields or is not an object")
        if not {"turn_id", "user_request"} <= set(row):
            raise ValueError(f"turn {index} requires turn_id and user_request")
        turns.append(HistoricalTurn(**row))
    return turns


def result_summary(result: RunResult) -> dict[str, Any]:
    # Full restored content is deliberately not included in CLI output.
    return {
        "answer": result.answer,
        "loaded_action_ids": list(result.loaded_action_ids),
        "dropped_action_ids": list(result.dropped_action_ids),
        "access_calls": sum(event["event"] == "components_loaded" for event in result.trace),
        "model_calls": len(result.calls),
        "invalid_action_count": result.invalid_action_count,
        "protocol_followed": result.protocol_followed,
        "submission_mode": result.submission_mode,
        "usage_by_call": [dict(call.usage) for call in result.calls],
    }


def offline_demo() -> RunResult:
    """Synthetic demonstration, not an evaluation or a model-quality result."""
    class ScriptedModel:
        fixed_parameters = {"model": "offline-scripted", "temperature": 0}

        def __init__(self) -> None:
            self.responses = deque([
                ModelResponse(tool_calls=(ToolCall("demo-load", "load_components", {
                    "selection": "selected", "action_ids": ["result:1"],
                }),)),
                ModelResponse(tool_calls=(ToolCall("demo-submit", "submit_final_result", {
                    "result": "I previously suggested lentil curry for dinner.",
                }),)),
            ])

        def complete(self, messages, tools) -> ModelResponse:
            return self.responses.popleft()

    return TemporalSpine(ScriptedModel()).run([
        HistoricalTurn(0, "I prefer vegetarian meals.", "I will remember that preference."),
        HistoricalTurn(1, "Give me a dinner idea.", "Try lentil curry."),
    ], "What dinner did you previously suggest?")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MemAddr: chronological User-Spine + bounded address access")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="run a synthetic offline example (no API key or network)")
    run = commands.add_parser("run", help="answer a question using your history and a tool-capable API")
    run.add_argument("--history", required=True, help="UTF-8 JSON history; see examples/history.json")
    run.add_argument("--question", required=True)
    run.add_argument("--current-date", help="question date, retained in the model context")
    run.add_argument("--model", default=os.getenv("MEMADDR_MODEL"))
    run.add_argument("--base-url", default=os.getenv("MEMADDR_BASE_URL"))
    run.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable name, NOT the key")
    run.add_argument("--access-rounds", type=int, choices=(0, 1, 2), default=1)
    run.add_argument("--batch-size", type=int, choices=range(1, 6), default=5)
    run.add_argument("--max-model-calls", type=int, default=3)
    run.add_argument("--max-tokens", type=int, default=8192, help="per-request output limit, including reasoning if applicable")
    run.add_argument("--token-parameter", choices=("max_tokens", "max_completion_tokens"), default="max_tokens")
    run.add_argument("--temperature", default="0", help='number or "omit" for APIs without this setting')
    run.add_argument("--thinking-json", help='optional provider-specific thinking object, e.g. {"type":"disabled"}')
    run.add_argument("--timeout", type=float, default=120)
    run.add_argument("--strict-tools", action="store_true", help="reject ordinary-text answers instead of audited capture")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = offline_demo()
        else:
            if not args.model or not args.base_url:
                parser.error("run requires --model and --base-url (or MEMADDR_MODEL and MEMADDR_BASE_URL)")
            thinking = json.loads(args.thinking_json) if args.thinking_json else None
            if thinking is not None and not isinstance(thinking, dict):
                raise ValueError("--thinking-json must be a JSON object")
            history = load_history(args.history)
            config = RunConfig(
                max_access_calls=args.access_rounds, max_model_calls=args.max_model_calls,
                max_components_per_load=args.batch_size, capture_natural_final=not args.strict_tools,
            )
            client = OpenAICompatibleClient(
                model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                max_tokens=args.max_tokens, token_parameter=args.token_parameter,
                temperature=None if args.temperature == "omit" else float(args.temperature),
                timeout=args.timeout, thinking=thinking,
            )
            result = TemporalSpine(client, config).run(history, args.question, current_date=args.current_date)
        print(json.dumps(result_summary(result), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, ProviderError, ProtocolError) as exc:
        print(f"memaddr: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
