"""Run the standalone OpenHands + Temporal-Spine integration.

Safe default:
    python examples/openhands_temporal_spine_demo.py --test-mode

Live coding-agent mode (requires the optional dependency and a model key):
    python examples/openhands_temporal_spine_demo.py --live \
        --api-key-env DEEPSEEK_API_KEY
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from temporal_spine.openhands_adapter import OpenHandsTemporalSpine, run_offline_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test-mode", action="store_true", help="Run without OpenHands or an API call.")
    mode.add_argument("--live", action="store_true", help="Run a real OpenHands coding conversation.")
    parser.add_argument(
        "--log-dir",
        default=".openhands-temporal-spine/test-run",
        help="Directory for events.jsonl, raw_context_*.json, and latest_raw_context.json.",
    )
    parser.add_argument("--workspace", default=".", help="Workspace OpenHands may inspect and edit.")
    parser.add_argument(
        "--task",
        default=(
            "Inspect this repository. Explain how the Temporal-Spine memory runtime works, "
            "then make no file changes and report which files you inspected."
        ),
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek/deepseek-chat"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-events", type=int, default=24)
    parser.add_argument("--keep-recent-events", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.live:
        result = run_offline_test(args.log_dir, print_context=True)
        print(f"Offline test logs: {Path(result.log_dir).resolve()}")
        print(f"Latest raw context: {result.latest_context.resolve()}")
        return

    adapter = OpenHandsTemporalSpine(
        log_dir=args.log_dir,
        max_events=args.max_events,
        keep_recent_events=args.keep_recent_events,
    )
    session = adapter.create_conversation(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        workspace=args.workspace,
    )
    try:
        session.send(args.task)
        print(f"OpenHands conversation finished. Logs: {Path(args.log_dir).resolve()}")
        print(f"Latest model-visible context: {session.latest_raw_context.resolve()}")
        print(f"Memory state: {adapter.memory.snapshot()}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
