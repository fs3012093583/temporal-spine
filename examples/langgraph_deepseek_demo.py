from __future__ import annotations

import os

from temporal_spine import HistoricalTurn, OpenAICompatibleClient, RunConfig
from temporal_spine.langgraph_adapter import LangGraphTemporalSpine


client = OpenAICompatibleClient(
    model=os.environ.get("MODEL_NAME", "deepseek-chat"),
    base_url=os.environ.get(
        "CHAT_COMPLETIONS_URL",
        "https://api.deepseek.com/chat/completions",
    ),
    api_key_env="DEEPSEEK_API_KEY",
    max_tokens=512,
)

# These turns represent older history after the first compression event.
compressed_history = [
    HistoricalTurn(
        0,
        "I am allergic to peanuts.",
        "Understood. I will avoid peanuts in future suggestions.",
    ),
    HistoricalTurn(
        1,
        "Suggest a quick breakfast.",
        "Try peanut-butter toast, yogurt with fruit, or oatmeal.",
    ),
]

result = LangGraphTemporalSpine(
    client,
    RunConfig(max_load_rounds=2, full_fallback_enabled=False),
).run(
    compressed_history,
    "Which earlier breakfast options are safe for me, and why?",
)

print(result.answer)
print("loaded:", result.loaded_action_ids)
print("model calls:", len(result.calls))
