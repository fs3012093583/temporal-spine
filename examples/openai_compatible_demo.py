import os

from temporal_spine import HistoricalTurn, OpenAICompatibleClient, RunConfig, TemporalSpine


client = OpenAICompatibleClient(
    model=os.environ.get("MODEL_NAME", "deepseek-chat"),
    base_url=os.environ.get(
        "CHAT_COMPLETIONS_URL", "https://api.deepseek.com/chat/completions"
    ),
    api_key_env=os.environ.get("API_KEY_ENV", "DEEPSEEK_API_KEY"),
    max_tokens=512,
)

history = [
    HistoricalTurn(0, "I prefer vegetarian meals.", "I will remember that preference."),
    HistoricalTurn(1, "Give me a dinner idea.", "A lentil curry would be a good option."),
]

result = TemporalSpine(client, RunConfig()).run(
    history,
    "What dinner did you previously suggest?",
)
print(result.answer)
print(result.loaded_action_ids)
print(result.protocol_followed)
