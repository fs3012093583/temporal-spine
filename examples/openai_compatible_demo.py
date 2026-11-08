import os

from temporal_spine import HistoricalTurn, OpenAICompatibleClient, RunConfig, TemporalSpine


client = OpenAICompatibleClient(
    model=os.environ["MEMADDR_MODEL"],
    base_url=os.environ["MEMADDR_BASE_URL"],
    api_key_env="OPENAI_API_KEY",
    max_tokens=8192,
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
