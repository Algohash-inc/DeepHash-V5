import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["ALGOHASH_BASE_URL"],
    api_key=os.environ["ALGOHASH_API_KEY"],
)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

resp = client.chat.completions.create(
    model="deephash-v5",
    messages=[{"role": "user", "content": "What is the weather in Tokyo right now?"}],
    tools=tools,
)
print(resp.choices[0].message.tool_calls)
