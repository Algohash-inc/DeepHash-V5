import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["ALGOHASH_BASE_URL"],
    api_key=os.environ["ALGOHASH_API_KEY"],
)

resp = client.chat.completions.create(
    model="deephash-v5",
    messages=[{"role": "user", "content": "17 × 23 = ?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.content)
