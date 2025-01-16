from openai import OpenAI
from getApiKey import get_key

client = OpenAI(
  api_key=get_key()
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message)