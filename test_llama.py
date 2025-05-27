from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)