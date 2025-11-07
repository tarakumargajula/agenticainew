# Streaming

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

stream = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": "Say 'double bubble bath' slowly and repeat it 10 times.",
        },
    ],
    stream=True,
)

for event in stream:
    print(event)