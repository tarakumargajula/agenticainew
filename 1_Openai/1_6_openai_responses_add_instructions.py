from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="Speak like Sherlock Holmes.",
    input="Are semicolons optional in JavaScript?",
)

print(response.output_text)