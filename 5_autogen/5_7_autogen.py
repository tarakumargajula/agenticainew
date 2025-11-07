import asyncio
import requests
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
from dotenv import load_dotenv

load_dotenv(override=True)

# MCP Tool Function
def get_cryptocurrency_price(crypto: str) -> str:
    """Gets the price of a cryptocurrency."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": crypto.lower(), "vs_currencies": "inr"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = data.get(crypto.lower(), {}).get("inr")
        if price is not None:
            return f"The price of {crypto} is Rs {price}."
        else:
            return f"Price for {crypto} not found."
    except Exception as e:
        return f"Error fetching price for {crypto}: {e}"

async def ask_question(client, question):
    result = await client.create([
        UserMessage(content=question, source="user")
    ])
    print(f"Question: {question}\nAnswer: {result.content}\n")

async def main():
    client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    # Get crypto name from user
    crypto_name = input("Enter cryptocurrency name (e.g., bitcoin, ethereum): ")
    crypto_price = get_cryptocurrency_price(crypto_name)
    print(f"Crypto Price: {crypto_price}\n")
    
    # Regular AI questions
    tasks = []
    
    await asyncio.gather(*tasks)
    await client.close()

asyncio.run(main())