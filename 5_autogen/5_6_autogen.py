import os
import asyncio
import yfinance as yf
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination

load_dotenv(override=True)

# Correct environment variable
api_key = os.getenv('SERPAPI_KEY')

# --- Async stock price fetching tool ---
async def fetch_stock_price(ticker: str, start_date: str, end_date: str) -> str:
    print(f"[StockFetchAgent] Fetching {ticker} prices from {start_date} to {end_date}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    
    if hist.empty:
        print(f"[StockFetchAgent] No data found for {ticker} between {start_date} and {end_date}.")
        return f"Failed to retrieve data for {ticker}."
    
    # Take first and last available closing prices
    start_price = hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]
    print(f"[StockFetchAgent] Start Price: {start_price}, End Price: {end_price}")
    return f"Start Price: {start_price}, End Price: {end_price}"

# --- OpenAI client ---
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# --- Agents ---
planning_agent = AssistantAgent(
    "PlanningAgent",
    model_client=model_client,
    system_message="You are a planning agent. Break down tasks into subtasks and delegate to other agents."
)

stock_fetch_agent = AssistantAgent(
    "StockFetchAgent",
    model_client=model_client,
    tools=[fetch_stock_price],
    system_message="You are a stock fetching agent. Retrieve stock prices using your tool."
)

data_analysis_agent = AssistantAgent(
    "DataAnalysisAgent",
    model_client=model_client,
    system_message="You are a data analyst. Compute percentage changes and analyze stock data provided."
)

# --- Termination ---
max_messages_termination = MaxMessageTermination(max_messages=10)
text_termination = TextMentionTermination("TERMINATE")
termination = text_termination | max_messages_termination

selector_prompt = """Select an agent to perform the next task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Only select one agent.
"""

team = SelectorGroupChat(
    [planning_agent, stock_fetch_agent, data_analysis_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
)

# --- Task ---
task = "Analyze the stock price change of Apple Inc. (AAPL) from September 1, 2025, to September 25, 2025."

# --- Main ---
async def main():
    print("=== Team Task Started ===\n")
    async for msg in team.run_stream(task=task):
        sender = getattr(msg, "sender", "Agent")
        content = getattr(msg, "content", str(msg))
        print(f"{sender}: {content}")
        # Print inner reflections if any
        inner_messages = getattr(msg, "inner_messages", None)
        if inner_messages:
            for inner in inner_messages:
                inner_sender = getattr(inner, "sender", "Agent")
                inner_content = getattr(inner, "content", str(inner))
                print(f"  [Reflection] {inner_sender}: {inner_content}")

    print("\n=== Team Task Finished ===")
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
