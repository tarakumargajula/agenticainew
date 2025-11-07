import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# Load API keys/config
load_dotenv(override=True)

# Initialize model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Define a simple tool the agent can call
async def get_forex_rate(target: str) -> float:
    """Get the forex rate of USD against a target currency."""
    return 90.10

# Create AssistantAgent
agent = AssistantAgent(
    name="forex_agent",
    model_client=model_client,
    tools=[get_forex_rate],
    system_message="You are a helpful assistant who loves giving fun facts!",
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# Create an interesting text message
text_message = TextMessage(
    content="Hello! Can you tell me an interesting fact about the USD to INR exchange rate today?", 
    source="User"
)

# Run the agent and stream the response to the console
async def main() -> None:
    # Use run_stream with the task
    await Console(agent.run_stream(task=text_message.content))
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
