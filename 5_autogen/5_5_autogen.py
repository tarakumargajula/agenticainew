import asyncio
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from dotenv import load_dotenv

load_dotenv(override=True)

# Create an OpenAI model client
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Create agents
primary_agent = AssistantAgent(
    name="primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

critic_agent = AssistantAgent(
    name="critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedback is addressed.",
)

# Termination condition
text_termination = TextMentionTermination("APPROVE")

# Create team
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

# --- Main async function ---
async def main():
    # Run the team task and print the result
    result = await team.run(task="Write a crisp note on text embedding")
    print("\nFinal Team Result:\n", result)

    # Close model client
    await model_client.close()

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
