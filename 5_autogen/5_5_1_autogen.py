import asyncio
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Setup model client ---
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# --- Setup agents ---
primary_agent = AssistantAgent(
    name="primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
    reflect_on_tool_use=True  # enable reflections/thoughts
)

critic_agent = AssistantAgent(
    name="critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedback is addressed.",
    reflect_on_tool_use=True
)

# --- Termination condition ---
text_termination = TextMentionTermination("APPROVE")

# --- Create team ---
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

# --- Main async function ---
async def main():
    print("=== Team Conversation Started ===\n")
    
    async for msg in team.run_stream(task="Write a crisp note on text embedding"):
        # Skip TaskResult objects
        from autogen_agentchat.base import TaskResult
        if isinstance(msg, TaskResult):
            continue  # or you could print the final result later
        
        # Safely get sender and content
        sender = getattr(msg, "sender", "Agent")
        content = getattr(msg, "content", str(msg))
        print(f"{sender}: {content}")
        
        # Print internal reflections if available
        inner_messages = getattr(msg, "inner_messages", None)
        if inner_messages:
            for inner in inner_messages:
                inner_sender = getattr(inner, "sender", "Agent")
                inner_content = getattr(inner, "content", str(inner))
                print(f"  [Reflection] {inner_sender}: {inner_content}")

    # After streaming finishes, get final TaskResult
    final_result = await team.run(task="Write a crisp note on text embedding")
    print("\n=== Final Task Result ===\n", final_result)

    await model_client.close()

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
