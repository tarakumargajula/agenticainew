import os
import json
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------- Setup ----------------
load_dotenv(override=True)

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
)

# ---------------- MCP Utility ----------------
async def call_mcp(server_file, tool_name, args):
    try:
        server_params = StdioServerParameters(command="python", args=[server_file])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool = next(t for t in tools if t.name == tool_name)
                result = await session.call_tool(tool.name, args)
                return json.loads(result.content[0].text)
    except Exception as e:
        print(f"[Error] MCP call failed: {e}")
        return {}

# ---------------- Tool Wrappers ----------------
async def get_city_from_ip(ip: str) -> str:
    data = await call_mcp("ip2location_mcp_server.py", "get_ip_info", {"ip": ip})
    return data.get("city_name", "Unknown City")

async def get_weather(city: str) -> str:
    data = await call_mcp("weather_mcp_server.py", "get_weather_info", {"location": city})
    return str(data.get("current", {}).get("temp_c", "N/A"))

async def get_news(city: str):
    data = await call_mcp("news_mcp_server.py", "get_search_results", {"location": city})
    try:
        articles = json.loads(data.get("result", "{}")).get("articles", [])
        return [a.get("title") for a in articles]
    except Exception:
        return []

# ---------------- Agents ----------------
planner = AssistantAgent(
    name="Planner",
    model_client=model_client,
    system_message="You are the Planner. Collect results from agents and summarize clearly."
)

ip_agent = AssistantAgent(
    name="IPAgent",
    model_client=model_client,
    tools=[get_city_from_ip],
    system_message="Resolve IPs into city names using your tool."
)

weather_agent = AssistantAgent(
    name="WeatherAgent",
    model_client=model_client,
    tools=[get_weather],
    system_message="Get the temperature for a city using your tool."
)

news_agent = AssistantAgent(
    name="NewsAgent",
    model_client=model_client,
    tools=[get_news],
    system_message="Get the latest news headlines for a city using your tool."
)

# ---------------- Swarm Team ----------------
termination = MaxMessageTermination(10)

research_team = Swarm(
    participants=[planner, ip_agent, weather_agent, news_agent],
    termination_condition=termination,
)

# ---------------- Run ----------------
async def main():
    ip = input("Enter the IP address: ").strip()
    task = f"Workflow: Resolve IP {ip} → city → get temperature → get news → summarize results."

    print("\n=== Swarm Starting ===\n")
    async for msg in research_team.run_stream(task=task):
        sender = getattr(msg, "sender", "Agent")
        content = getattr(msg, "content", str(msg))
        print(f"{sender}: {content}")
        # Print any inner messages/reflections
        for inner in getattr(msg, "inner_messages", []):
            print(f"  [Reflection] {inner.sender}: {inner.content}")

    print("\n=== Swarm Finished ===")
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
