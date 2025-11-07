import asyncio
import pandas as pd
from typing import Optional
from pydantic import BaseModel
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from dotenv import load_dotenv

# Define structured output model
class JobInfo(BaseModel):
    company_name: Optional[str]
    role: Optional[str]
    location: Optional[str]

df = pd.read_csv("c:\\code\\agenticai\\5_autogen\\clean_jobs.csv")

# Take 5th, 105th, 205th, 305th, 405th rows
indices = [4, 104, 204, 304, 404]
descriptions = df.loc[indices, "description"].tolist()

# Step 3: Setup model client
load_dotenv(override=True)
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Step 4: Create AssistantAgent with structured output
agent = AssistantAgent(
    name="job_extractor",
    model_client=model_client,
    system_message="Extract company name, role, and location from the job description in JSON format.",
    output_content_type=JobInfo,
    model_client_stream=True,
)

async def main():
    # Step 5: Process each selected job description
    for i, desc in enumerate(descriptions, start=1):
        print(f"\n--- Job {i} ---")

        # Stream structured output
        async for message in agent.run_stream(task=desc):
            if isinstance(message, StructuredMessage) and isinstance(message.content, JobInfo):
                job_info: JobInfo = message.content
                print(f"\rCompany: {job_info.company_name or 'N/A'} | "
                      f"Role: {job_info.role or 'N/A'} | "
                      f"Location: {job_info.location or 'N/A'}", end="")

        print()  # Newline after each job

    # Step 6: Close model client
    await model_client.close()
    print("\nAll selected jobs processed. Model client closed.")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
