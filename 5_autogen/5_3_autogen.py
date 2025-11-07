import asyncio
from PIL import Image
from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image as AGImage
from autogen_agentchat.agents import AssistantAgent

load_dotenv(override=True)

# Load the image
image_path = "c://code//agenticai//5_autogen//1911_Solvay_conference.jpg"
pil_image = Image.open(image_path)
img = AGImage(pil_image)
print("Image loaded successfully.")

# Setup model client
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Create Image Agent
image_agent = AssistantAgent(
    name="image_agent",
    model_client=model_client
)

# Create MultiModal message for description
description_message = MultiModalMessage(
    content=["Can you describe the content of this image?", img],
    source="User"
)

async def main():
    # Send description message and get response
    description_response = await image_agent.on_messages([description_message], cancellation_token=None)
    description_text = description_response.chat_message.content
    print("\nImage Agent finished processing.")
    print("Image Agent's response:")
    print(description_text)

    # Close agent and model client
    await image_agent.close()
    await model_client.close()
    print("\nAgent and model client closed. Done.")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
