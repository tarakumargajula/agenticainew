# pip install boto3
import boto3
import json

# Create a Bedrock client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Model ID (Titan text generation)
model_id = "amazon.titan-text-lite-v1"

# Helper function to send prompt to Bedrock
def ask_bedrock(prompt: str) -> str:
    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 200,
            "temperature": 0.7,
            "topP": 0.9
        }
    }
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )
    output = json.loads(response["body"].read())
    return output["results"][0]["outputText"]

# Simple chatbot loop
print("Chatbot is ready! Type 'exit' or 'quit' to stop.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot ended.")
        break
    answer = ask_bedrock(user_input)
    print("Bot:", answer)
