# pip install boto3
import boto3
import json

# ----------------------------
# Bedrock Setup
# ----------------------------
client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "amazon.titan-text-lite-v1"


# ----------------------------
# Function: Streaming Ask
# ----------------------------
def ask_bedrock_stream(prompt: str):
    response = client.invoke_model_with_response_stream(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 200,
                "temperature": 0.7,
                "topP": 0.9
            }
        })
    )

    # Collect streamed output
    full_response = ""
    print("AI: ", end="", flush=True)

    for event in response["body"]:
        chunk = event.get("chunk")
        if chunk:
            data = json.loads(chunk.get("bytes").decode("utf-8"))
            text = data.get("outputText", "")
            full_response += text
            print(text, end="", flush=True)  # stream print

    print()  # newline after response
    return full_response


# ----------------------------
# Chat Loop
# ----------------------------
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        ask_bedrock_stream(user_input)
