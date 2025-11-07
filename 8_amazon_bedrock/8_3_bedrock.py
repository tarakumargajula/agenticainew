# pip install boto3
import boto3
import json

# Create a Bedrock client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Model ID (Titan text generation)
model_id = "amazon.titan-text-lite-v1"

# Chat history (list of dicts)
chat_history = []

# Helper function to send prompt to Bedrock with history
def ask_bedrock(user_input: str) -> str:
    # Build conversation context
    history_text = ""
    for turn in chat_history:
        history_text += f"You: {turn['user']}\nAI: {turn['bot']}\n"
    
    # Current query
    full_prompt = history_text + f"You: {user_input}\nBot:"
    
    payload = {
        "inputText": full_prompt,
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
    ai_answer = output["results"][0]["outputText"]
    
    # Save to history
    chat_history.append({"user": user_input, "bot": ai_answer})
    return ai_answer

# Chatbot loop
print("Chatbot is ready! Type 'exit' or 'quit' to stop.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot ended.")
        break
    answer = ask_bedrock(user_input)
    print("Bot: ", answer)
