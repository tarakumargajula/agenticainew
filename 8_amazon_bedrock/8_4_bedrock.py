# pip install boto3
import boto3
import json
import re

# --------------------------
# AWS Bedrock Setup
# --------------------------
client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "amazon.titan-text-lite-v1"

# --------------------------
# Chat history
# --------------------------
chat_history = []

# --------------------------
# Guardrails
# --------------------------
def check_guardrails(user_input: str) -> str | None:
    """
    Check input against basic guardrails.
    Return an error message if input is blocked, else None.
    """
    blocked_words = ["hack", "terrorist", "bomb", "kill"]   # extend as needed
    max_length = 300  # prevent prompt injection with overly long input

    # Block malicious words
    for word in blocked_words:
        if re.search(rf"\b{word}\b", user_input.lower()):
            return f"Sorry, I cannot discuss topics related to '{word}'."

    # Block too long inputs
    if len(user_input) > max_length:
        return "Your input is too long. Please shorten your question."

    return None


# --------------------------
# Ask Bedrock with history
# --------------------------
def ask_bedrock(user_input: str) -> str:
    # Guardrail check first
    violation = check_guardrails(user_input)
    if violation:
        return violation

    # Build conversation history
    history_text = ""
    for turn in chat_history[-5:]:  # keep last 5 turns
        history_text += f"You: {turn['user']}\nAI: {turn['ai']}\n"

    full_prompt = history_text + f"You: {user_input}\nAI:"

    payload = {
        "inputText": full_prompt,
        "textGenerationConfig": {
            "maxTokenCount": 200, # maximum number of tokens to be generated
            "temperature": 0.7, # higher = more random
            "topP": 0.9 # means sample from top 90%, i.e. less random
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
    chat_history.append({"user": user_input, "ai": ai_answer})
    return ai_answer


# --------------------------
# Chatbot Loop
# --------------------------
print("Chatbot with Guardrails is ready! Type 'exit' or 'quit' to stop.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot ended.")
        break
    answer = ask_bedrock(user_input)
    print("AI:", answer)
