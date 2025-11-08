# customer_service_agents_handoff_fixed.py
# pip install openai sentence-transformers python-dotenv pandas scikit-learn

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from agents import Agent, Runner, function_tool, handoff
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from typing import Any

load_dotenv(override=True)
client = OpenAI()

# -------------------------------------------------
# Paths and model
# -------------------------------------------------
CSV_PATH = "c://code//agenticai//2_openai_agents//customer_service_training_data.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOG_FILE = "c://code//agenticai//2_openai_agents//agent_logs.txt"

model = SentenceTransformer(MODEL_NAME)

# -------------------------------------------------
# Load data + create embeddings
# -------------------------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Training data not found at: {CSV_PATH}")

print("Loading training data and generating embeddings...")
df = pd.read_csv(CSV_PATH)

# Expect columns: flags, utterance, category, intent
utterances = df["utterance"].astype(str).tolist()
intents = df["intent"].astype(str).tolist()
categories = df["category"].astype(str).tolist()
utterance_embeddings = model.encode(utterances, convert_to_numpy=True)

# -------------------------------------------------
# Logging utilities
# -------------------------------------------------
def log_issue(issue_type: str, query: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] New {issue_type} issue raised: {query}\n")

def log_agent(agent_name: str, query: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {agent_name} handled query: {query}\n")

# -------------------------------------------------
# Semantic intent classifier
# -------------------------------------------------
def classify_intent_semantic(user_query: str):
    query_emb = model.encode([user_query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, utterance_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    return {
        "intent": intents[best_idx],
        "category": categories[best_idx],
        "matched_utterance": utterances[best_idx],
        "similarity": round(float(similarities[best_idx]), 3),
    }

# -------------------------------------------------
# Specialized agent tools
# -------------------------------------------------
@function_tool
async def handle_account(query: str):
    log_issue("account", query)
    log_agent("Account Agent", query)
    return "Your request to close or manage your account is being handled."

@function_tool
async def handle_order(query: str):
    log_issue("order", query)
    log_agent("Order Agent", query)
    return "Your order-related request is being processed."

@function_tool
async def handle_delivery(query: str):
    log_issue("delivery", query)
    log_agent("Delivery Agent", query)
    return "Your delivery inquiry is being handled."

@function_tool
async def handle_feedback(query: str):
    log_issue("feedback", query)
    log_agent("Feedback Agent", query)
    return "Thank you for your feedback. It will be shared with the right team."

@function_tool
async def handle_payment(query: str):
    log_issue("payment", query)
    log_agent("Payment Agent", query)
    return "Your payment-related request is being handled."

# -------------------------------------------------
# Agent definitions
# -------------------------------------------------
account_agent = Agent(
    name="Account Agent",
    instructions="Handle account management and profile issues.",
    tools=[handle_account],
)

order_agent = Agent(
    name="Order Agent",
    instructions="Handle customer order and purchase issues.",
    tools=[handle_order],
)

delivery_agent = Agent(
    name="Delivery Agent",
    instructions="Handle shipping and delivery inquiries.",
    tools=[handle_delivery],
)

feedback_agent = Agent(
    name="Feedback Agent",
    instructions="Handle customer feedback, reviews, and complaints.",
    tools=[handle_feedback],
)

payment_agent = Agent(
    name="Payment Agent",
    instructions="Handle billing, payments, and refunds.",
    tools=[handle_payment],
)

# Helper mapping for fallback routing
AGENT_MAP = {
    "account": account_agent,
    "order": order_agent,
    "delivery": delivery_agent,
    "payment": payment_agent,
    "feedback": feedback_agent,
}

# -------------------------------------------------
# Triage tool using LLM + embeddings + handoff
# -------------------------------------------------
@function_tool
async def triage_logic(query: str):
    result = classify_intent_semantic(query)

    route_prompt = f"""
    A customer asked: "{query}"
    The embedding model classified this as:
    Intent: {result['intent']}
    Category: {result['category']}
    Similarity: {result['similarity']}

    Which team should handle this — Account, Order, Delivery, Feedback, or Payment?
    Reply with exactly one word: Account, Order, Delivery, Feedback, or Payment.
    """

    llm_response = client.responses.create(
        model="gpt-4o-mini",
        input=route_prompt,
        temperature=0
    )

    try:
        route_text = llm_response.output_text.strip().lower()
    except AttributeError:
        route_text = ""
        for item in llm_response.output[0].content:
            if hasattr(item, "text"):
                route_text += item.text
        route_text = route_text.strip().lower()

    # Prefer to return a handoff directive when we can detect a clear team
    if "account" in route_text:
        print("[Router] → Handoff to Account Agent")
        return handoff(to=account_agent, input=query)
    elif "order" in route_text:
        print("[Router] → Handoff to Order Agent")
        return handoff(to=order_agent, input=query)
    elif "delivery" in route_text or "ship" in result["category"].lower():
        print("[Router] → Handoff to Delivery Agent")
        return handoff(to=delivery_agent, input=query)
    elif "payment" in route_text or "refund" in result["category"].lower():
        print("[Router] → Handoff to Payment Agent")
        return handoff(to=payment_agent, input=query)
    elif "feedback" in route_text or "review" in result["category"].lower():
        print("[Router] → Handoff to Feedback Agent")
        return handoff(to=feedback_agent, input=query)
    else:
        # If the LLM didn't give a clear team, return a plain routing sentence (SDK may provide this).
        # We keep it simple and return a user-facing sentence; the caller will fallback to embeddings.
        return f"Could not determine exact team. Semantic category: {result['category']}"

# -------------------------------------------------
# Triage Agent definition
# -------------------------------------------------
triage_agent = Agent(
    name="Triage Agent",
    instructions="Decide which specialized agent should handle a given customer query using handoff().",
    tools=[triage_logic],
    handoffs=[account_agent, order_agent, delivery_agent, feedback_agent, payment_agent],
)

# -------------------------------------------------
# Utility to examine possible SDK handoff structure
# -------------------------------------------------
def extract_handoff_from_result(result_obj: Any):
    """
    The SDK may return a dict-like handoff directive or an object. Detect common patterns:
    - dict with 'handoff' key
    - object with attribute 'handoff'
    - fallback to None
    """
    if isinstance(result_obj, dict) and "handoff" in result_obj:
        return result_obj["handoff"]
    # try attribute-style
    if hasattr(result_obj, "handoff"):
        return getattr(result_obj, "handoff")
    return None

# -------------------------------------------------
# Orchestration with robust handoff execution
# -------------------------------------------------
async def chat_with_customer(query: str):
    # Run triage
    triage_session = await Runner.run(triage_agent, query)
    triage_result = triage_session.final_output

    # 1) If triage returned an SDK-style handoff directive, execute it
    handoff_directive = extract_handoff_from_result(triage_result)
    if handoff_directive:
        # Expect directive to contain 'to' (Agent) and 'input' (text)
        target_agent = None
        input_text = query
        # directive might be a dict or object; handle common shapes
        if isinstance(handoff_directive, dict):
            target_agent = handoff_directive.get("to") or handoff_directive.get("agent") or None
            input_text = handoff_directive.get("input", query)
        else:
            # attribute access
            target_agent = getattr(handoff_directive, "to", None) or getattr(handoff_directive, "agent", None)
            input_text = getattr(handoff_directive, "input", query)

        if isinstance(target_agent, Agent):
            print(f"[Router] → Executing {target_agent.name}")
            response_session = await Runner.run(target_agent, input_text)
            return response_session.final_output
        # if the directive didn't contain an Agent object, continue to fallback routing

    # 2) If triage_result is plain text, try substring routing
    if isinstance(triage_result, str):
        triage_text = triage_result.strip().lower()
        # Try to detect a team name mentioned in text
        for key, agent in AGENT_MAP.items():
            if key in triage_text:
                print(f"[Router] → Routed to: {agent.name}")
                response_session = await Runner.run(agent, query)
                return response_session.final_output

    # 3) Fallback to semantic classifier (guarantees we route somewhere)
    sem = classify_intent_semantic(query)
    cat = sem.get("category", "").lower()
    for key, agent in AGENT_MAP.items():
        if key in cat:
            print(f"[Router] → Fallback routed to: {agent.name} (semantic category: {cat})")
            response_session = await Runner.run(agent, query)
            return response_session.final_output

    # 4) Last-resort: feedback agent
    print("[Router] → Fallback routed to: Feedback Agent")
    response_session = await Runner.run(feedback_agent, query)
    return response_session.final_output

# -------------------------------------------------
# Interactive loop
# -------------------------------------------------
if __name__ == "__main__":
    async def main():
        print("Customer Service Chatbot (type 'exit' to quit)\n")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Chatbot: Goodbye!")
                break

            response = await chat_with_customer(user_input)
            # response may be an object or string depending on Runner; convert to str
            if hasattr(response, "output_text"):
                output_text = response.output_text
            else:
                output_text = str(response)
            print(f"Chatbot: {output_text}\n")

    asyncio.run(main())
