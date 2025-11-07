# ------------------------------------------------------------
# ReAct Agent using Claude + Chroma + SerpAPI + ntfy + Guardrails
# ------------------------------------------------------------

import os
import re
import requests
import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic

# ---- Load environment variables ----
load_dotenv()

# ---- Blocked words guardrail ----
BLOCKED_WORDS = ['hack', 'exploit', 'illegal', 'bomb', 'violence', 'malware', 'virus']

def has_blocked_words(text):
    """Check if text contains blocked words"""
    text_lower = text.lower()
    for word in BLOCKED_WORDS:
        if word in text_lower:
            return True, word
    return False, None


# ---- Setup Vector DB ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Chroma (persistent vector store)
vectordb = Chroma(
    persist_directory="c:/code/agenticai/3_langgraph/product_embeddings_chroma",
    embedding_function=embeddings,
)

# ---- Setup LLM ----
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    streaming=True,
)


# ---- Tools ----
def tool_search(query: str) -> str:
    """Searches in Chroma DB using semantic similarity."""
    results = vectordb.similarity_search(query, k=2)
    return ", ".join([doc.metadata.get("title", "Untitled") for doc in results]) or "No results found."


def tool_serp(query: str) -> str:
    """Performs online search using SerpAPI."""
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": os.getenv("SERPAPI_API_KEY"), "num": 2}
    try:
        data = requests.get(url, params=params).json()
        results = [f"{r['title']}: {r['snippet']}" for r in data.get("organic_results", [])[:2]]
        return "\n".join(results) if results else "No SERP results found."
    except Exception as e:
        return f"SerpAPI error: {e}"


def tool_ntfy(message: str) -> str:
    """Sends a short notification via ntfy.sh."""
    ntfy_topic = os.getenv("NTFY_TOPIC")
    ntfy_url = f"https://ntfy.sh/{ntfy_topic}"
    try:
        response = requests.post(ntfy_url, data=message, timeout=5)
        response.raise_for_status()
        return "Notification sent successfully."
    except requests.exceptions.RequestException as e:
        return f"Failed to send notification: {e}"


# ---- Agent Logic ----
def react_agent(query: str):
    """ReAct-style reasoning loop."""
    blocked, word = has_blocked_words(query)
    if blocked:
        yield f"Cannot process queries containing '{word}'", f"Blocked word: {word}"
        return
    
    state = {"query": query, "history": [f"User: {query}"], "final": ""}

    while True:
        prompt = f"""
You are a ReAct-style agent. 
You MUST always follow this exact output format:

Thought: (one short sentence of reasoning)
Action: (exactly one of the following)
- Search[some query]
- SerpSearch[some query]
- Ntfy[some message]
- Finalize[some final answer]

You should use the Ntfy tool to send a notification whenever the user asks 
about the 'latest iPhone' or related queries like 'new iPhone', 'iPhone 16', etc. 
The message for the notification should be concise and directly related to the user's query, 
for example, "User inquired about latest iPhone." 
After sending the notification, continue to answer the user's question.

Do NOT output anything else. 
Do NOT answer directly unless using Finalize[].

Example:
Thought: I should look up reviews for the product.
Action: SerpSearch[best headphones reviews]

Conversation so far:
{chr(10).join(state['history'])}

User question: {state['query']}
Now continue.
"""

        # Stream LLM output
        response = ""
        for chunk in llm.stream(prompt):
            if chunk.content:
                response += chunk.content
        response = response.strip()
        state["history"].append(response)

        # Parse model action
        action_match = re.search(r"Action\s*:\s*(\w+)\s*\[(.*)\]", response)
        if not action_match:
            break  # stop if Claude does not follow format

        action, arg = action_match.group(1).lower(), action_match.group(2).strip()

        # Execute the appropriate action
        if action == "search":
            obs = tool_search(arg)
            state["history"].append(f"Observation: {obs}")

        elif action == "serpsearch":
            obs = tool_serp(arg)
            state["history"].append(f"Observation: {obs}")

        elif action == "ntfy":
            obs = tool_ntfy(arg)
            state["history"].append(f"Observation: {obs}")

        elif action == "finalize":
            state["final"] = arg
            yield state["final"], "\n".join(state["history"])
            break

        yield None, "\n".join(state["history"])


# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("# ReAct Agent with Guardrails + Chroma + ntfy + SerpAPI")

    chatbot = gr.Chatbot(label="Agent Trace")
    query = gr.Textbox(label="Ask something", placeholder="e.g. latest iphone news")

    def respond(user_input, chat_history):
        chat_history.append(("User: " + user_input, ""))

        for final, trace in react_agent(user_input):
            if final:  # final answer
                chat_history[-1] = (
                    chat_history[-1][0],
                    f"**Final Answer:** {final}\n\n---\n**Trace:**\n{trace}"
                )
                yield chat_history
            else:  # intermediate progress
                chat_history[-1] = (
                    chat_history[-1][0],
                    f"Working...\n\n**Trace so far:**\n{trace}"
                )
                yield chat_history

    query.submit(respond, [query, chatbot], [chatbot])

if __name__ == "__main__":
    demo.launch()
