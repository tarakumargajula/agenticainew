# Use house prices chromadb to respond to users' questions 
# via a langgraph agent
# pip install langgraph openai chromadb sentence-transformers python-dotenv

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
import time

# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv(override=True)


# ------------------------------
# State
# ------------------------------

class AgentState(TypedDict):
    """Holds all message exchanges and latest result"""
    messages: Annotated[list, "Conversation messages"]
    query: Annotated[str, "User query"]
    retrieved_docs: Annotated[str, "Results from ChromaDB search"]
    answer: Annotated[str, "Final LLM-generated answer"]


# ------------------------------
# ChromaDB Setup (already created)
# ------------------------------

client = chromadb.PersistentClient(path="c://code//agenticai//3_langgraph//chroma")
collection = client.get_collection("house_prices")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------
# Search Function (no @tool)
# ------------------------------

def search_house_prices(query: str) -> str:
    """Search house prices database using semantic similarity"""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)

    # Filter by cosine similarity threshold (ChromaDB returns distances, not similarities)
    # For cosine distance: similarity = 1 - distance
    # We want high similarity (low distance), so threshold on distance < 0.5
    # which means similarity > 0.5
    SIMILARITY_THRESHOLD = 0.5
    
    out = []
    documents = results["documents"][0]
    distances = results["distances"][0] if "distances" in results else [0] * len(documents)
    
    for i, (doc, dist) in enumerate(zip(documents, distances)):
        similarity = 1 - dist
        if similarity >= SIMILARITY_THRESHOLD:
            out.append(f"{i+1}. [Similarity: {similarity:.2f}] {doc}")
    
    if not out:
        return "No highly relevant results found. Try rephrasing your query or being more specific."
    
    return "\n".join(out)


# ------------------------------
# Node 1 — Parse user query
# ------------------------------
def parse_input(state: AgentState) -> AgentState:
    """Extract the latest user query from conversation messages"""
    user_msgs = [m for m in state["messages"] if isinstance(m, tuple) and m[0] == "user"]
    if user_msgs:
        state["query"] = user_msgs[-1][1]
    else:
        state["query"] = ""
    return state


# ------------------------------
# Node 2 — Search ChromaDB
# ------------------------------
def search_node(state: AgentState) -> AgentState:
    """Perform vector search in ChromaDB"""
    query = state.get("query", "")
    if not query:
        state["retrieved_docs"] = "No query provided."
        return state

    try:
        # Directly call search function (no .invoke)
        results = search_house_prices(query)
        state["retrieved_docs"] = results
    except Exception as e:
        state["retrieved_docs"] = f"Error during search: {e}"
    return state


# ------------------------------
# Node 3 — Generate answer using LLM
# ------------------------------
def llm_response_node(state: AgentState) -> AgentState:
    """Use LLM to summarize or answer user's question based on retrieved docs"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query = state.get("query", "")
    docs = state.get("retrieved_docs", "")

    # Build prompt context
    system_prompt = (
        "You are a helpful real estate assistant. Use the provided information "
        "about house prices to answer the user's question clearly and concisely."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User query: {query}\n\nSearch results:\n{docs}")
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    state["answer"] = answer
    state["messages"].append(("ai", answer))
    return state


# ------------------------------
# Build Graph
# ------------------------------

def build_graph():
    """Construct the LangGraph workflow"""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse", parse_input)
    graph.add_node("search", search_node)
    graph.add_node("llm_response", llm_response_node)

    # Define flow
    graph.set_entry_point("parse")
    graph.add_edge("parse", "search")
    graph.add_edge("search", "llm_response")
    graph.add_edge("llm_response", END)

    return graph.compile()


# ------------------------------
# Run with Streaming
# ------------------------------

if __name__ == "__main__":
    graph = build_graph()

    print("Real Estate Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        # Initial state
        state = {
            "messages": [("user", user_input)],
            "query": "",
            "retrieved_docs": "",
            "answer": ""
        }

        print("Bot: ", end="", flush=True)
        start = time.time()

        # Stream through graph nodes
        for step in graph.stream(state, stream_mode="updates"):
            # Each step may output partial state updates
            if "answer" in step:
                print(step["answer"], end="", flush=True)

        print(f"\n(Completed in {int((time.time()-start)*1000)} ms)\n")
