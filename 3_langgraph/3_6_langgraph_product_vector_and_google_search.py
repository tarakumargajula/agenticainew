# pip install langchain-google-genai langchain-huggingface langgraph chromadb requests gradio
import os
import gradio as gr
import requests
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Step 1: Define state
class State(TypedDict):
    query: str
    vector: str
    serp: str
    llm: str

# Step 2: Setup embeddings, Chroma, and LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//product_embeddings_chroma",
    embedding_function=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Step 3: Define graph nodes
def vector_search(state: State) -> State:
    """Search for similar products in Chroma vector database"""
    results = vectordb.similarity_search(state["query"], k=2)
    if not results:
        state["vector"] = "No matching products found."
        return state
    state["vector"] = "\n".join([doc.metadata.get("title", "Unknown product") for doc in results])
    return state


def serp_search(state: State) -> State:
    """Fetch web results using SERP API"""
    url = "https://serpapi.com/search"
    params = {
        "q": f"{state['query']} reviews",
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": 2
    }
    data = requests.get(url, params=params).json()
    organic = data.get("organic_results", [])
    if not organic:
        state["serp"] = "No web results found."
        return state
    results = [f"{r.get('title', 'No title')}: {r.get('snippet', '')}" for r in organic[:2]]
    state["serp"] = "\n".join(results)
    return state


def llm_analyze(state: State) -> State:
    """Combine results and get AI-generated analysis"""
    prompt = (
        f"Analyze the following query:\n\n"
        f"Query: {state['query']}\n\n"
        f"Vector DB Results:\n{state['vector']}\n\n"
        f"Web Results:\n{state['serp']}"
    )
    response = llm.invoke(prompt)
    state["llm"] = response.content
    return state

# Step 4: Build the graph
graph = StateGraph(State)
graph.add_node("vector_node", vector_search)
graph.add_node("serp_node", serp_search)
graph.add_node("llm_node", llm_analyze)

graph.set_entry_point("vector_node")
graph.add_edge("vector_node", "serp_node")
graph.add_edge("serp_node", "llm_node")
graph.add_edge("llm_node", END)

runnable = graph.compile()

# Step 5: Gradio interface
def search(query, chat_history):
    result = runnable.invoke({"query": query})
    answer = (
        f"Vector DB Results:\n{result['vector']}\n\n"
        f"Web Search Results:\n{result['serp']}\n\n"
        f"AI Analysis:\n{result['llm']}"
    )
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

demo = gr.ChatInterface(
    fn=search,
    title="Product Search with LangGraph and Chroma",
    examples=["iPhone 15", "best budget laptop", "Samsung Galaxy S23 reviews"],
    type="messages"
)

if __name__ == "__main__":
    demo.launch()
