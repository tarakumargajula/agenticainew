import gradio as gr
import sqlite3
import time
from datetime import datetime
from typing import TypedDict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

# ---------------------------------
# 1. Define the State
# ---------------------------------
class ProductState(TypedDict):
    query: str
    results: str
    memory_hit: bool
    explanation: str


# ---------------------------------
# 2. SQLite setup (for memory + logs)
# ---------------------------------
conn = sqlite3.connect('memory.db', check_same_thread=False)
cursor = conn.cursor()

# Store past query results
cursor.execute("""
CREATE TABLE IF NOT EXISTS memory (
    query TEXT PRIMARY KEY,
    results TEXT,
    timestamp TEXT
)
""")

# Store performance logs
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    timestamp TEXT,
    query TEXT,
    memory_hit INTEGER,
    latency_ms INTEGER,
    error TEXT
)
""")
conn.commit()


# ---------------------------------
# 3. Helper functions for memory + logs
# ---------------------------------
def get_memory(query: str) -> Tuple[str, bool, str]:
    """Check if query already exists in SQLite memory."""
    cursor.execute("SELECT results, timestamp FROM memory WHERE query = ?", (query,))
    row = cursor.fetchone()
    if row:
        results, timestamp = row
        explanation = f"Loaded from memory; last updated at {timestamp}."
        return f"(Loaded from memory: {timestamp})\n{results}", True, explanation
    return None, False, ""


def save_memory(query: str, results: str):
    """Save query results with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT OR REPLACE INTO memory (query, results, timestamp) VALUES (?, ?, ?)",
        (query, results, timestamp)
    )
    conn.commit()


def log_event(query: str, memory_hit: bool, latency_ms: int, error: str = None):
    """Log query execution details."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO logs (timestamp, query, memory_hit, latency_ms, error) VALUES (?, ?, ?, ?, ?)",
        (timestamp, query, int(memory_hit), latency_ms, error)
    )
    conn.commit()


# ---------------------------------
# 4. Setup ChromaDB with embeddings
# ---------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//product_embeddings_chroma",
    embedding_function=embeddings
)


# ---------------------------------
# 5. Node 1 — Search + Explainability + Logging
# ---------------------------------
def search_products(state: ProductState) -> ProductState:
    start_time = time.time()
    error_message = None

    try:
        # Check if in memory
        cached_result, memory_hit, explanation = get_memory(state["query"])
        if memory_hit:
            state["results"] = cached_result
            state["memory_hit"] = True
            state["explanation"] = explanation
        else:
            # Search Chroma vector DB
            results = vectordb.similarity_search_with_score(state["query"], k=3)

            if not results:
                state["results"] = "No products found."
                state["memory_hit"] = False
                state["explanation"] = "No matches found."
            else:
                titles, explain_details = [], []
                for doc, score in results:
                    title = doc.metadata.get("title", "Unknown product")
                    titles.append(f"{title} (score: {score:.4f})")
                    explain_details.append(f"Matched '{title}' with similarity score {score:.4f}")

                result_text = "\n".join(titles)
                explanation = "\n".join(explain_details)
                state["results"] = result_text
                state["memory_hit"] = False
                state["explanation"] = explanation

                # Save to memory for future
                save_memory(state["query"], result_text)

    except Exception as e:
        error_message = str(e)
        state["results"] = "An error occurred during search."
        state["memory_hit"] = False
        state["explanation"] = f"Error details: {error_message}"

    # Log query performance
    latency_ms = int((time.time() - start_time) * 1000)
    log_event(state["query"], state["memory_hit"], latency_ms, error_message)

    return state


# ---------------------------------
# 6. Node 2 — Format response neatly
# ---------------------------------
def format_response(state: ProductState) -> ProductState:
    if state["results"]:
        prefix = "Found products:" if not state.get("memory_hit") else ""
        explanation = state.get("explanation", "")
        state["results"] = (
            f"{prefix}\n{state['results']}\n\nExplanation:\n{explanation}"
            if prefix else f"{state['results']}\n\nExplanation:\n{explanation}"
        )
    else:
        state["results"] = "No products found."
    return state


# ---------------------------------
# 7. Build LangGraph pipeline
# ---------------------------------
graph = StateGraph(ProductState)
graph.add_node("search", search_products)
graph.add_node("format", format_response)
graph.set_entry_point("search")
graph.add_edge("search", "format")
graph.add_edge("format", END)
runnable = graph.compile()


# ---------------------------------
# 8. Gradio Search Interface
# ---------------------------------
def search(query):
    result = runnable.invoke({"query": query})
    return result["results"]


def view_logs():
    """Display last 20 log entries."""
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 20")
    rows = cursor.fetchall()
    return "\n".join([
        f"[{r[0]}] Query: {r[1]}, Memory: {bool(r[2])}, Latency: {r[3]} ms, Error: {r[4]}"
        for r in rows
    ])


def chat_fn(message, history):
    """Handle Gradio chat flow."""
    response = search(message)
    history = history or []
    history.append([message, response])
    return history


# ---------------------------------
# 9. Gradio UI (Chat + Logs)
# ---------------------------------
demo = gr.Blocks()

with demo:
    gr.Markdown("# Product Search (ChromaDB + Memory + Explainability + Logs)")

    with gr.Tab("Search"):
        chatbot = gr.Chatbot(label="Agent Output")
        query = gr.Textbox(label="Ask something", placeholder="e.g. wireless headphones")
        query.submit(chat_fn, [query, chatbot], [chatbot])

    with gr.Tab("Admin Logs"):
        gr.Markdown("### Recent Logs")
        logs_output = gr.Textbox(label="Logs", lines=20, interactive=False)
        refresh = gr.Button("Refresh Logs")
        refresh.click(fn=view_logs, inputs=[], outputs=[logs_output])

if __name__ == "__main__":
    demo.launch()
