import gradio as gr
import sqlite3
import time
from datetime import datetime
from typing import TypedDict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

# ------------------------------------------------------
# 1. Define the State
# ------------------------------------------------------
class ProductState(TypedDict):
    query: str
    results: str
    memory_hit: bool
    explanation: str


# ------------------------------------------------------
# 2. SQLite setup (for memory + logs)
# ------------------------------------------------------
conn = sqlite3.connect('hitl_memory.db', check_same_thread=False)
cursor = conn.cursor()

# Store approved or edited results
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


# ------------------------------------------------------
# 3. Helper functions for memory + logs
# ------------------------------------------------------
def get_memory(query: str) -> Tuple[str, bool, str]:
    cursor.execute("SELECT results, timestamp FROM memory WHERE query = ?", (query,))
    row = cursor.fetchone()
    if row:
        results, timestamp = row
        explanation = f"Loaded from memory (last updated: {timestamp})"
        return results, True, explanation
    return None, False, ""


def save_memory(query: str, results: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT OR REPLACE INTO memory (query, results, timestamp) VALUES (?, ?, ?)",
        (query, results, timestamp)
    )
    conn.commit()


def log_event(query: str, memory_hit: bool, latency_ms: int, error: str = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO logs (timestamp, query, memory_hit, latency_ms, error) VALUES (?, ?, ?, ?, ?)",
        (timestamp, query, int(memory_hit), latency_ms, error)
    )
    conn.commit()


def view_logs():
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 20")
    rows = cursor.fetchall()
    return "\n".join([
        f"[{r[0]}] Query: {r[1]}, Memory: {bool(r[2])}, Latency: {r[3]} ms, Error: {r[4]}"
        for r in rows
    ]) or "No logs yet."


def view_memory():
    cursor.execute("SELECT * FROM memory ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    if not rows:
        return "Memory is empty"
    lines = []
    for query, results, ts in rows:
        lines.append(f"Query: {query}")
        lines.append(results)
        lines.append(f"(Saved on: {ts})")
        lines.append("-" * 40)
    return "\n".join(lines)


# ------------------------------------------------------
# 4. Setup ChromaDB with embeddings
# ------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//product_embeddings_chroma",
    embedding_function=embeddings
)


# ------------------------------------------------------
# 5. Node 1 — Search + Explainability + Logging
# ------------------------------------------------------
def search_products(state: ProductState) -> ProductState:
    start_time = time.time()
    error_message = None

    try:
        # Check SQLite memory first
        cached_result, memory_hit, explanation = get_memory(state["query"])
        if memory_hit:
            state["results"] = cached_result
            state["memory_hit"] = True
            state["explanation"] = explanation
        else:
            # Search Chroma
            results = vectordb.similarity_search_with_score(state["query"], k=3)
            if not results:
                state["results"] = "No products found."
                state["memory_hit"] = False
                state["explanation"] = "No matches found."
            else:
                titles, explain_details = [], []
                for doc, score in results:
                    title = doc.metadata.get("title", "Unknown product")
                    titles.append(f"- {title} (score: {score:.4f})")
                    explain_details.append(f"Matched '{title}' with similarity {score:.4f}")

                result_text = "\n".join(titles)
                explanation = "\n".join(explain_details)
                state["results"] = result_text
                state["memory_hit"] = False
                state["explanation"] = explanation

    except Exception as e:
        error_message = str(e)
        state["results"] = "An error occurred during search."
        state["memory_hit"] = False
        state["explanation"] = f"Error: {error_message}"

    latency_ms = int((time.time() - start_time) * 1000)
    log_event(state["query"], state["memory_hit"], latency_ms, error_message)

    return state


# ------------------------------------------------------
# 6. Node 2 — Format response
# ------------------------------------------------------
def format_response(state: ProductState) -> ProductState:
    prefix = "Found products:" if not state.get("memory_hit") else "Loaded from memory:"
    explanation = state.get("explanation", "")
    state["results"] = f"{prefix}\n{state['results']}\n\nExplanation:\n{explanation}"
    return state


# ------------------------------------------------------
# 7. Build Graph
# ------------------------------------------------------
graph = StateGraph(ProductState)
graph.add_node("search", search_products)
graph.add_node("format", format_response)
graph.set_entry_point("search")
graph.add_edge("search", "format")
graph.add_edge("format", END)
runnable = graph.compile()


# ------------------------------------------------------
# 8. Core functions for Gradio
# ------------------------------------------------------
def run_search(query):
    if not query.strip():
        return "Please enter a query", "", "", view_memory(), view_logs()
    
    result = runnable.invoke({"query": query, "results": "", "memory_hit": False, "explanation": ""})
    return result["results"], result["results"], "", view_memory(), view_logs()


def approve(query, results):
    if not query.strip():
        return "No query provided", view_memory(), view_logs()
    save_memory(query, results)
    return f"Approved and saved for '{query}'", view_memory(), view_logs()


def edit(query, edited_results):
    if not query.strip():
        return "No query provided", view_memory(), view_logs()
    save_memory(query, edited_results)
    return f"Edited and saved for '{query}'", view_memory(), view_logs()


# ------------------------------------------------------
# 9. Gradio UI (Search + HITL + Logs)
# ------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Product Search with ChromaDB + HITL + SQLite Memory + Logs")

    with gr.Row():
        with gr.Column():
            query_box = gr.Textbox(label="Enter query", placeholder="e.g., wireless headphones")
            search_btn = gr.Button("Search", variant="primary")

            results_box = gr.Textbox(label="Search Results", lines=6)
            edit_box = gr.Textbox(label="Edit Results (optional)", lines=6)

            with gr.Row():
                approve_btn = gr.Button("Approve")
                edit_btn = gr.Button("Save Edited")

            status_box = gr.Textbox(label="Status", lines=2)
        
        with gr.Column():
            memory_box = gr.Textbox(label="SQLite Memory", lines=20, interactive=False)
            logs_box = gr.Textbox(label="Query Logs", lines=20, interactive=False)
            refresh_btn = gr.Button("Refresh Logs and Memory")

    # Bind buttons
    search_btn.click(run_search, inputs=[query_box], outputs=[results_box, edit_box, status_box, memory_box, logs_box])
    approve_btn.click(approve, inputs=[query_box, results_box], outputs=[status_box, memory_box, logs_box])
    edit_btn.click(edit, inputs=[query_box, edit_box], outputs=[status_box, memory_box, logs_box])
    refresh_btn.click(lambda: ("", view_memory(), view_logs()), inputs=[], outputs=[status_box, memory_box, logs_box])

if __name__ == "__main__":
    demo.launch()
