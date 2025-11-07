from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Step 1: Setup HuggingFace Embeddings and Chroma Vector Store ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name="greetings_memory", embedding_function=embeddings)


# --- Step 2: Define helper functions ---
def save_memory(texts):
    """Save multiple text memories."""
    vectorstore.add_texts(texts)


def recall_memory_with_scores(query, howmany=2):
    """Recall most similar texts along with Euclidean distances."""
    """Lesser distance = More similar """
    results = vectorstore.similarity_search_with_score(query, k=howmany)
    # Each result: (Document, Euclidean distance)
    return [(r[0].page_content, r[1]) for r in results]


# --- Step 3: Stateful simulation (like MemorySaver example) ---
state = {}

# --- Step 4: First run ---
print("\n--- First Run ---")
if state.get("first_run", True):
    greetings = ["Hi there", "Bye, see you later", "Greetings!", "Do not come", "Gone already"]
    save_memory(greetings)
    state["msg"] = "First run: Saved greetings to Chroma vector memory!"
    state["first_run"] = False
    print(state["msg"])


# --- Step 5: Second run ---
print("\n--- Second Run ---")
query = "Hello"
results = recall_memory_with_scores(query, 5)

if results:
    state["msg"] = f"Second run: Compared '{query}' with stored greetings.\nMost similar ones:"
    for i, (text, score) in enumerate(results, start=1):
        state["msg"] += f"\n{i}. {text}  (distance: {score:.3f})"
else:
    state["msg"] = "No results found in Chroma memory."

print(state["msg"])


# --- Step 6: Inspect stored Chroma memory ---
print("\n--- Chroma DB Contents ---")
print("Total items stored:", vectorstore._collection.count())
