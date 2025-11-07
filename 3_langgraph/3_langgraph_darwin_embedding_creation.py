from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import json
import faiss
import pickle

# --- Load environment ---
load_dotenv(override=True)
client = OpenAI()

# --- Function to split text into smaller chunks ---
def split_text(text, chunk_size=2000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

# --- Read all text files from directory and split into chunks ---
TEXT_DIR = "c://code//agenticai//3_langgraph//darwin"
texts = []

print("Reading text files from directory and splitting into chunks...")
for filename in os.listdir(TEXT_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(TEXT_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = split_text(content, chunk_size=2000)
            texts.extend(chunks)
print(f"Total chunks created: {len(texts)}")

# --- Compute embeddings for all chunks ---
print("Computing embeddings for text chunks...")
chunk_embeddings = []

def compute_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

for text in texts:
    embedding = compute_embedding(text)
    chunk_embeddings.append(embedding)
print("Embeddings ready!")

# --- Build FAISS index ---
embedding_dim = len(chunk_embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(chunk_embeddings).astype('float32'))
print("FAISS index built!")

# --- Save embeddings and FAISS index ---
EMBEDDINGS_FILE = "chunk_embeddings.pkl"
INDEX_FILE = "faiss_index.bin"

# Save embeddings with pickle
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump({
        "texts": texts,
        "embeddings": chunk_embeddings
    }, f)
print(f"Chunk embeddings saved to {EMBEDDINGS_FILE}")

# Save FAISS index
faiss.write_index(index, INDEX_FILE)
print(f"FAISS index saved to {INDEX_FILE}")