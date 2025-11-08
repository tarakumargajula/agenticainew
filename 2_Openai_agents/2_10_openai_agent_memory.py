import os
import pandas as pd
from datetime import datetime, timedelta
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

# Configuration
load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"

# Setup directories
BASE_PATH = "c://code//agenticai//2_openai_agents//data/memory"
SHORT_TERM_DIR = f"{BASE_PATH}/short_term"
LONG_TERM_DIR = f"{BASE_PATH}/long_term"

# Initialize stores
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
short_term = Chroma(collection_name="short_term", embedding_function=embeddings, persist_directory=SHORT_TERM_DIR)
long_term = Chroma(collection_name="long_term", embedding_function=embeddings, persist_directory=LONG_TERM_DIR)

SHORT_TERM_HOURS = 2

# Core functions
def add_memory(store, text):
    meta = {"timestamp": datetime.now().isoformat()}
    store.add_texts([text], metadatas=[meta])

def search_memory(store, query, k=3):
    try:
        count = len(store._collection.get()["ids"])
        if count == 0:
            return []
        return store.similarity_search_with_score(query, k=min(k, count))
    except:
        return []

def cleanup_short_term():
    try:
        items = short_term._collection.get(include=["metadatas"])
        now = datetime.now()
        expired = []
        
        for idx, meta in enumerate(items.get("metadatas", [])):
            if meta and "timestamp" in meta:
                timestamp = datetime.fromisoformat(meta["timestamp"])
                if (now - timestamp) >= timedelta(hours=SHORT_TERM_HOURS):
                    expired.append(items["ids"][idx])
        
        if expired:
            short_term._collection.delete(ids=expired)
    except:
        pass

def load_knowledge_base(csv_path="c://code//agenticai//2_openai_agents//Conversation.csv"):
    try:
        existing = len(long_term._collection.get()["ids"])
        if existing > 0:
            print(f"Knowledge base loaded: {existing} records\n")
            return
    except:
        pass
    
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        add_memory(long_term, f"Q: {row['question']}\nA: {row['answer']}")
    print(f"Loaded {len(df)} records\n")

def get_response(user_input):
    cleanup_short_term()
    
    # Search both memories
    recent = search_memory(short_term, user_input, k=2)
    knowledge = search_memory(long_term, user_input, k=2)
    
    # Show what was found
    print("\n" + "="*60)
    if recent:
        print("SHORT TERM:")
        for doc, score in recent:
            sim = max(0, (1 - (score ** 2) / 2) * 100)
            print(f"  {sim:.1f}% - {doc.page_content[:80]}")
    
    if knowledge:
        print("LONG TERM:")
        for doc, score in knowledge:
            sim = max(0, (1 - (score ** 2) / 2) * 100)
            print(f"  {sim:.1f}% - {doc.page_content[:80]}")
    print("="*60 + "\n")
    
    # Build context
    context = "\n".join([doc.page_content for doc, _ in recent + knowledge])
    
    # Get AI response
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nUser: {user_input}"}
        ]
    )
    
    answer = response.choices[0].message.content
    
    # Store conversation
    conv = f"User: {user_input}\nBot: {answer}"
    add_memory(short_term, conv)
    
    # Only save to long term if important words present
    important_words = ["remember", "important", "always", "never", "policy", "how to"]
    if any(word in user_input.lower() for word in important_words):
        add_memory(long_term, conv)
        print("Saved to long term memory\n")
    
    return answer

def main():
    load_knowledge_base()
    print("Chat ready. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
        
        answer = get_response(user_input)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    main()