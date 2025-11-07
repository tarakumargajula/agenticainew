# pip install chromadb pandas sentence-transformers

from typing import Dict
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
import time


def prepare_house_prices_document() -> Dict:
    """
    Convert the house prices dataset to a list of ChromaDB-ready documents.
    Each house becomes a searchable document.
    """
    print("Reading house_prices.csv...")
    df = pd.read_csv("c://code//agenticai//3_langgraph//house_prices.csv")
    print(f"Loaded {len(df)} rows.")

    documents = []
    ids = []

    for index, row in df.iterrows():
        if index % 50 == 0 and index > 0:
            print(f"  Processing row {index}/{len(df)}...")

        # Safely handle nulls for optional fields
        carpet_area = row["Carpet Area"] if pd.notnull(row["Carpet Area"]) else ""
        balcony = row["Balcony"] if pd.notnull(row["Balcony"]) else 0
        car_parking = row["Car Parking"] if pd.notnull(row["Car Parking"]) else ""

        # Create rich document text for semantic search
        document_text = f"""
        Size and location: {row['Title']}
        Description: {row['Description']}
        Price: {row['Amount(in rupees)']}
        City: {row['location']}
        Carpet Area: {carpet_area}
        Floor: {row['Floor']}
        New/Resale: {row['Transaction']}
        Furnishing: {row['Furnishing']}
        Bathrooms: {row['Bathroom']}
        Balconies: {balcony}
        Car Parking: {car_parking}
        """

        documents.append(document_text.strip())
        ids.append(f"house_{index}")

    print(f"Prepared {len(documents)} documents for ChromaDB.")
    return {"documents": documents, "ids": ids}


def setup_house_prices_chromadb(batch_size=5000):
    """
    Create and populate ChromaDB collection with house prices data in batches
    without using metadata.
    """
    import time
    start_time = time.time()  # Start timing

    print("Loading Hugging Face embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")

    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path="C:\\code\\agenticai\\3_langgraph\\chroma")

    print("Deleting existing collection (if any)...")
    try:
        client.delete_collection("house_prices")
        print("Old collection deleted.")
    except BaseException:
        print("No existing collection found. Creating new one.")

    print("Creating new collection...")
    collection = client.create_collection(
        name="house_prices",
        metadata={"description": "Housing prices data"},
    )
    print("Collection created.")

    # Prepare documents
    data = prepare_house_prices_document()

    print("Generating embeddings...")
    embeddings = model.encode(data["documents"]).tolist()
    print("Embeddings generated.")

    # Add documents in batches
    print(f"Adding documents to ChromaDB in batches of {batch_size}...")
    total_docs = len(data["documents"])
    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)
        collection.add(
            documents=data["documents"][start_idx:end_idx],
            ids=data["ids"][start_idx:end_idx],
            embeddings=embeddings[start_idx:end_idx],
        )
        print(f"  Added documents {start_idx} to {end_idx-1}")

    end_time = time.time()
    print(f"Total time taken to set up ChromaDB: {end_time - start_time:.2f} seconds")

    return collection, model



def query_chromadb(collection, model, query: str, n_results: int = 3):
    """
    Perform semantic search on the ChromaDB collection using Hugging Face embeddings.
    """
    print(f"Querying ChromaDB for: '{query}'")
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    print(f"Found {len(results['documents'][0])} results.")
    return results


if __name__ == "__main__":
    collection, model = setup_house_prices_chromadb()
    print("ChromaDB setup complete. Ready for queries!\n")

    # Example query
    query = "2 BHK flats in Mumbai under 1 crore"
    results = query_chromadb(collection, model, query)

    print("\nSearch Results:")
    for i, doc in enumerate(results["documents"][0]):
        print(f"{i+1}. {doc}...\n")
