import pandas as pd
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

def create_and_save_embeddings():
    """
    Extract product titles from CSV, create embeddings, and save to files
    """
    
    # Step 1: Read CSV and extract titles
    print("Reading amazon_products.csv...")
    try:
        df = pd.read_csv('amazon_products.csv')
        print(f"Loaded {len(df)} products from CSV")
        
        # Extract title column (strip whitespace from column names)
        df.columns = df.columns.str.strip()
        
        if 'title' not in df.columns:
            # Check for common title column variations
            title_cols = [col for col in df.columns if 'title' in col.lower()]
            if title_cols:
                title_col = title_cols[0]
                print(f"Using column '{title_col}' as title")
            else:
                print("Available columns:", list(df.columns))
                raise ValueError("No 'title' column found. Please check your CSV structure.")
        else:
            title_col = 'title'
        
        # Remove any NaN/empty titles
        df = df.dropna(subset=[title_col])
        df = df[df[title_col].str.strip() != '']
        
        print(f"Processing {len(df)} valid product titles...")
        
    except FileNotFoundError:
        print("Error: amazon_products.csv not found in current directory")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Step 2: Create documents from titles
    print("Creating documents...")
    documents = []
    for idx, row in df.iterrows():
        title = str(row[title_col]).strip()
        # Create document with title as content and include index for reference
        doc = Document(
            page_content=title,
            metadata={
                "title": title,
                "csv_index": idx  # Store original CSV index for reference
            }
        )
        documents.append(doc)
    
    # Step 3: Split documents (optional for titles, but keeping for consistency)
    print("Splitting documents...")
    splitter = CharacterTextSplitter(
        chunk_size=200,  # Titles are usually short
        chunk_overlap=20,
        separator=" "
    )
    docs = splitter.split_documents(documents)
    print(f"Created {len(docs)} document chunks")
    
    # Step 4: Initialize embeddings model
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
    )
    
    # Step 5: Create vector database
    print("Creating embeddings and vector database...")
    vectordb = FAISS.from_documents(docs, embeddings)
    print("Vector database created successfully!")
    
    # Step 6: Save embeddings and vector database
    print("Saving embeddings to files...")
    
    # Save FAISS vector database
    vectordb.save_local("product_embeddings_faiss")
    
    # Save the original DataFrame for reference
    df.to_pickle("product_data.pkl")
    
    # Save embedding model info for later loading
    embedding_info = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "total_documents": len(docs),
        "total_products": len(df)
    }
    
    with open("embedding_info.pkl", "wb") as f:
        pickle.dump(embedding_info, f)
    
    print("Embeddings saved successfully!")
    print(f"Files created:")
    print(f"   - product_embeddings_faiss/ (FAISS vector database)")
    print(f"   - product_data.pkl (original product data)")
    print(f"   - embedding_info.pkl (metadata)")
    print(f"Statistics:")
    print(f"   - Total products processed: {len(df)}")
    print(f"   - Total document chunks: {len(docs)}")

def load_embeddings_example():
    """
    Example function showing how to load the saved embeddings
    """
    print("\n" + "="*50)
    print("EXAMPLE: How to load saved embeddings")
    print("="*50)
    
    try:
        # Load embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load FAISS vector database
        vectordb = FAISS.load_local(
            "product_embeddings_faiss", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load original data
        df = pd.read_pickle("product_data.pkl")
        
        # Load metadata
        with open("embedding_info.pkl", "rb") as f:
            info = pickle.load(f)
        
        print("Successfully loaded embeddings!")
        print(f"Loaded {info['total_products']} products with {info['total_documents']} document chunks")
        
        # Test search
        test_query = "Luggage suitcase"
        results = vectordb.similarity_search(test_query, k=3)
        
        print(f"\nTest search for '{test_query}':")
        for i, doc in enumerate(results, 1):
            print(f"   {i}. {doc.page_content}")
            
    except FileNotFoundError:
        print("Embedding files not found. Run create_and_save_embeddings() first.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")

if __name__ == "__main__":
    # Create and save embeddings
    create_and_save_embeddings()
    
    # Show example of how to load them
    load_embeddings_example()