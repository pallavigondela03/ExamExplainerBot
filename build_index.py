import os
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()

# --- CONFIGURATION ---
# These paths are relative to the folder where you run the script
DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
DOCS_FILE = os.path.join(VECTOR_STORE_DIR, "docs.pkl")

def prepare_vector_store():
    """Processes PDFs in /data and saves FAISS index to /vector_store."""
    
    # 1. Ensure directories exist
    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)
        print(f"Created directory: {VECTOR_STORE_DIR}")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"⚠️ Created '{DATA_DIR}' folder. Please drop your PDFs there and run again.")
        return

    # 2. Load Embedding Model
    print("🚀 Loading Embedding Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    all_chunks = []

    # 3. Extract and Chunk Text
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ Error: No PDF files found in '{DATA_DIR}/'. Place your exam rules there.")
        return

    print(f"📖 Reading {len(pdf_files)} PDF(s) from {DATA_DIR}...")
    for file_name in pdf_files:
        path = os.path.join(DATA_DIR, file_name)
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content
            
            if not text.strip():
                print(f"⚠️ Warning: {file_name} appears to be empty or scanned (no selectable text).")
                continue

            # Intelligent Chunking: 700 chars with 100 char overlap
            chunk_size = 700
            overlap = 100
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                all_chunks.append(chunk)
                documents.append({
                    "text": chunk,
                    "source": file_name
                })
        except Exception as e:
            print(f"❌ Could not read {file_name}: {e}")

    if not all_chunks:
        print("❌ No text chunks extracted. Indexing aborted.")
        return

    # 4. Generate Embeddings
    print(f"🧠 Generating embeddings for {len(all_chunks)} chunks... (This may take a moment)")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # 5. Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) 
    index.add(embeddings)

    # 6. Save Files
    print("💾 Saving FAISS index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(documents, f)

    print(f"✅ SUCCESS! Vector store created at '{VECTOR_STORE_DIR}/'")
    print(f"   Files created: {os.path.basename(INDEX_FILE)} and {os.path.basename(DOCS_FILE)}")

if __name__ == "__main__":
    prepare_vector_store()