import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class StudyRetriever:
    def __init__(self):
        # 1. Configuration from .env
        self.index_path = os.getenv("INDEX_PATH", "vector_store/faiss_index.bin")
        self.docs_path = os.getenv("METADATA_PATH", "vector_store/docs.pkl")
        self.top_k = int(os.getenv("TOP_K", 3))

        # 2. Load the Embedding Model
        # Must be the same model used in build_index.py
        print("🔄 Loading Embedding Model for Retrieval...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # 3. Load FAISS index (The Vector Database)
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"✅ FAISS Index Loaded: {self.index_path}")
        else:
            raise FileNotFoundError(f"❌ Index not found at {self.index_path}. Run build_index.py!")

        # 4. Load Metadata (The actual text chunks)
        if os.path.exists(self.docs_path):
            with open(self.docs_path, "rb") as f:
                self.docs = pickle.load(f)
            print(f"✅ Metadata Loaded: {self.docs_path}")
        else:
            raise FileNotFoundError(f"❌ Metadata file not found at {self.docs_path}.")

    def get_relevant_context(self, query, top_k=None):
        """
        Converts a user query into a vector and finds the best matching text chunks.
        """
        if top_k is None:
            top_k = self.top_k

        # Convert question to numbers (vector)
        query_vector = self.model.encode([query])
        
        # Search the index for the 'k' nearest neighbors
        distances, indices = self.index.search(query_vector, top_k)
        
        # Fetch the actual text for the indices found
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.docs):
                results.append(self.docs[idx])
        
        return results