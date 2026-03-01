import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

def verify_system():
    print("--- System Health Check ---")
    
    # 1. Check Files
    files = ["vector_store/faiss_index.bin", "vector_store/docs.pkl"]
    for f in files:
        if os.path.exists(f):
            print(f"✅ Found {f}")
        else:
            print(f"❌ Missing {f}")

    # 2. Test Retrieval logic
    try:
        index = faiss.read_index("vector_store/faiss_index.bin")
        with open("vector_store/docs.pkl", "rb") as f:
            docs = pickle.load(f)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_query = "What are the passing marks?"
        query_vector = model.encode([test_query])
        
        D, I = index.search(query_vector, k=1)
        
        print(f"✅ Retrieval Test: Found match in {docs[I[0][0]]['source']}")
        print("--- System is 100% Ready ---")
    except Exception as e:
        print(f"❌ Test Failed: {e}")

if __name__ == "__main__":
    verify_system()