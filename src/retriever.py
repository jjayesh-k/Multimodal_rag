import os
import faiss
import pickle
import numpy as np
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load API keys
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
hf_client = InferenceClient(token=HF_TOKEN)

# Global variables to hold our loaded indexes
VECTOR_INDEX = None
BM25_INDEX = None
CHUNK_MAP = None

def load_indexes(save_dir="./index_storage"):
    """Loads the FAISS, BM25, and metadata mapping from disk into memory."""
    global VECTOR_INDEX, BM25_INDEX, CHUNK_MAP
    
    try:
        print("[Retriever] Loading indexes from disk...")
        VECTOR_INDEX = faiss.read_index(os.path.join(save_dir, "faiss_hnsw.index"))
        
        with open(os.path.join(save_dir, "bm25.pkl"), "rb") as f:
            BM25_INDEX = pickle.load(f)
            
        with open(os.path.join(save_dir, "chunk_map.pkl"), "rb") as f:
            CHUNK_MAP = pickle.load(f)
            
        print(f"[Retriever] Successfully loaded {len(CHUNK_MAP)} chunks.")
        return True
    except Exception as e:
        print(f"[Retriever] Error loading indexes: {e}")
        print("Did you run the indexer.py script first?")
        return False

def perform_hybrid_search(query: str, k: int = 5) -> list:
    """
    Hybrid search: vector (FAISS) + keyword (BM25) fused with RRF.
    Returns the actual chunk objects with metadata for the LLM.
    """
    if VECTOR_INDEX is None:
        load_indexes()

    # --- 1. Embed the query via Hugging Face SDK ---
    try:
        embed_response = hf_client.feature_extraction(
            text=[query],
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Format for FAISS (needs to be float32 numpy array)
        if isinstance(embed_response, np.ndarray):
            embed_np = embed_response.astype(np.float32)
        else:
            embed_np = np.array(embed_response).astype(np.float32)
            
    except Exception as e:
        print(f"[Retriever] Embedding failed: {e}")
        return []

    # --- 2. Vector Search (FAISS) ---
    # FAISS returns distances (D) and indices (I)
    D, I = VECTOR_INDEX.search(embed_np, k)

    # --- 3. Keyword Search (BM25) ---
    # Tokenize the query identically to how we indexed it
    tokenized_query = re.findall(r'\b[a-z0-9]+\b', query.lower())
    bm25_scores = BM25_INDEX.get_scores(tokenized_query)
    top_n_bm25 = np.argsort(bm25_scores)[::-1][:k]

    # --- 4. Reciprocal Rank Fusion (RRF) ---
    RRF_K = 60
    final_scores = {}

    # Automotive Domain Boosting: If they ask for codes, boost tables. If they ask for diagrams, boost images.
    wants_code = any(word in query.lower() for word in ["code", "p0", "dtc", "table"])
    wants_visual = any(word in query.lower() for word in ["diagram", "show", "schematic", "where"])

    def get_boost(chunk_idx: int) -> float:
        chunk = CHUNK_MAP.get(chunk_idx, {})
        chunk_type = chunk.get("chunk_type", "text")
        
        boost = 0.0
        if wants_code and chunk_type == "table":
            boost += 0.2
        if wants_visual and chunk_type == "image":
            boost += 0.2
        return boost

    # Add FAISS scores to RRF
    for rank, idx in enumerate(I[0]):
        if idx == -1: continue
        final_scores[int(idx)] = (1.0 / (rank + RRF_K)) + get_boost(int(idx))

    # Add BM25 scores to RRF
    for rank, idx in enumerate(top_n_bm25):
        if idx not in final_scores:
            final_scores[int(idx)] = 0.0
        final_scores[int(idx)] += (1.0 / (rank + RRF_K)) + get_boost(int(idx))

    # --- 5. Noise Gate & Formatting ---
    sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_candidates:
        return []

    best_score = sorted_candidates[0][1]
    results = []
    
    for idx, score in sorted_candidates:
        # Dynamic cutoff: Drop anything that scores less than 50% of the best match
        if score >= best_score * 0.5:
            # Grab the FULL metadata object, not just the text
            chunk_data = CHUNK_MAP[idx]
            # Attach the search score for debugging
            chunk_data["search_score"] = round(score, 4)
            results.append(chunk_data)

    # Return top K results
    return results[:k]


if __name__ == "__main__":
    print("\n--- Testing Hybrid Retriever ---")
    # Using a query that tests the BM25 keyword matching (P0420)
    test_query = "What does the P0420 fault code mean?"
    
    retrieved_chunks = perform_hybrid_search(test_query, k=3)
    
    print(f"\nFound {len(retrieved_chunks)} relevant chunks:\n")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Result {i+1} (Score: {chunk['search_score']})")
        print(f"Type: {chunk['chunk_type']}")
        print(f"Source: Page {chunk['metadata'].get('page', 'Unknown')}")
        print(f"Content: {chunk['content'][:100]}...\n")
        print("-" * 40)