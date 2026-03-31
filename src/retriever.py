import os
import faiss
import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# Load the EXACT SAME model used in the indexer
print("[Init] Loading IBM Granite 30M Embedder for Retriever...")
embedder = SentenceTransformer("ibm-granite/granite-embedding-30m-english")

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
    # 1. Lazy load indexes if not already in memory
    if VECTOR_INDEX is None:
        if not load_indexes():
            return []

    # --- 2. Embed the query via Local Granite Model ---
    try:
        # Encode locally with Granite. normalize_embeddings=True must match indexer!
        embed_np = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    except Exception as e:
        print(f"[Retriever] Embedding failed: {e}")
        return []

    # --- 3. Vector Search (FAISS) ---
    # FAISS returns distances (D) and indices (I)
    try:
        D, I = VECTOR_INDEX.search(embed_np, k)
    except Exception as e:
        print(f"[Retriever] FAISS Search Error: {e}")
        print("This usually happens if indexer.py was run with a different embedder (Dimension mismatch).")
        return []

    # --- 4. Keyword Search (BM25) ---
    # Tokenize the query identically to how we indexed it
    tokenized_query = re.findall(r'\b[a-z0-9]+\b', query.lower())
    bm25_scores = BM25_INDEX.get_scores(tokenized_query)
    
    # Get the indices of the top K highest scoring documents
    top_n_bm25 = np.argsort(bm25_scores)[::-1][:k]

    # --- 5. Reciprocal Rank Fusion (RRF) ---
    RRF_K = 60
    final_scores = {}

    # Automotive Domain Boosting
    wants_code = any(word in query.lower() for word in ["code", "p0", "dtc", "table"])
    wants_visual = any(word in query.lower() for word in ["diagram", "show", "schematic", "where", "look"])

    def get_boost(chunk_idx: int) -> float:
        # Depending on how the data was ingested (mock dict vs object), extract the type safely
        chunk = CHUNK_MAP.get(chunk_idx, {})
        chunk_type = chunk.chunk_type if hasattr(chunk, 'chunk_type') else chunk.get("chunk_type", "text")
        
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

    # --- 6. Noise Gate & Formatting ---
    sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_candidates:
        return []

    best_score = sorted_candidates[0][1]
    results = []
    
    for idx, score in sorted_candidates:
        # Dynamic cutoff: Drop anything that scores less than 50% of the best match
        if score >= best_score * 0.5:
            # Safely convert parsed object to dictionary if it isn't one already
            chunk_obj = CHUNK_MAP[idx]
            if hasattr(chunk_obj, '__dict__'):
                chunk_data = chunk_obj.__dict__.copy()
            else:
                chunk_data = chunk_obj.copy()
                
            chunk_data["search_score"] = round(score, 4)
            results.append(chunk_data)

    # Return top K results
    return results[:k]


if __name__ == "__main__":
    print("\n--- Testing Hybrid Retriever ---")
    
    # Test 1: Querying for an Image
    print("\n[Test 1] Searching for an image description...")
    test_query_visual = "What does the flowchart for engine misfires look like?"
    retrieved_chunks = perform_hybrid_search(test_query_visual, k=3)
    
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Result {i+1} (Score: {chunk['search_score']}) | Type: {chunk['chunk_type']} | Page {chunk['metadata'].get('page', 'Unknown')}")
        print(f"Content: {chunk['content'][:80]}...\n")
        
    # Test 2: Querying for a Table/Code
    print("-" * 40)
    print("\n[Test 2] Searching for a fault code...")
    test_query_code = "What does the P0420 fault code mean?"
    retrieved_chunks = perform_hybrid_search(test_query_code, k=3)
    
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Result {i+1} (Score: {chunk['search_score']}) | Type: {chunk['chunk_type']} | Page {chunk['metadata'].get('page', 'Unknown')}")
        print(f"Content: {chunk['content'][:80]}...\n")