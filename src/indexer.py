import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import re
import os
import pickle
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load the API keys from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize the official Hugging Face SDK Client
hf_client = InferenceClient(token=HF_TOKEN)

def simple_tokenize(text):
    """Preserves numbers and acronyms for BM25 Keyword Search"""
    text = re.sub(r'\[.*?\]', '', text)
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return tokens

def build_rag_index(parsed_chunks, save_dir="./index_storage"):
    """
    Accepts a list of ParsedChunk objects (or dicts) with metadata.
    Builds FAISS HNSW and BM25 indexes, and saves them to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[Indexer] Using Hugging Face Inference API for MiniLM Embeddings")
    if not parsed_chunks:
        print("[Indexer] No chunks to index.")
        return None, None, {}

    valid_embeddings = []
    valid_chunks = [] 
    
    BATCH_SIZE = 10 
    
    # 1. SAFE EMBEDDING LOOP (Official SDK Version)
    for i in range(0, len(parsed_chunks), BATCH_SIZE):
        batch = parsed_chunks[i : i + BATCH_SIZE]
        batch_text = [chunk.content if hasattr(chunk, 'content') else chunk['content'] for chunk in batch]
        
        try:
            # Swapped to MiniLM - The most stable free model on Hugging Face
            batch_vectors = hf_client.feature_extraction(
                text=batch_text,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # The SDK might return a numpy array, ensure it's a list
            if isinstance(batch_vectors, np.ndarray):
                batch_vectors = batch_vectors.tolist()
            
            if len(batch_vectors) != len(batch_text):
                print(f"Mismatch in Batch {i}. Skipping batch to maintain alignment.")
                continue

            valid_embeddings.extend(batch_vectors)
            valid_chunks.extend(batch) 
            print(f"   -> Successfully embedded chunks {i} to {i+len(batch_text)-1} via Hugging Face SDK!")
            
        except Exception as e:
            print(f"Error embedding batch {i}: {e}")
            print("Tip: If the error is empty, the HF server is waking up. Try running it one more time.")
            continue

    if not valid_embeddings:
        print("CRITICAL: No embeddings generated.")
        return None, None, {}

    # 2. BUILD FAISS HNSW (Graph) INDEX
    dimension = len(valid_embeddings[0])
    np_embeddings = np.array(valid_embeddings).astype('float32')
    
    vector_index = faiss.IndexHNSWFlat(dimension, 32)
    vector_index.hnsw.efConstruction = 40
    vector_index.add(np_embeddings)
    
    faiss.write_index(vector_index, os.path.join(save_dir, "faiss_hnsw.index"))

    # 3. BUILD KEYWORD INDEX (BM25)
    print(f"[Indexer] Building BM25 Index...")
    tokenized_corpus = [simple_tokenize(chunk.content if hasattr(chunk, 'content') else chunk['content']) for chunk in valid_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    with open(os.path.join(save_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25_index, f)

    # 4. BUILD METADATA MAPPING
    chunk_map = {i: chunk for i, chunk in enumerate(valid_chunks)}
    
    with open(os.path.join(save_dir, "chunk_map.pkl"), "wb") as f:
        pickle.dump(chunk_map, f)
        
    print(f"[Indexer] Hybrid Index built and saved! Total docs: {len(valid_chunks)}")
    return vector_index, bm25_index, chunk_map

if __name__ == "__main__":
    print("\n--- Starting Indexer Test ---")
    
    mock_parsed_chunks = [
        {
            "id": 0, 
            "page_num": 1, 
            "chunk_type": "text", 
            "content": "The Tata Code of Conduct outlines the core values and ethical principles of the company.", 
            "metadata": {"source": "Tata_Code_Of_Conduct.pdf", "page": 1}
        },
        {
            "id": 1, 
            "page_num": 2, 
            "chunk_type": "table", 
            "content": "| Fault Code | Description |\n|---|---|\n| P0300 | Random/Multiple Cylinder Misfire Detected |\n| P0420 | Catalyst System Efficiency Below Threshold |", 
            "metadata": {"source": "Service_Manual.pdf", "page": 2}
        },
        {
            "id": 2, 
            "page_num": 3, 
            "chunk_type": "image", 
            "content": "A detailed flowchart showing the diagnostic steps for engine misfires. If the position is not precise, it loops back to the starter motor.", 
            "metadata": {"source": "Service_Manual.pdf", "page": 3, "image_file": "diagram_01.png"}
        }
    ]

    save_directory = "./index_storage"
    vector_idx, bm25_idx, chunk_mapping = build_rag_index(mock_parsed_chunks, save_dir=save_directory)

    if vector_idx and bm25_idx and chunk_mapping:
        print("\n✅ Verification Successful!")
        print(f"📁 Files saved in: {os.path.abspath(save_directory)}")
        print(f"   - faiss_hnsw.index exists: {os.path.exists(os.path.join(save_directory, 'faiss_hnsw.index'))}")
        print(f"   - bm25.pkl exists: {os.path.exists(os.path.join(save_directory, 'bm25.pkl'))}")
        print(f"   - chunk_map.pkl exists: {os.path.exists(os.path.join(save_directory, 'chunk_map.pkl'))}")
        print("\n🔍 Inspecting Metadata for Chunk ID 1:")
        print(f"Chunk Type: {chunk_mapping[1]['chunk_type']}")
        print(f"Source Page: {chunk_mapping[1]['metadata']['page']}")
        print(f"Content Preview: {chunk_mapping[1]['content'][:40]}...")
    else:
        print("\n❌ Indexing failed. Check your HF_TOKEN and internet connection.")