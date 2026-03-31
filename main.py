from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
import time
from dotenv import load_dotenv
import shutil
from collections import Counter

# Import our highly tuned hybrid retriever
import src.retriever as retriever
from src.retriever import perform_hybrid_search, load_indexes
from src.parser import SmartMultiColumnParser
from src.indexer import build_rag_index

SERVER_START_TIME = time.time()

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL") # Fetched from your .env (ngrok URL)

# --- 1. Failover Model List ---
FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3n-e2b-it:free",
    "qwen/qwen3-coder:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "openai/gpt-oss-120b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free"
]

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class SourceCitation(BaseModel):
    chunk_type: str
    page: str
    score: float
    preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    model_used: str 

# --- Initialize FastAPI ---
app = FastAPI(
    title="Automotive Multimodal RAG API",
    description="Hybrid Search with Cloud-to-Local Failover",
    version="1.2.0"
)

@app.on_event("startup")
async def startup_event():
    print("Starting FastAPI Server...")
    success = load_indexes("./index_storage")
    if not success:
        print("WARNING: Indexes not found.")

def get_index_size_mb(directory_path: str = "./index_storage") -> float:
    """Calculates the total size of a directory in Megabytes."""
    total_size = 0
    if os.path.exists(directory_path):
        for dirpath, _, filenames in os.walk(directory_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)

@app.get("/health")
async def health_check():
    """
    Returns system status: model readiness, number of indexed documents, 
    index size, and server uptime.
    """
    # 1. Calculate Uptime
    uptime_seconds = round(time.time() - SERVER_START_TIME, 2)
    
    # 2. Check Model Readiness
    # If VECTOR_INDEX is not None, the Granite embedder and FAISS are loaded
    is_ready = retriever.VECTOR_INDEX is not None
    
    # 3. Get Indexed Document (Chunk) Count
    # Safely check the length of the CHUNK_MAP dictionary
    num_chunks = len(retriever.CHUNK_MAP) if retriever.CHUNK_MAP else 0
    
    # 4. Calculate Index Size on Disk
    index_size_mb = get_index_size_mb("./index_storage")

    return {
        "status": "online",
        "system": "Automotive RAG Pipeline active",
        "model_readiness": "ready" if is_ready else "not_loaded",
        "number_of_indexed_chunks": num_chunks,
        "index_size_mb": index_size_mb,
        "uptime_seconds": uptime_seconds
    }

# Create a standalone function for the heavy lifting
def process_pdf_in_background(temp_pdf_path: str, filename: str):
    try:
        print(f"\nBackground Processing Started for: {filename}")
        start_time = time.time()

        # 1. Parse and Extract
        parser = SmartMultiColumnParser()
        extracted_chunks = parser.parse_and_chunk(temp_pdf_path, verbose=True)
        
        if not extracted_chunks:
            print("Failed to extract any content.")
            return

        # 2. Embed and Index
        print("\nSending chunks to Vector Indexer...")
        build_rag_index(extracted_chunks, save_dir="./index_storage")

        # 3. Hot-Reload
        print("Reloading retriever indexes into memory...")
        load_indexes(save_dir="./index_storage")

        processing_time = round(time.time() - start_time, 2)
        print(f"SUCCESS: Ingestion complete in {processing_time} seconds!")

    except Exception as e:
        print(f"Background Ingestion Error: {e}")
        
    finally:
        # Cleanup
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print("Cleaned up temporary files.")


# Your new, lightning-fast endpoint
@app.post("/ingest")
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a PDF and hands it off to a background worker to prevent 504 Timeouts.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_pdf_path = f"temp_{file.filename}"
    
    # Save the file quickly
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Hand the heavy lifting to FastAPI's background worker
    background_tasks.add_task(process_pdf_in_background, temp_pdf_path, file.filename)
    
    # Return immediately to the user
    return {
        "message": f"File '{file.filename}' received successfully! Processing has started in the background.",
        "status": "Processing",
        "tip": "Check your terminal to see the live extraction progress."
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    # 1. Retrieve the best multimodal chunks
    retrieved_chunks = perform_hybrid_search(request.question, k=request.top_k)
    
    if not retrieved_chunks:
        return QueryResponse(
            answer="I could not find any relevant information in the diagnostic manuals.",
            sources=[],
            model_used="None"
        )

    # 2. Format the context for the LLM
    context_text = ""
    sources = []
    for i, chunk in enumerate(retrieved_chunks):
        page_num = str(chunk.get("metadata", {}).get("page", "Unknown"))
        c_type = chunk.get("chunk_type", "text")
        content = chunk.get("content", "")
        
        context_text += f"\n--- SOURCE {i+1} (Type: {c_type}, Page: {page_num}) ---\n{content}\n"
        sources.append(SourceCitation(
            chunk_type=c_type, page=page_num,
            score=chunk.get("search_score", 0.0),
            preview=content[:100] + "..."
        ))

    # 3. System Prompt
    system_prompt = """You are an expert Automotive Diagnostic AI. 
    Answer the user's question using ONLY the provided context. 
    Explicitly cite the Page Number (e.g., "According to Page 2...").
    If the context doesn't have the answer, say you don't know."""

    # 4. Phase 1: Cloud Failover Loop (OpenRouter)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/jjayesh-k",
        "X-Title": "Automotive Multimodal RAG"
    }

    last_error = ""

    for model_id in FREE_MODELS:
        try:
            print(f"[LLM-Cloud] Attempting: {model_id}")
            
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {request.question}"}
                ],
                "temperature": 0.1
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=20 
            )

            if response.status_code == 200:
                answer_text = response.json()["choices"][0]["message"]["content"]
                print(f"Cloud Success: {model_id}")
                return QueryResponse(
                    answer=answer_text,
                    sources=sources,
                    model_used=f"Cloud: {model_id}"
                )
            else:
                print(f"Cloud {model_id} busy (Status {response.status_code}).")
                last_error = response.text
                continue

        except Exception as e:
            print(f"Cloud Connection error: {e}")
            last_error = str(e)
            continue

    # 5. Phase 2: Local Fallback (Ollama @ Home)
    # This runs only if the entire Cloud loop fails
    if LOCAL_OLLAMA_URL:
        try:
            print(f"Cloud exhausted. Phoning home to Local RTX 4060...")
            
            # The secret key to bypassing the ngrok warning page
            local_headers = {
                "ngrok-skip-browser-warning": "true"
            }

            # Ollama native generation format
            ollama_payload = {
                "model": "mistral:7b",
                "prompt": f"SYSTEM: {system_prompt}\n\nCONTEXT:\n{context_text}\n\nUSER QUESTION: {request.question}",
                "stream": False
            }

            local_response = requests.post(
                f"{LOCAL_OLLAMA_URL}/api/generate",
                headers=local_headers,
                json=ollama_payload,
                timeout=60 # Local inference can take a bit longer than cloud
            )

            if local_response.status_code == 200:
                answer_text = local_response.json().get("response", "")
                print(f"Local Fallback Success: mistral:7b")
                return QueryResponse(
                    answer=answer_text,
                    sources=sources,
                    model_used="Local: mistral:7b (via Ngrok)"
                )
            else:
                print(f"Local Ollama returned error: {local_response.status_code}")
                last_error += f" | Local Error: {local_response.text}"
        except Exception as e:
            print(f"Local Fallback failed: {e}")
            last_error += f" | Local Exception: {str(e)}"

    # 6. Final Exception if both Cloud and Local are unreachable
    raise HTTPException(
        status_code=503, 
        detail=f"All resources exhausted. Last recorded error: {last_error}"
    )