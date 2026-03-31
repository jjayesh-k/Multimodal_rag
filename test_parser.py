import os
from collections import Counter
from dotenv import load_dotenv

# Assuming you saved the parser code in src/parser.py
# If you saved it in the root directory, use: from parser import SmartMultiColumnParser
from src.parser import SmartMultiColumnParser

# Load environment variables (for OpenRouter API Key)
load_dotenv()

def test_multimodal_parser(pdf_path: str):
    print(f"--- Starting Parser Test on: {pdf_path} ---")
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find '{pdf_path}'. Please upload a sample PDF to your Codespace.")
        return

    # Initialize the parser
    parser = SmartMultiColumnParser(chunk_size=1000, chunk_overlap=200)
    
    # Run the extraction
    try:
        chunks = parser.parse_and_chunk(pdf_path, verbose=True)
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return

    # --- Print Statistics ---
    print("\n" + "="*50)
    print("📊 EXTRACTION SUMMARY")
    print("="*50)
    
    if not chunks:
        print("No chunks extracted. The PDF might be empty or corrupted.")
        return

    # Count the types of chunks we found
    type_counts = Counter([chunk.chunk_type for chunk in chunks])
    print(f"Total Chunks: {len(chunks)}")
    print(f"Text Chunks:  {type_counts.get('text', 0)}")
    print(f"Table Chunks: {type_counts.get('table', 0)}")
    print(f"Image Chunks: {type_counts.get('image', 0)}")

    # --- Print a Preview ---
    print("\n" + "="*50)
    print("🔍 PREVIEW OF EXTRACTED CHUNKS (Showing first 5)")
    print("="*50)
    
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n[{i+1}] Type: {chunk.chunk_type.upper()} | Page: {chunk.page_num}")
        print("-" * 40)
        
        # Truncate content for a cleaner terminal view
        content_preview = chunk.content[:300].replace('\n', ' ')
        if len(chunk.content) > 300:
            content_preview += " ... [TRUNCATED]"
            
        print(content_preview)

if __name__ == "__main__":
    # Ensure you have a PDF file named 'sample.pdf' in your workspace
    TEST_FILE = "/workspaces/Multimodal_rag/sample_documents/Tata Code Of Conduct.pdf" 
    
    # If OpenRouter key is missing, warn the user
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️ WARNING: OPENROUTER_API_KEY not found in .env.")
        print("Text and tables will be extracted, but Image/Scanned Page summaries will fail.\n")
        
    test_multimodal_parser(TEST_FILE)