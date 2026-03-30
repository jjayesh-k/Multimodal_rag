import pymupdf4llm
import fitz  # PyMuPDF
import re
import os
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ParsedChunk:
    id: int
    page_num: int
    chunk_type: str  # Will now accurately be 'text', 'table', or 'image'
    content: str
    metadata: Dict

class SmartMultiColumnParser:
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size
        self.chunk_counter = 0

    def _normalize(self, text):
        return re.sub(r'\s+', '', text).lower()

    def parse_and_chunk(self, pdf_path: str, verbose: bool = True) -> List[ParsedChunk]:
        if verbose: print(f"Parsing (Hybrid Mode): {pdf_path}")
        
        doc = fitz.open(pdf_path)
        
        # FIX #1: Enable image extraction in pymupdf4llm
        # This will save images to the output directory and reference them in the MD
        image_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images")
        os.makedirs(image_dir, exist_ok=True)
        
        md_pages = pymupdf4llm.to_markdown(
            pdf_path, 
            page_chunks=True,
            write_images=True, 
            image_path=image_dir
        )
        
        all_chunks = []
        self.chunk_counter = 0

        for i, md_data in enumerate(md_pages):
            page_num = i + 1
            smart_text = md_data['text']
            
            # --- LOST TEXT RECOVERY (Kept exactly as you wrote it) ---
            raw_page = doc[i]
            raw_blocks = raw_page.get_text("blocks", sort=True)
            missing_text = []
            smart_text_norm = self._normalize(smart_text)
            
            for b in raw_blocks:
                block_text = b[4].strip()
                if len(block_text) < 3: continue
                if self._normalize(block_text) not in smart_text_norm:
                    missing_text.append(block_text)

            final_page_content = smart_text
            if missing_text:
                recovered_str = "\n".join(missing_text)
                final_page_content += f"\n\n--- [ADDITIONAL NOTES] ---\n{recovered_str}"
                if verbose: print(f"   + Page {page_num}: Recovered {len(missing_text)} blocks.")

            # --- FIX #2: SEMANTIC CHUNKING ---
            # Split by double newline to preserve paragraphs and Markdown tables!
            blocks = final_page_content.split('\n\n')
            
            current_chunk_text = ""
            
            for block in blocks:
                block = block.strip()
                if not block: continue

                # Identify Chunk Type
                chunk_type = "text"
                if "|" in block and "-|-" in block:
                    chunk_type = "table"
                elif "![" in block and "](" in block:
                    chunk_type = "image" # We caught an image reference!

                # If the block itself is a table or image, save current text and isolate the table
                if chunk_type in ["table", "image"]:
                    if current_chunk_text:
                        all_chunks.append(self._create_chunk(current_chunk_text, page_num, "text"))
                        current_chunk_text = ""
                    all_chunks.append(self._create_chunk(block, page_num, chunk_type))
                    continue

                # Normal text batching (safely avoiding infinite loops)
                if len(current_chunk_text) + len(block) > self.max_chunk_size and current_chunk_text:
                    all_chunks.append(self._create_chunk(current_chunk_text, page_num, "text"))
                    current_chunk_text = block
                else:
                    current_chunk_text += "\n\n" + block if current_chunk_text else block

            # Catch leftover text
            if current_chunk_text:
                all_chunks.append(self._create_chunk(current_chunk_text, page_num, "text"))

        if verbose: print(f"Extracted {len(all_chunks)} chunks total.")
        return all_chunks

    def _create_chunk(self, text: str, page_num: int, chunk_type: str) -> ParsedChunk:
        chunk = ParsedChunk(
            id=self.chunk_counter,
            page_num=page_num,
            chunk_type=chunk_type,
            content=text.strip(),
            metadata={'page': page_num}
        )
        self.chunk_counter += 1
        return chunk

if __name__ == "__main__":
    pdf_filename = r"D:\JK\RAG\RAG_APP_FINAL\input_files\Tata Code Of Conduct.pdf" 

    if os.path.exists(pdf_filename):
        parser = SmartMultiColumnParser()
        result_chunks = parser.parse_and_chunk(pdf_filename)

        print(f"\n✅ Successfully created {len(result_chunks)} chunks.")
        
        # Preview specific types
        tables = [c for c in result_chunks if c.chunk_type == 'table']
        print(f"📊 Found {len(tables)} intact tables!")
    else:
        print(f"❌ File '{pdf_filename}' not found.")