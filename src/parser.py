"""
Multimodal PDF Parser (Text + Tables + Images + Scanned Pages)
==============================================================
Upgrades:
1. Detects scanned pages and passes the whole page to a Vision Model.
2. Extracts embedded images and generates text summaries via VLM.
3. Automatically categorizes chunk_type as 'text', 'table', or 'image'.
"""

import pymupdf4llm
import fitz  # PyMuPDF
import re
import os
import base64
import requests
from typing import List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Use a fast, free vision model for image summarization
VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

@dataclass
class ParsedChunk:
    id: int
    page_num: int
    chunk_type: str # 'text', 'table', or 'image'
    content: str
    metadata: Dict

def summarize_image_with_vlm(base64_image: str, prompt: str = "Describe this image in detail. If it is a scanned document or table, extract all the text and data.") -> str:
    """Sends a base64 image to OpenRouter Vision Model for summarization."""
    if not OPENROUTER_API_KEY:
        return "Image description unavailable. Missing API Key."
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"VLM Error: {response.status_code} - {response.text}")
            return "Failed to generate image summary."
    except Exception as e:
        print(f"VLM Connection Error: {e}")
        return "Failed to connect to Vision API."


class SmartMultiColumnParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_counter = 0

    def _normalize(self, text):
        return re.sub(r'\s+', '', text).lower()

    def parse_and_chunk(self, pdf_path: str, verbose: bool = True) -> List[ParsedChunk]:
        if verbose: print(f"Parsing Multimodal Document: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        all_chunks = []
        self.chunk_counter = 0

        for i, md_data in enumerate(md_pages):
            page_num = i + 1
            smart_text = md_data['text']
            raw_page = doc[i]
            
            # --- 1. SCANNED PAGE DETECTION ---
            # If the page has almost no text, treat the whole page as an image
            raw_text = raw_page.get_text("text").strip()
            if len(raw_text) < 50:
                if verbose: print(f"Page {page_num} appears to be a scanned image. Sending to VLM...")
                pix = raw_page.get_pixmap(dpi=150) # Render page to image
                img_data = pix.tobytes("jpeg")
                b64_img = base64.b64encode(img_data).decode('utf-8')
                
                summary = summarize_image_with_vlm(b64_img, "This is a scanned document page. Extract and summarize all readable text, tables, and visual information.")
                
                all_chunks.append(ParsedChunk(
                    id=self.chunk_counter, page_num=page_num, chunk_type="image",
                    content=f"[SCANNED PAGE SUMMARY]\n{summary}", metadata={'page': page_num}
                ))
                self.chunk_counter += 1
                continue # Skip normal text extraction for this page

            # --- 2. EXTRACT EMBEDDED IMAGES ---
            image_list = raw_page.get_images(full=True)
            if image_list:
                if verbose: print(f"Page {page_num}: Found {len(image_list)} embedded images. Summarizing...")
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    b64_img = base64.b64encode(image_bytes).decode('utf-8')
                    
                    summary = summarize_image_with_vlm(b64_img, "Describe this diagram, chart, or picture in detail. What information does it convey?")
                    
                    all_chunks.append(ParsedChunk(
                        id=self.chunk_counter, page_num=page_num, chunk_type="image",
                        content=f"[IMAGE SUMMARY]\n{summary}", metadata={'page': page_num}
                    ))
                    self.chunk_counter += 1

            # --- 3. TEXT & TABLE RECOVERY (Your existing logic) ---
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
                final_page_content += "\n\n--- [ADDITIONAL NOTES] ---\n" + "\n".join(missing_text)

            # --- 4. CHUNKING & TYPE TAGGING ---
            page_chunks = self._create_sliding_window_chunks(final_page_content, page_num)
            all_chunks.extend(page_chunks)

        if verbose: print(f"Extracted {len(all_chunks)} total chunks.")
        return all_chunks

    def _create_sliding_window_chunks(self, text: str, page_num: int) -> List[ParsedChunk]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1: end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                # Simple heuristic: If it contains Markdown table formatting, tag it as a table
                c_type = "table" if "|---" in chunk_text or "|:" in chunk_text else "text"
                
                chunks.append(ParsedChunk(
                    id=self.chunk_counter, page_num=page_num, chunk_type=c_type,
                    content=chunk_text, metadata={'page': page_num}
                ))
                self.chunk_counter += 1
            
            start = end - self.chunk_overlap
            if start >= end: start = end 

        return chunks