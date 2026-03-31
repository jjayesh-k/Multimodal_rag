"""
Multimodal PDF Parser (Text + Tables + Images + Scanned Pages)
==============================================================
Upgrades:
1. Detects scanned pages and passes the whole page to a Vision Model.
2. Extracts embedded images and generates text summaries via VLM.
3. Automatically categorizes chunk_type as 'text', 'table', or 'image'.
4. Groq-only Vision pipeline with smart 429 retry + model failover.
"""

import pymupdf4llm
import fitz  # PyMuPDF
import re
import os
import time
import base64
import requests
from typing import List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Groq Free Vision Models (failover order) ---
# Maverick was deprecated March 9 2026; replaced by gpt-oss-120b.
# Scout is the only free-tier vision model as of March 2026.
# gpt-oss-120b is the paid fallback if you ever upgrade.
GROQ_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Primary: free, 30k TPM
    "openai/gpt-oss-120b",                         # Fallback: paid tier only
]

# How many times to retry a 429 before moving to the next model
MAX_RETRIES_ON_429 = 3


@dataclass
class ParsedChunk:
    id: int
    page_num: int
    chunk_type: str  # 'text', 'table', or 'image'
    content: str
    metadata: Dict


def _parse_retry_after(response: requests.Response) -> float:
    """
    Extracts how many seconds to wait from a 429 response.
    Groq returns the wait time in the error message (e.g. 'try again in 5.57s')
    and also in the x-ratelimit-reset-tokens header.
    Falls back to 10s if neither is parseable.
    """
    # Try the response header first (most reliable)
    reset_header = response.headers.get("x-ratelimit-reset-tokens", "")
    if reset_header:
        try:
            # Header is like "5.57s" or "176ms"
            if reset_header.endswith("ms"):
                return float(reset_header[:-2]) / 1000 + 0.5
            elif reset_header.endswith("s"):
                return float(reset_header[:-1]) + 0.5
        except ValueError:
            pass

    # Fall back to parsing the message body
    try:
        msg = response.json()["error"]["message"]
        match = re.search(r"try again in (\d+(?:\.\d+)?)(\w+)", msg)
        if match:
            value, unit = float(match.group(1)), match.group(2)
            return (value / 1000 + 0.5) if unit == "ms" else (value + 0.5)
    except Exception:
        pass

    return 10.0  # safe default


def summarize_image_with_vlm(
    base64_image: str,
    prompt: str = "Describe this image in detail. If it is a scanned document or table, extract all the text and data."
) -> str:
    """
    Sends a base64 image to Groq Vision models.
    - On 429: waits the exact retry-after duration, retries the SAME model up to MAX_RETRIES_ON_429 times.
    - On other errors: moves to next model in list.
    """
    if not GROQ_API_KEY:
        return "Image description unavailable: GROQ_API_KEY not set."

    messages_payload = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]

    groq_headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    for model in GROQ_VISION_MODELS:
        retries = 0
        while retries <= MAX_RETRIES_ON_429:
            try:
                print(f"[Parser] Attempting Vision Extraction with Groq ({model})...")
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=groq_headers,
                    json={"model": model, "messages": messages_payload, "temperature": 0.1},
                    timeout=60
                )

                if response.status_code == 200:
                    print("   -> Groq Vision Extraction Successful!")
                    return response.json()["choices"][0]["message"]["content"]

                elif response.status_code == 429:
                    wait = _parse_retry_after(response)
                    retries += 1
                    if retries <= MAX_RETRIES_ON_429:
                        print(f"   -> Rate limited (429). Waiting {wait:.1f}s before retry {retries}/{MAX_RETRIES_ON_429}...")
                        time.sleep(wait)
                    else:
                        print(f"   -> Rate limit retries exhausted for {model}. Trying next model...")
                        break  # move to next model

                else:
                    print(f"   -> Failed (Status {response.status_code}): {response.text}")
                    break  # non-retryable error, move to next model

            except Exception as e:
                print(f"   -> Connection Error: {e}")
                break  # move to next model

    return "Image description unavailable. All Groq Vision models exhausted or failed."


class SmartMultiColumnParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_counter = 0

    def _normalize(self, text):
        return re.sub(r'\s+', '', text).lower()

    def parse_and_chunk(self, pdf_path: str, verbose: bool = True) -> List[ParsedChunk]:
        if verbose:
            print(f"Parsing Multimodal Document: {pdf_path}")

        doc = fitz.open(pdf_path)
        md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

        all_chunks = []
        self.chunk_counter = 0

        for i, md_data in enumerate(md_pages):
            page_num = i + 1
            smart_text = md_data['text']
            raw_page = doc[i]

            # --- 1. SCANNED PAGE DETECTION ---
            raw_text = raw_page.get_text("text").strip()
            if len(raw_text) < 50:
                if verbose:
                    print(f"Page {page_num} appears to be a scanned image. Sending to Groq Vision...")
                pix = raw_page.get_pixmap(dpi=150)
                img_data = pix.tobytes("jpeg")
                b64_img = base64.b64encode(img_data).decode('utf-8')

                summary = summarize_image_with_vlm(
                    b64_img,
                    "This is a scanned document page. Extract and summarize all readable text, tables, and visual information."
                )

                all_chunks.append(ParsedChunk(
                    id=self.chunk_counter, page_num=page_num, chunk_type="image",
                    content=f"[SCANNED PAGE SUMMARY]\n{summary}", metadata={'page': page_num}
                ))
                self.chunk_counter += 1
                continue

            # --- 2. EXTRACT EMBEDDED IMAGES ---
            image_list = raw_page.get_images(full=True)
            if image_list:
                if verbose:
                    print(f"Page {page_num}: Found {len(image_list)} embedded images. Summarizing...")
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    b64_img = base64.b64encode(image_bytes).decode('utf-8')

                    summary = summarize_image_with_vlm(
                        b64_img,
                        "Describe this diagram, chart, or picture in detail. What information does it convey?"
                    )

                    all_chunks.append(ParsedChunk(
                        id=self.chunk_counter, page_num=page_num, chunk_type="image",
                        content=f"[IMAGE SUMMARY]\n{summary}", metadata={'page': page_num}
                    ))
                    self.chunk_counter += 1

            # --- 3. TEXT & TABLE RECOVERY ---
            raw_blocks = raw_page.get_text("blocks", sort=True)
            missing_text = []
            smart_text_norm = self._normalize(smart_text)

            for b in raw_blocks:
                block_text = b[4].strip()
                if len(block_text) < 3:
                    continue
                if self._normalize(block_text) not in smart_text_norm:
                    missing_text.append(block_text)

            final_page_content = smart_text
            if missing_text:
                final_page_content += "\n\n--- [ADDITIONAL NOTES] ---\n" + "\n".join(missing_text)

            # --- 4. CHUNKING & TYPE TAGGING ---
            page_chunks = self._create_sliding_window_chunks(final_page_content, page_num)
            all_chunks.extend(page_chunks)

        if verbose:
            print(f"Extracted {len(all_chunks)} total chunks.")
        return all_chunks

    def _create_sliding_window_chunks(self, text: str, page_num: int) -> List[ParsedChunk]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                c_type = "table" if "|---" in chunk_text or "|:" in chunk_text else "text"

                chunks.append(ParsedChunk(
                    id=self.chunk_counter, page_num=page_num, chunk_type=c_type,
                    content=chunk_text, metadata={'page': page_num}
                ))
                self.chunk_counter += 1

            start = end - self.chunk_overlap
            if start >= end:
                start = end

        return chunks