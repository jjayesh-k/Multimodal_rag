# Multimodal RAG System for Automotive Diagnostics

**Course:** Multimodal Retrieval-Augmented Generation Bootcamp (BITS Pilani WILP)  
**Submission Type:** Individual Assignment  

---

## 1. Problem Statement

**Domain Identification**
This project is situated within the domain of Automotive Engineering and Vehicle Diagnostics, specifically focusing on the optimization of service and maintenance operations. As modern vehicles transition into complex, software-defined machines containing dozens of interconnected Electronic Control Units (ECUs), the maintenance and diagnostic processes have become exponentially more difficult.

**Problem Description**
Automotive technicians and mechanical engineers currently face a massive information retrieval bottleneck. When diagnosing a vehicle malfunction, technicians must consult OEM (Original Equipment Manufacturer) service manuals. These manuals are incredibly dense, often spanning thousands of pages for a single vehicle model. The critical problem is that the data within these manuals is highly multimodal. A single diagnostic procedure might require the technician to read procedural text, cross-reference a Diagnostic Trouble Code (DTC) in a complex table, and trace a wiring path on an exploded isometric diagram of an engine block. Currently, technicians waste hours manually scrolling through PDFs because traditional Ctrl+F keyword search is entirely blind to the data locked inside tables and images.

**Why This Problem Is Unique**
This domain presents unique challenges that distinguish it from standard document Q&A systems. First, automotive data requires absolute precision; a hallucinated torque specification (e.g., 150 Nm instead of 15 Nm) can destroy an engine block. Second, the terminology is highly specialized and heavily reliant on specific alphanumeric strings (e.g., "P0420 Catalyst System Efficiency Below Threshold"). Dense vector embeddings often struggle with exact alphanumeric matching, making standard semantic search unreliable. Finally, the visual data is not decorative—it is functional. An engine schematic or a flowchart dictating diagnostic logic (e.g., "If voltage < 5V, check relay A") contains explicit operational data that must be extracted and synthesized with the surrounding text. 

**Why RAG Is the Right Approach**
A Multimodal Retrieval-Augmented Generation (RAG) approach is the only viable solution for this problem. Alternative approaches, such as fine-tuning a Large Language Model on the service manuals, are fundamentally flawed for this use case. Fine-tuned models are prone to hallucinating exact numerical specifications and cannot easily cite the exact page number of a schematic, which a technician requires for verification. Furthermore, manuals are updated annually; retraining a model for every new car release is prohibitively expensive. RAG solves this by decoupling the knowledge base from the reasoning engine. By parsing the multimodal documents into a searchable vector index, the system can dynamically retrieve the exact table, text, and diagram summary needed, passing them to the LLM strictly as contextual grounding. This ensures zero-hallucination answers backed by verifiable source citations.

**Expected Outcomes**
A successful implementation of this Multimodal RAG system will transform diagnostic workflows. It will allow a technician to ask natural language questions such as, *"What is the diagnostic procedure for a P0300 code, and where is the corresponding sensor located?"* The system will retrieve the procedural text, the specific row from the DTC table, and the text summary of the engine bay diagram, synthesizing them into a clear, actionable instruction. It will drastically reduce diagnostic time, minimize human error in reading torque specifications, and support rapid decision-making on the workshop floor.

---

## 2. Architecture Overview

The system utilizes a hybrid local/cloud architecture. Document parsing, hybrid indexing (FAISS + BM25), and vector storage run locally, while heavy LLM/VLM reasoning is offloaded to OpenRouter APIs.

```mermaid
graph TD
    subgraph Ingestion Pipeline
        A[Automotive PDF] -->|PyMuPDF/Docling| B(Document Parser)
        B --> C{Chunk Type}
        C -->|Text/Paragraphs| D[Semantic Text Chunks]
        C -->|Markdown Tables| E[Table Chunks]
        C -->|Extracted Images| F[Vision Pre-Filter]
        
        F -->|Prompt: Reject Logos| G[Junk/Decorative]
        F -->|Prompt: Extract Data| H[Diagram/Flowchart Summaries]
        G -->|Discarded| Z((Trash))
        
        D --> I[Local Embedding Model]
        E --> I
        H --> I
        
        I --> J[(FAISS HNSW Index)]
        I --> K[(BM25 Sparse Index)]
    end

    subgraph Query Pipeline
        L[User Query] --> M[FastAPI /query]
        M --> N[Embedding Model]
        N --> O{Hybrid Retriever}
        
        O -->|Semantic| J
        O -->|Keyword| K
        
        O --> P[Cross-Modal Context Merging]
        P --> Q[Generation LLM]
        Q --> R[Grounded Answer + Citations]
    end