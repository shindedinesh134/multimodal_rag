# Multimodal RAG System for Engine Cooling Technical Documentation

## Problem Statement

### Domain Identification
**Domain:** Automotive Engineering / Thermal Management Systems

This system addresses information retrieval challenges in the automotive engineering domain, specifically focusing on engine cooling system documentation. My professional background involves working with heavy machinery and vehicle thermal management systems.

### Problem Description
Engine cooling system manuals present a significant information retrieval challenge for engineers, technicians, and maintenance personnel. These documents are inherently multimodal, containing:

- **Text content**: Installation procedures, troubleshooting guides, safety warnings, and technical specifications
- **Tables**: Coolant capacity charts, temperature thresholds, pressure specifications, and component compatibility matrices
- **Images**: Cooling system flow diagrams, component cross-sections, hose routing schematics, and exploded assembly views

Currently, technicians face several pain points:
1. **Manual searching**: Finding specific information in 200+ page PDFs requires scanning each document manually
2. **Cross-modal queries**: Questions like "Show me the temperature warnings from the diagram on page 15" require correlating images with text
3. **Table interpretation**: Extracting specific values from capacity charts or specification tables is error-prone
4. **Context switching**: Moving between different sections of a manual disrupts workflow and increases diagnosis time

### Why This Problem Is Unique
This problem differs from generic document Q&A in several critical ways:

1. **Technical terminology**: Engine cooling systems use specialized terms like "thermostat cracking temperature," "water pump flow coefficient," and "coolant freeze point" that generic models struggle with

2. **Table complexity**: Cooling specifications often contain nested headers, merged cells, and footnotes with conditional values (e.g., "at 50% glycol concentration")

3. **Diagram semantics**: Cooling system diagrams contain flow arrows, temperature gradients, and component labels that must be interpreted correctly

4. **Regulatory context**: Many cooling systems must meet specific emissions or safety standards, adding another layer of complexity

5. **Numerical precision**: Temperature thresholds, pressure ratings, and mixture ratios require exact values - approximations are unacceptable

### Why RAG Is the Right Approach
RAG is superior to alternatives for this use case because:

| Alternative | Limitation |
|-------------|-------------|
| **Fine-tuning** | Would require thousands of labeled Q&A pairs per document; cannot handle new documents without retraining; loses ability to cite exact sources |
| **Keyword search** | Cannot handle semantic queries like "which engine has the largest cooling capacity?"; misses context-dependent information |
| **Manual search** | Time-prohibitive for large document sets; inconsistent results across different technicians |

**RAG advantages for this domain:**
- **Grounding**: Answers are based on actual documentation, reducing hallucination risks for safety-critical information
- **Source citation**: Technicians can verify answers against original pages - crucial for compliance
- **Cross-modal retrieval**: Can combine text, table, and image information in a single query
- **Document flexibility**: New manuals can be added without retraining the model

### Expected Outcomes
A successful system will enable:

**Query Examples:**
- "What is the recommended coolant mixture ratio for the XE-200 engine at -20°C ambient temperature?"
- "Show me the pressure drop values from the cooling system specifications table"
- "Explain the coolant flow path shown in Figure 3 - what components does it pass through?"
- "Which engines in the document have a coolant capacity greater than 15 liters?"
- "What are the troubleshooting steps for overheating with code P0128?"

**Supported Decisions:**
- Selecting appropriate coolant type for specific operating conditions
- Diagnosing cooling system failures based on symptom patterns
- Determining replacement part compatibility from specification tables
- Validating system design against temperature/pressure requirements

---

## Architecture Overview

```mermaid
graph TB
    subgraph "Ingestion Pipeline"
        A[PDF Document] --> B[PDF Parser]
        B --> C[Text Extraction]
        B --> D[Table Extraction]
        B --> E[Image Extraction]
        E --> F[VLM Processing]
        F --> G[Image Summaries]
        D --> H[Table Processor]
        H --> I[NL Descriptions]
        C --> J[Text Chunker]
        I --> J
        G --> J
        J --> K[Embedding Model]
        K --> L[(Vector Store - FAISS)]
    end
    
    subgraph "Query Pipeline"
        M[User Question] --> N[Embedding Model]
        N --> O[Similarity Search]
        O --> L
        L --> P[Retrieved Chunks]
        P --> Q[Context Builder]
        Q --> R[LLM Generation]
        R --> S[Grounded Answer]
        P --> T[Source Citations]
    end
    
    subgraph "API Layer"
        U[FastAPI Server] --> V[/health]
        U --> W[/ingest]
        U --> X[/query]
        U --> Y[/docs]
    end