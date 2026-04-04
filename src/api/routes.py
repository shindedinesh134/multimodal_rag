"""
FastAPI route definitions
"""

import os
import time
import tempfile
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from .schemas import (
    IngestionResponse, QueryRequest, QueryResponse, 
    HealthResponse, ErrorResponse, DocumentInfo
)
from .dependencies import get_ingestion_pipeline, get_rag_chain, get_system_stats

# Global router
router = APIRouter(tags=["RAG System"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    rag_chain = Depends(get_rag_chain),
    stats = Depends(get_system_stats)
):
    """
    System health endpoint.
    Returns current status including indexed documents and model readiness.
    """
    try:
        vector_store_status = "ready" if rag_chain.vector_store else "not_initialized"
        index_size = 0
        chunks_by_type = {"text": 0, "table": 0, "image": 0}
        
        if rag_chain.vector_store and hasattr(rag_chain.vector_store, 'index'):
            index_size = rag_chain.vector_store.get_index_size()
            chunks_by_type = rag_chain.vector_store.get_chunk_type_counts()
        
        uptime = (datetime.now() - stats["startup_time"]).total_seconds()
        
        return HealthResponse(
            status="healthy",
            indexed_documents=stats.get("document_count", 0),
            total_chunks=index_size,
            chunks_by_type=chunks_by_type,
            vector_store_size_bytes=None,  # Could add if needed
            uptime_seconds=uptime,
            model_status={
                "llm": "ready" if rag_chain.llm else "unavailable",
                "embedder": "ready" if rag_chain.embeddings else "unavailable",
                "vector_store": vector_store_status
            },
            last_ingestion=stats.get("last_ingestion")
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    file: UploadFile = File(..., description="PDF document to ingest"),
    ingestion_pipeline = Depends(get_ingestion_pipeline),
    rag_chain = Depends(get_rag_chain)
):
    """
    Upload and ingest a multimodal PDF document.
    
    The PDF can contain:
    - Text paragraphs and headings
    - Tables (coolant specs, temperature charts, part numbers)
    - Images (diagrams, schematics, component photos)
    
    Returns summary of extracted chunks by type.
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Validate file size (50MB max)
    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: 50MB"
        )
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(contents)
        tmp_path = tmp_file.name
    
    try:
        # Parse document
        logger.info(f"📄 Processing document: {file.filename}")
        parsed_doc = ingestion_pipeline.parse_document(tmp_path, file.filename)
        
        # Process images through VLM
        logger.info(f"🖼️ Processing {len(parsed_doc['images'])} images with VLM...")
        parsed_doc = ingestion_pipeline.process_images(parsed_doc)
        
        # Process tables
        logger.info(f"📊 Processing {len(parsed_doc['tables'])} tables...")
        parsed_doc = ingestion_pipeline.process_tables(parsed_doc)
        
        # Create chunks
        logger.info("✂️ Creating chunks...")
        chunks = ingestion_pipeline.create_chunks(parsed_doc)
        
        # Generate embeddings and add to vector store
        logger.info(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        chunk_ids = rag_chain.add_chunks(chunks)
        
        processing_time = time.time() - start_time
        
        # Get updated stats
        index_size = rag_chain.vector_store.get_index_size() if rag_chain.vector_store else 0
        
        return IngestionResponse(
            filename=file.filename,
            total_chunks=len(chunks),
            text_chunks=sum(1 for c in chunks if c['type'] == 'text'),
            table_chunks=sum(1 for c in chunks if c['type'] == 'table'),
            image_chunks=sum(1 for c in chunks if c['type'] == 'image'),
            processing_time_seconds=round(processing_time, 2),
            status="success",
            vector_store_size=index_size,
            message=f"Successfully ingested {file.filename}"
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Cleanup temp file
        os.unlink(tmp_path)


@router.post("/query", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    rag_chain = Depends(get_rag_chain)
):
    """
    Ask a question about the engine cooling system documentation.
    
    The system will:
    1. Retrieve relevant chunks (text, tables, image summaries)
    2. Generate a grounded answer using the LLM
    3. Provide source references for verification
    
    Example questions:
    - "What is the recommended coolant mixture ratio for the XE-200 engine?"
    - "Show me the temperature thresholds from the cooling system specifications table"
    - "Explain the coolant flow path shown in the system diagram"
    - "What are the troubleshooting steps for overheating?"
    """
    start_time = time.time()
    
    # Validate vector store exists
    if not rag_chain.vector_store or rag_chain.vector_store.get_index_size() == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents have been ingested yet. Please upload a PDF via /ingest first."
        )
    
    try:
        # Retrieve relevant chunks
        retrieved_chunks = rag_chain.retrieve(request.question, top_k=request.top_k)
        
        if not retrieved_chunks:
            return QueryResponse(
                question=request.question,
                answer="I couldn't find any relevant information in the ingested documents to answer this question.",
                sources=[],
                context_used=[],
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
                model_used="none"
            )
        
        # Generate answer
        answer_data = rag_chain.generate_answer(
            question=request.question,
            chunks=retrieved_chunks
        )
        
        # Format sources
        sources = []
        if request.include_sources:
            for chunk in retrieved_chunks:
                sources.append({
                    "filename": chunk.get("filename", "unknown"),
                    "page_number": chunk.get("page", 0),
                    "chunk_type": chunk.get("type", "text"),
                    "content_preview": chunk.get("content", "")[:200],
                    "similarity_score": chunk.get("score", 0.0)
                })
        
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return QueryResponse(
            question=request.question,
            answer=answer_data["answer"],
            sources=sources if sources else None,
            context_used=[c.get("content", "")[:500] for c in retrieved_chunks[:3]],
            processing_time_ms=processing_time_ms,
            model_used=answer_data.get("model", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    rag_chain = Depends(get_rag_chain)
):
    """
    List all ingested documents and their chunk counts.
    """
    if not rag_chain.vector_store:
        return []
    
    try:
        documents = rag_chain.vector_store.get_document_stats()
        return documents
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    rag_chain = Depends(get_rag_chain)
):
    """
    Delete a document from the vector store.
    """
    if not rag_chain.vector_store:
        raise HTTPException(status_code=404, detail="No vector store found")
    
    try:
        deleted_count = rag_chain.vector_store.delete_document(filename)
        return JSONResponse(
            status_code=200,
            content={"message": f"Deleted {deleted_count} chunks from {filename}"}
        )
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats(
    rag_chain = Depends(get_rag_chain)
):
    """Get detailed system statistics"""
    if not rag_chain.vector_store:
        return {"status": "no_vector_store"}
    
    return {
        "total_chunks": rag_chain.vector_store.get_index_size(),
        "chunks_by_type": rag_chain.vector_store.get_chunk_type_counts(),
        "documents": rag_chain.vector_store.get_document_stats()
    }