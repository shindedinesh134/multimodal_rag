"""
Multimodal RAG System - FastAPI Entry Point
Domain: Engine Cooling System Technical Documentation
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router
from src.api.errors import setup_exception_handlers
from src.ingestion.ingest_pipeline import IngestionPipeline
from src.retrieval.rag_chain import RAGChain

# Global instances
ingestion_pipeline = None
rag_chain = None
startup_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global ingestion_pipeline, rag_chain, startup_time
    import datetime
    startup_time = datetime.datetime.now()
    
    logger.info("🚀 Starting Multimodal RAG System...")
    
    # Initialize pipelines
    ingestion_pipeline = IngestionPipeline()
    rag_chain = RAGChain()
    
    # Load existing vector store if available
    try:
        rag_chain.load_vector_store()
        logger.info("✅ Loaded existing vector store")
    except FileNotFoundError:
        logger.info("ℹ️ No existing vector store found - ready for ingestion")
    except Exception as e:
        logger.error(f"⚠️ Error loading vector store: {e}")
    
    yield
    
    # Shutdown cleanup
    logger.info("🛑 Shutting down...")
    if rag_chain and rag_chain.vector_store:
        rag_chain.vector_store.persist()

# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG System - Engine Cooling Documentation",
    description="""
    ## Intelligent Query System for Engine Cooling Technical Manuals
    
    This system enables semantic search and question answering across 
    multimodal technical documentation including:
    - **Text**: Installation procedures, specifications, troubleshooting
    - **Tables**: Coolant capacity charts, temperature thresholds, part numbers
    - **Images**: Cooling system diagrams, flow schematics, component photos
    
    ### Features
    - Upload PDF manuals via `/ingest`
    - Ask technical questions via `/query`
    - Get answers with source references (page, chunk type)
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup exception handlers
setup_exception_handlers(app)

# Include routes
app.include_router(router)

# Health check endpoint is in routes.py, but adding a root for convenience
@app.get("/")
async def root():
    return {
        "message": "Multimodal RAG System for Engine Cooling Documentation",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )