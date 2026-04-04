"""
Dependency injection for FastAPI routes
"""

from datetime import datetime
from typing import Dict, Any

# Global instances (set in main.py)
_ingestion_pipeline = None
_rag_chain = None
_startup_time = None
_document_count = 0
_last_ingestion = None


def set_globals(ingestion_pipeline, rag_chain, startup_time):
    """Set global instances from main"""
    global _ingestion_pipeline, _rag_chain, _startup_time
    _ingestion_pipeline = ingestion_pipeline
    _rag_chain = rag_chain
    _startup_time = startup_time


def update_stats(document_count=None, last_ingestion=None):
    """Update system statistics"""
    global _document_count, _last_ingestion
    if document_count is not None:
        _document_count = document_count
    if last_ingestion is not None:
        _last_ingestion = last_ingestion


async def get_ingestion_pipeline():
    """Dependency to get ingestion pipeline instance"""
    if _ingestion_pipeline is None:
        from src.ingestion.ingest_pipeline import IngestionPipeline
        return IngestionPipeline()
    return _ingestion_pipeline


async def get_rag_chain():
    """Dependency to get RAG chain instance"""
    if _rag_chain is None:
        from src.retrieval.rag_chain import RAGChain
        return RAGChain()
    return _rag_chain


async def get_system_stats() -> Dict[str, Any]:
    """Dependency to get system statistics"""
    return {
        "startup_time": _startup_time or datetime.now(),
        "document_count": _document_count,
        "last_ingestion": _last_ingestion
    }