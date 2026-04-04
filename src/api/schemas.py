"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class IngestionResponse(BaseModel):
    """Response for /ingest endpoint"""
    filename: str
    total_chunks: int
    text_chunks: int
    table_chunks: int
    image_chunks: int
    processing_time_seconds: float
    status: str = "success"
    vector_store_size: int
    message: Optional[str] = None


class SourceReference(BaseModel):
    """Source reference for RAG answer"""
    filename: str
    page_number: int
    chunk_type: ChunkType
    content_preview: str
    similarity_score: float


class QueryRequest(BaseModel):
    """Request for /query endpoint"""
    question: str = Field(..., min_length=3, max_length=500, 
                          description="Natural language question about engine cooling systems")
    top_k: int = Field(default=5, ge=1, le=20, 
                       description="Number of chunks to retrieve")
    include_sources: bool = Field(default=True, 
                                  description="Include source references in response")


class QueryResponse(BaseModel):
    """Response for /query endpoint"""
    question: str
    answer: str
    sources: Optional[List[SourceReference]] = None
    context_used: List[str] = Field(default_factory=list)
    processing_time_ms: float
    model_used: str


class HealthResponse(BaseModel):
    """Response for /health endpoint"""
    status: str
    indexed_documents: int
    total_chunks: int
    chunks_by_type: Dict[str, int]
    vector_store_size_bytes: Optional[int] = None
    uptime_seconds: float
    model_status: Dict[str, str]
    last_ingestion: Optional[datetime] = None


class DocumentInfo(BaseModel):
    """Information about an indexed document"""
    filename: str
    chunk_count: int
    pages: int
    ingestion_time: datetime


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)