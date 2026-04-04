"""
Complete RAG chain with retrieval, context building, and generation
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from .vector_store import VectorStore
from ..models.embedding_model import EmbeddingModel
from ..models.llm import LLMWrapper


class RAGChain:
    """End-to-end RAG pipeline"""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "openai",
        llm_model: str = "gpt-3.5-turbo",
        vector_store_type: str = "faiss"
    ):
        self.embeddings = EmbeddingModel(embedding_model_name)
        self.llm = LLMWrapper(provider=llm_provider, model=llm_model)
        self.vector_store = VectorStore(embedding_dim=self.embeddings.get_dimension())
        
        logger.info("RAG chain initialized")
    
    def load_vector_store(self):
        """Load existing vector store (called on startup)"""
        # Vector store already loads on init
        size = self.vector_store.get_index_size()
        logger.info(f"Vector store loaded with {size} chunks")
        return size
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Generate embeddings and add chunks to vector store.
        
        Returns:
            List of chunk IDs
        """
        # Extract content for embedding
        contents = [self._get_searchable_text(chunk) for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(contents)
        
        # Add to vector store
        chunk_ids = self.vector_store.add_chunks(chunks, embeddings)
        
        return chunk_ids
    
    def _get_searchable_text(self, chunk: Dict[str, Any]) -> str:
        """Get searchable text representation for a chunk"""
        chunk_type = chunk.get("type", "text")
        content = chunk.get("content", "")
        
        # Add type prefix for better retrieval
        return f"[{chunk_type.upper()}] {content}"
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Returns:
            List of chunks with scores
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search
        results = self.vector_store.search(query_embedding, top_k)
        
        logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
        
        return results
    
    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        max_context_tokens: int = 3000
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved chunks as context.
        
        Returns:
            Dict with 'answer' and 'model'
        """
        if not chunks:
            return {
                "answer": "No relevant information found.",
                "model": self.llm.model_name
            }
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for idx, chunk in enumerate(chunks, 1):
            chunk_type = chunk.get("type", "text")
            page = chunk.get("page", "unknown")
            filename = chunk.get("filename", "unknown")
            content = chunk.get("content", "")
            
            context_parts.append(
                f"[Source {idx}] - Type: {chunk_type}, Document: {filename}, Page: {page}\n"
                f"Content: {content}\n"
            )
            sources.append(f"{filename} (page {page}, {chunk_type})")
        
        context = "\n---\n".join(context_parts)
        
        # Truncate context if needed
        if len(context) > max_context_tokens * 4:  # Rough char to token estimate
            context = context[:max_context_tokens * 4]
        
        # Build prompt
        prompt = self._build_prompt(question, context)
        
        # Generate answer
        answer = self.llm.generate(prompt)
        
        return {
            "answer": answer,
            "model": self.llm.model_name,
            "sources": sources
        }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the RAG prompt with domain-specific instructions"""
        
        system_prompt = """You are a technical assistant specialized in engine cooling systems and thermal management.
Your role is to answer questions based STRICTLY on the provided context from technical documentation.

IMPORTANT RULES:
1. ONLY use information from the provided context - DO NOT use external knowledge
2. If the context doesn't contain the answer, say "I cannot find this information in the provided documentation"
3. Cite which source document and page number you're referencing
4. For table data, present it clearly, possibly using markdown table format
5. For technical specifications (temperatures, pressures, flow rates), include units
6. Be concise but thorough

Context from engine cooling documentation:
{context}

Question: {question}

Answer (using only the context above):"""
        
        return system_prompt.format(context=context, question=question)