"""
Vector store abstraction for FAISS and ChromaDB
"""

import os
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using fallback storage.")


class VectorStore:
    """Multi-modal vector store supporting FAISS"""
    
    def __init__(self, persist_dir: str = "data/vector_store", embedding_dim: int = 384):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        
        self.index = None
        self.chunks = []  # Store chunk metadata
        self.embeddings_list = []  # Store embeddings
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        index_path = self.persist_dir / "faiss.index"
        chunks_path = self.persist_dir / "chunks.pkl"
        
        if FAISS_AVAILABLE and index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
                logger.info(f"Loaded existing index with {len(self.chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.index = None
        self.chunks = []
        self.embeddings_list = []
        logger.info("Created new vector store")
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[np.ndarray]) -> List[str]:
        """
        Add chunks with their embeddings to the index.
        
        Returns:
            List of chunk IDs
        """
        if not chunks or not embeddings:
            return []
        
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")
        
        chunk_ids = []
        start_idx = len(self.chunks)
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"chunk_{start_idx + idx}"
            chunk["chunk_id"] = chunk_id
            self.chunks.append(chunk)
            self.embeddings_list.append(embedding)
            chunk_ids.append(chunk_id)
        
        # Update FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            embeddings_array = np.vstack(embeddings).astype('float32')
            self.index.add(embeddings_array)
        
        # Persist
        self.persist()
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return chunk_ids
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks.
        
        Returns:
            List of chunks with similarity scores
        """
        if not self.chunks:
            return []
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    # Convert distance to similarity (L2 distance)
                    similarity = 1.0 / (1.0 + dist)
                    chunk["score"] = float(similarity)
                    chunk["index"] = int(idx)
                    results.append(chunk)
            
            return results
        else:
            # Fallback: linear search
            return self._linear_search(query_embedding, top_k)
    
    def _linear_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Linear search fallback when FAISS is unavailable"""
        if not self.embeddings_list:
            return []
        
        similarities = []
        for idx, emb in enumerate(self.embeddings_list):
            # Cosine similarity
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(emb)
            if norm_q > 0 and norm_e > 0:
                sim = np.dot(query_embedding.flatten(), emb.flatten()) / (norm_q * norm_e)
            else:
                sim = 0
            similarities.append((idx, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, sim in similarities[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["score"] = sim
            chunk["index"] = idx
            results.append(chunk)
        
        return results
    
    def get_index_size(self) -> int:
        """Get number of chunks in index"""
        return len(self.chunks)
    
    def get_chunk_type_counts(self) -> Dict[str, int]:
        """Get counts by chunk type"""
        counts = {"text": 0, "table": 0, "image": 0}
        for chunk in self.chunks:
            chunk_type = chunk.get("type", "text")
            if chunk_type in counts:
                counts[chunk_type] += 1
        return counts
    
    def get_document_stats(self) -> List[Dict]:
        """Get statistics per document"""
        doc_stats = {}
        for chunk in self.chunks:
            filename = chunk.get("filename", "unknown")
            if filename not in doc_stats:
                doc_stats[filename] = {"chunk_count": 0, "pages": set()}
            doc_stats[filename]["chunk_count"] += 1
            if "page" in chunk:
                doc_stats[filename]["pages"].add(chunk["page"])
        
        return [
            {
                "filename": name,
                "chunk_count": stats["chunk_count"],
                "pages": len(stats["pages"])
            }
            for name, stats in doc_stats.items()
        ]
    
    def delete_document(self, filename: str) -> int:
        """
        Delete all chunks belonging to a document.
        
        Returns:
            Number of chunks deleted
        """
        # Find indices to keep
        indices_to_keep = []
        chunks_to_keep = []
        embeddings_to_keep = []
        
        for idx, chunk in enumerate(self.chunks):
            if chunk.get("filename") != filename:
                indices_to_keep.append(idx)
                chunks_to_keep.append(chunk)
                if idx < len(self.embeddings_list):
                    embeddings_to_keep.append(self.embeddings_list[idx])
        
        deleted_count = len(self.chunks) - len(chunks_to_keep)
        
        # Update
        self.chunks = chunks_to_keep
        self.embeddings_list = embeddings_to_keep
        
        # Rebuild FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            if embeddings_to_keep:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                embeddings_array = np.vstack(embeddings_to_keep).astype('float32')
                self.index.add(embeddings_array)
            else:
                self._create_new_index()
        
        self.persist()
        logger.info(f"Deleted {deleted_count} chunks from {filename}")
        
        return deleted_count
    
    def persist(self):
        """Save index and metadata to disk"""
        if FAISS_AVAILABLE and self.index is not None:
            index_path = self.persist_dir / "faiss.index"
            faiss.write_index(self.index, str(index_path))
        
        chunks_path = self.persist_dir / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        
        if self.embeddings_list:
            embeddings_path = self.persist_dir / "embeddings.npy"
            np.save(embeddings_path, np.array(self.embeddings_list))
        
        logger.debug(f"Persisted vector store to {self.persist_dir}")