"""
Embedding model wrapper for generating document and query embeddings
"""

from typing import List, Union
import numpy as np
from loguru import logger

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")


class EmbeddingModel:
    """Wrapper for embedding models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.model = None
        else:
            logger.warning("No embedding model available - using mock embeddings")
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Default for MiniLM
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of documents"""
        if not texts:
            return []
        
        if self.model is not None:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return [emb for emb in embeddings]
        else:
            # Return random embeddings as fallback
            dim = self.get_dimension()
            return [np.random.randn(dim) for _ in texts]
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        if not query:
            return np.zeros(self.get_dimension())
        
        if self.model is not None:
            embedding = self.model.encode(query)
            return embedding
        else:
            return np.random.randn(self.get_dimension())