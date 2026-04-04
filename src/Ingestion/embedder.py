cat > src/ingestion/embedder.py << 'EOF'
"""
Embedder Module - Creates vector embeddings for text, tables, and image summaries.
"""

from typing import List, Dict, Any
from openai import OpenAI
import numpy as np
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoolingSystemEmbedder:
    """
    Embedder for cooling system documentation.
    Uses OpenAI's text-embedding-3-small model (1536 dimensions).
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536  # text-embedding-3-small dimension
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
        
        Returns:
            Numpy array of embeddings
        """
        if not text or text.strip() == "":
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.dimension)
        
        # Truncate if too long (model has 8191 token limit)
        if len(text) > 8000:
            text = text[:8000]
            logger.warning(f"Truncated text to 8000 chars")
        
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        
        embedding = np.array(response.data[0].embedding)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding arrays
        """
        embeddings = []
        for text in texts:
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                embeddings.append(np.zeros(self.dimension))
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Any]) -> List[Dict]:
        """
        Embed a list of DocumentChunk objects.
        
        Args:
            chunks: List of DocumentChunk objects
        
        Returns:
            List of chunks with embeddings added
        """
        texts_to_embed = []
        
        # For image chunks, use the summary if available
        for chunk in chunks:
            if chunk.chunk_type == 'image_summary' and hasattr(chunk, 'summary'):
                content = chunk.summary
            else:
                content = chunk.content
            
            # Add type prefix for better retrieval
            type_prefix = {
                'text': "TEXT: ",
                'table': "TABLE: ",
                'image_summary': "IMAGE: "
            }.get(chunk.chunk_type, "")
            
            texts_to_embed.append(type_prefix + content)
        
        embeddings = self.embed_batch(texts_to_embed)
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
EOF