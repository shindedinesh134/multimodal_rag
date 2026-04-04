"""
Complete ingestion pipeline orchestrating parsing, processing, and chunking
"""

from typing import Dict, Any, List
from pathlib import Path
from loguru import logger

from .parser import PDFParser
from .chunker import DocumentChunker
from .image_processor import ImageProcessor
from .table_processor import TableProcessor


class IngestionPipeline:
    """Orchestrates the complete document ingestion process"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vision_provider: str = "replicate",
        vision_model: str = "llava-13b"
    ):
        self.parser = PDFParser()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.image_processor = ImageProcessor(vision_provider, vision_model)
        self.table_processor = TableProcessor()
        
        logger.info("Ingestion pipeline initialized")
    
    def parse_document(self, pdf_path: str, filename: str) -> Dict[str, Any]:
        """Parse PDF and extract all content"""
        return self.parser.parse_document(pdf_path, filename)
    
    def process_images(self, parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Process images through VLM to generate summaries"""
        return self.image_processor.process_document_images(parsed_doc)
    
    def process_tables(self, parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Process tables for enhanced retrieval"""
        return self.table_processor.process_tables(parsed_doc)
    
    def create_chunks(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from processed content"""
        return self.chunker.create_chunks(parsed_doc)
    
    def run_full_pipeline(self, pdf_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Run complete ingestion pipeline.
        
        Returns:
            List of chunks ready for embedding and indexing
        """
        logger.info(f"Starting ingestion pipeline for: {filename}")
        
        # Step 1: Parse PDF
        parsed = self.parse_document(pdf_path, filename)
        
        # Step 2: Process images with VLM
        parsed = self.process_images(parsed)
        
        # Step 3: Process tables
        parsed = self.process_tables(parsed)
        
        # Step 4: Create chunks
        chunks = self.create_chunks(parsed)
        
        logger.info(f"Ingestion complete: {len(chunks)} chunks created")
        
        return chunks