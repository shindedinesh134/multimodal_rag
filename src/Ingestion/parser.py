"""
PDF parsing for multimodal content extraction
"""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image
import io
import base64
from loguru import logger


class PDFParser:
    """Extract text, tables, and images from PDF documents"""
    
    def __init__(self, max_image_size: int = 1024):
        self.max_image_size = max_image_size
    
    def parse_document(self, pdf_path: str, filename: str) -> Dict[str, Any]:
        """
        Parse PDF and extract all multimodal content.
        
        Returns:
            Dict with keys: text, tables, images, metadata
        """
        logger.info(f"Parsing PDF: {filename}")
        
        result = {
            "filename": filename,
            "pages": [],
            "text": [],
            "tables": [],
            "images": [],
            "metadata": {}
        }
        
        # Extract using PyMuPDF for text and images
        doc = fitz.open(pdf_path)
        result["metadata"] = {
            "num_pages": len(doc),
            "title": doc.metadata.get("title", filename),
            "author": doc.metadata.get("author", ""),
            "creation_date": doc.metadata.get("creationDate", "")
        }
        
        # Process each page
        for page_num, page in enumerate(doc, start=1):
            page_data = {
                "page_num": page_num,
                "text": page.get_text(),
                "images": [],
                "tables": []
            }
            
            # Extract images from this page
            image_list = page.get_images()
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Resize if needed
                    if max(pil_image.size) > self.max_image_size:
                        ratio = self.max_image_size / max(pil_image.size)
                        new_size = (int(pil_image.size[0] * ratio), 
                                   int(pil_image.size[1] * ratio))
                        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    image_data = {
                        "page": page_num,
                        "image_index": img_idx,
                        "image": pil_image,
                        "format": image_ext,
                        "size": pil_image.size,
                        "image_bytes": image_bytes
                    }
                    result["images"].append(image_data)
                    page_data["images"].append(image_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image on page {page_num}: {e}")
            
            # Add text content
            if page_data["text"].strip():
                result["text"].append({
                    "page": page_num,
                    "content": page_data["text"],
                    "type": "text"
                })
            
            result["pages"].append(page_data)
        
        doc.close()
        
        # Extract tables using pdfplumber (better table extraction)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0 and len(table[0]) > 0:
                            # Convert to markdown table format for better LLM understanding
                            markdown_table = self._table_to_markdown(table)
                            result["tables"].append({
                                "page": page_num,
                                "table_index": table_idx,
                                "content": table,
                                "markdown": markdown_table,
                                "type": "table"
                            })
        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")
        
        logger.info(f"Extracted: {len(result['text'])} text blocks, "
                   f"{len(result['tables'])} tables, {len(result['images'])} images")
        
        return result
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """Convert extracted table to markdown format"""
        if not table or not table[0]:
            return ""
        
        # Clean table data
        cleaned_rows = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_rows.append(cleaned_row)
        
        # Build markdown
        markdown_lines = []
        
        # Header row
        header = cleaned_rows[0]
        markdown_lines.append("| " + " | ".join(header) + " |")
        markdown_lines.append("|" + "|".join([" --- " for _ in header]) + "|")
        
        # Data rows
        for row in cleaned_rows[1:]:
            # Ensure row has same length as header
            while len(row) < len(header):
                row.append("")
            markdown_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(markdown_lines)