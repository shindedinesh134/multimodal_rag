"""
Advanced table processing for better retrieval
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from loguru import logger


class TableProcessor:
    """Process tables for enhanced understanding and retrieval"""
    
    def __init__(self):
        self.processed_count = 0
    
    def process_tables(self, parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process tables to add additional context and searchable text.
        
        - Convert to structured format
        - Generate natural language description
        - Extract key values for better retrieval
        """
        tables = parsed_doc.get("tables", [])
        
        for table in tables:
            processed_table = self._process_single_table(table)
            table.update(processed_table)
            self.processed_count += 1
        
        logger.info(f"Processed {len(tables)} tables")
        return parsed_doc
    
    def _process_single_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single table and add enhanced fields"""
        content = table.get("content", [])
        
        if not content or len(content) == 0:
            return {"nl_description": "Empty table", "key_values": {}}
        
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(content[1:], columns=content[0] if content else None)
            
            # Generate natural language description
            nl_desc = self._table_to_natural_language(df, table.get("page", 0))
            
            # Extract key-value pairs for important columns
            key_values = self._extract_key_values(df)
            
            # Enhanced markdown with more context
            enhanced_markdown = self._enhance_markdown(table.get("markdown", ""), nl_desc)
            
            return {
                "nl_description": nl_desc,
                "key_values": key_values,
                "enhanced_markdown": enhanced_markdown,
                "dataframe": df.to_dict() if len(df) < 50 else None  # Limit size
            }
            
        except Exception as e:
            logger.warning(f"Table processing error: {e}")
            return {
                "nl_description": f"Table from page {table.get('page', 0)} with {len(content)} rows",
                "key_values": {}
            }
    
    def _table_to_natural_language(self, df: pd.DataFrame, page: int) -> str:
        """Convert table to natural language description"""
        if df.empty:
            return f"Empty table on page {page}"
        
        rows, cols = df.shape
        
        # Get column names
        col_names = list(df.columns) if df.columns is not None else [f"col_{i}" for i in range(cols)]
        
        description = f"Table on page {page} with {rows} rows and {cols} columns.\n"
        description += f"Columns: {', '.join(str(c) for c in col_names[:10])}"
        if cols > 10:
            description += f" and {cols - 10} more"
        description += ".\n\n"
        
        # Add sample rows (first 3)
        if rows > 0:
            description += "Sample data:\n"
            for idx in range(min(3, rows)):
                row_data = [str(df.iloc[idx][col]) for col in col_names[:5]]
                description += f"Row {idx+1}: {' | '.join(row_data)}\n"
        
        # Add numeric summaries if applicable
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            description += "\nNumeric summary:\n"
            for col in numeric_cols[:3]:
                description += f"- {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
        
        return description
    
    def _extract_key_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract important key-value pairs from table"""
        key_values = {}
        
        # Look for parameter-value type columns
        if len(df.columns) >= 2:
            first_col = str(df.columns[0])
            second_col = str(df.columns[1])
            
            # If first column looks like parameters/names, treat as keys
            for idx in range(min(10, len(df))):
                key = str(df.iloc[idx][first_col])
                val = str(df.iloc[idx][second_col])
                if key and val and len(key) < 50:
                    key_values[key] = val
        
        return key_values
    
    def _enhance_markdown(self, original_markdown: str, nl_description: str) -> str:
        """Add natural language description to markdown"""
        return f"{nl_description}\n\n```markdown\n{original_markdown}\n```"
    
    def get_searchable_text(self, table: Dict[str, Any]) -> str:
        """Get searchable text representation for embedding"""
        parts = []
        
        # Add natural language description
        if "nl_description" in table:
            parts.append(table["nl_description"])
        
        # Add key values
        if "key_values" in table and table["key_values"]:
            parts.append("Key parameters: " + ", ".join(
                f"{k}={v}" for k, v in list(table["key_values"].items())[:10]
            ))
        
        # Add raw markdown
        if "markdown" in table:
            parts.append(table["markdown"])
        
        return "\n\n".join(parts)