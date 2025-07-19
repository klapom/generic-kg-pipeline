"""
Content detection utilities for identifying tables, charts, and other structured content
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from core.parsers.interfaces.data_models import SegmentType, TextSubtype, TableSubtype, VisualSubtype


class ContentDetector:
    """Detect and classify content types in document segments"""
    
    # Table detection patterns
    TABLE_PATTERNS = [
        r'Table\s+\d+[:\.]',  # Table 1: or Table 1.
        r'Tabelle\s+\d+[:\.]',  # German
        r'\bTable\s+[IVX]+[:\.]',  # Roman numerals
        r'^\s*\|.*\|.*\|',  # Markdown-style tables
        r'.*\t.*\t.*',  # Tab-separated values
    ]
    
    # Chart/Figure detection patterns
    CHART_PATTERNS = [
        r'Chart\s+\d+[:\.]',
        r'Figure\s+\d+[:\.]', 
        r'Fig\.\s+\d+[:\.]',
        r'Diagram\s+\d+[:\.]',
        r'Graph\s+\d+[:\.]',
        r'Abbildung\s+\d+[:\.]',  # German
    ]
    
    @classmethod
    def detect_content_type(cls, content: str) -> Tuple[SegmentType, Optional[str]]:
        """
        Detect the content type and subtype from text content
        
        Returns:
            Tuple of (SegmentType, subtype)
        """
        # Check for table patterns
        for pattern in cls.TABLE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return SegmentType.TABLE, TableSubtype.DATA
        
        # Check for chart/figure patterns
        for pattern in cls.CHART_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return SegmentType.VISUAL, VisualSubtype.CHART
        
        # Check if content looks like structured table data
        if cls._is_structured_table(content):
            return SegmentType.TABLE, TableSubtype.DATA
        
        # Default to text paragraph
        return SegmentType.TEXT, TextSubtype.PARAGRAPH
    
    @classmethod
    def _is_structured_table(cls, content: str) -> bool:
        """Check if content appears to be structured table data"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check for consistent delimiters
        delimiters = ['|', '\t', ',']
        for delimiter in delimiters:
            if all(delimiter in line for line in lines[:3] if line.strip()):
                # Count delimiter occurrences
                counts = [line.count(delimiter) for line in lines if line.strip()]
                if counts and all(c == counts[0] for c in counts[:5]):  # Check first 5 rows
                    return True
        
        return False
    
    @classmethod
    def extract_table_structure(cls, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from table content
        
        Returns:
            Dictionary with table structure or None if not a table
        """
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Try different delimiters
        for delimiter in ['|', '\t', ',']:
            if delimiter in lines[0]:
                return cls._parse_delimited_table(lines, delimiter)
        
        # Try fixed-width detection
        if cls._looks_like_fixed_width(lines):
            return cls._parse_fixed_width_table(lines)
        
        return None
    
    @classmethod
    def _parse_delimited_table(cls, lines: List[str], delimiter: str) -> Dict[str, Any]:
        """Parse a delimited table into structured format"""
        rows = []
        headers = None
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Split and clean cells
            cells = [cell.strip() for cell in line.split(delimiter)]
            
            # Remove empty cells at edges (common with | delimiter)
            if delimiter == '|':
                cells = [c for c in cells if c]
            
            if cells:
                if i == 0 or headers is None:
                    headers = cells
                else:
                    rows.append(cells)
        
        # Create structured representation
        structured_data = {
            "type": "table",
            "headers": headers,
            "rows": rows,
            "num_columns": len(headers) if headers else 0,
            "num_rows": len(rows)
        }
        
        # Create triple-friendly representation
        triples = []
        if headers and rows:
            for row_idx, row in enumerate(rows):
                row_entity = f"row_{row_idx + 1}"
                for col_idx, (header, value) in enumerate(zip(headers, row)):
                    if col_idx < len(row):  # Ensure we don't exceed row length
                        triple = {
                            "subject": row_entity,
                            "predicate": header,
                            "object": value
                        }
                        triples.append(triple)
        
        structured_data["triples"] = triples
        
        return structured_data
    
    @classmethod
    def _looks_like_fixed_width(cls, lines: List[str]) -> bool:
        """Check if lines appear to be fixed-width formatted"""
        if len(lines) < 3:
            return False
        
        # Check if lines have similar lengths
        lengths = [len(line) for line in lines[:5] if line.strip()]
        if not lengths:
            return False
        
        avg_length = sum(lengths) / len(lengths)
        return all(abs(length - avg_length) < avg_length * 0.2 for length in lengths)
    
    @classmethod
    def _parse_fixed_width_table(cls, lines: List[str]) -> Dict[str, Any]:
        """Parse fixed-width table (simplified version)"""
        # This is a simplified implementation
        # In production, you'd want more sophisticated column detection
        
        rows = []
        headers = None
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Simple space-based splitting
            cells = line.split()
            
            if cells:
                if i == 0 or headers is None:
                    headers = cells
                else:
                    rows.append(cells)
        
        return {
            "type": "table",
            "format": "fixed_width",
            "headers": headers,
            "rows": rows,
            "num_columns": len(headers) if headers else 0,
            "num_rows": len(rows)
        }
    
    @classmethod
    def enhance_segment_with_structure(cls, segment: Any) -> Any:
        """
        Enhance a segment with detected content type and structure
        
        Args:
            segment: The segment to enhance
            
        Returns:
            Enhanced segment
        """
        # Detect content type
        segment_type, subtype = cls.detect_content_type(segment.content)
        
        # Update segment type if it's currently generic
        if segment.segment_type == SegmentType.TEXT and segment.segment_subtype in [None, "paragraph"]:
            segment.segment_type = segment_type
            segment.segment_subtype = subtype
        
        # Extract table structure if it's a table
        if segment_type == SegmentType.TABLE:
            table_structure = cls.extract_table_structure(segment.content)
            if table_structure:
                segment.metadata['table_structure'] = table_structure
                segment.metadata['has_structured_data'] = True
                
                # Add triple count for easy reference
                if 'triples' in table_structure:
                    segment.metadata['triple_count'] = len(table_structure['triples'])
        
        return segment