"""
Table and text separation utilities for hybrid parsing
"""

import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ContentRegion:
    """Represents a region of content with its type"""
    content: str
    region_type: str  # 'text', 'table', 'header', 'footer'
    start_pos: int
    end_pos: int
    metadata: Optional[Dict[str, Any]] = None


class TableTextSeparator:
    """Separates table content from regular text in documents"""
    
    def __init__(self):
        # Patterns that indicate table-like content
        self.table_patterns = [
            # Multiple numbers/values in a row
            r'\d+[\s,./]\d+\s+\d+[\s,./]\d+',
            # Currency values
            r'\d+\.\d{3}[\s€$]',
            # Table headers with units
            r'\[[^\]]+\]\s+\[[^\]]+\]',
            # Multiple columns of data
            r'(\S+\s+){3,}\d+',
        ]
    
    def separate_content(self, raw_text: str, tables: List[Dict]) -> Dict[str, Any]:
        """
        Separate table content from regular text
        
        Args:
            raw_text: Full raw text from page
            tables: List of structured table data
            
        Returns:
            Dictionary with separated content
        """
        result = {
            'pure_text': '',
            'table_regions': [],
            'mixed_regions': [],
            'metadata': {}
        }
        
        # Find table boundaries in raw text
        table_boundaries = self._find_table_boundaries(raw_text, tables)
        
        # Extract pure text (non-table content)
        pure_text_parts = []
        last_end = 0
        
        for start, end in table_boundaries:
            # Add text before table
            if start > last_end:
                text_part = raw_text[last_end:start].strip()
                if text_part:
                    pure_text_parts.append(text_part)
            last_end = end
        
        # Add remaining text after last table
        if last_end < len(raw_text):
            text_part = raw_text[last_end:].strip()
            if text_part:
                pure_text_parts.append(text_part)
        
        result['pure_text'] = '\n\n'.join(pure_text_parts)
        result['table_regions'] = table_boundaries
        
        return result
    
    def separate_tables_from_text(self, content: str) -> Tuple[List[str], List[str]]:
        """
        Separate tables from text content
        
        Args:
            content: Mixed content with tables and text
            
        Returns:
            Tuple of (text_parts, table_parts)
        """
        text_parts = []
        table_parts = []
        
        # Split by table markers
        parts = re.split(r'\[TABLE\]|\[/TABLE\]', content)
        
        in_table = False
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if content.find(f'[TABLE]\n{part}\n[/TABLE]') != -1:
                # This is a table
                table_parts.append(part)
            else:
                # This is text
                text_parts.append(part)
        
        return text_parts, table_parts
    
    def _find_table_boundaries(self, raw_text: str, tables: List[Dict]) -> List[Tuple[int, int]]:
        """Find start and end positions of tables in raw text"""
        boundaries = []
        
        for table in tables:
            # Try to find table content in raw text
            # Support both 'data' and 'rows' formats
            table_data = table.get('data') or table.get('rows', [])
            
            if table_data and len(table_data) > 0:
                first_row = table_data[0]
                last_row = table_data[-1]
                
                # Find positions
                start_pos = self._find_row_position(raw_text, first_row)
                end_pos = self._find_row_position(raw_text, last_row, start_from=start_pos)
                
                if start_pos != -1 and end_pos != -1:
                    # Extend to end of line
                    end_of_line = raw_text.find('\n', end_pos)
                    if end_of_line != -1:
                        end_pos = end_of_line
                    
                    boundaries.append((start_pos, end_pos))
        
        # Sort by start position
        boundaries.sort(key=lambda x: x[0])
        
        # Merge overlapping regions
        merged = []
        for start, end in boundaries:
            if merged and start <= merged[-1][1]:
                # Overlapping, merge
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged
    
    def _find_row_position(self, text: str, row: List[str], start_from: int = 0) -> int:
        """Find position of a table row in text"""
        # Clean row values
        clean_values = [str(v).strip() for v in row if v and str(v).strip()]
        
        if not clean_values:
            return -1
        
        # Try to find at least 2 consecutive values from the row
        for i in range(len(clean_values) - 1):
            pattern = re.escape(clean_values[i]) + r'[\s\n]*' + re.escape(clean_values[i + 1])
            match = re.search(pattern, text[start_from:])
            if match:
                return start_from + match.start()
        
        return -1
    
    def create_separated_segments(self, page_content: Dict) -> List[Dict]:
        """
        Create separate segments for text and table content
        
        Args:
            page_content: Dictionary with 'text' and 'tables' keys
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        # Separate content
        separated = self.separate_content(
            page_content.get('text', ''),
            page_content.get('tables', [])
        )
        
        # Create text segment (if there's meaningful text)
        if separated['pure_text'] and len(separated['pure_text']) > 50:
            segments.append({
                'content': separated['pure_text'],
                'segment_type': 'text',
                'metadata': {
                    'content_type': 'pure_text',
                    'has_tables': False,
                    'original_length': len(page_content.get('text', '')),
                    'cleaned_length': len(separated['pure_text'])
                }
            })
        
        # Create table segments
        for i, table in enumerate(page_content.get('tables', [])):
            segments.append({
                'content': '',  # Empty content, data is in metadata
                'segment_type': 'table',
                'metadata': {
                    'content_type': 'structured_table',
                    'table_id': i + 1,
                    'table_data': table,
                    'row_count': table.get('row_count', len(table.get('data', []))),
                    'col_count': table.get('col_count', len(table.get('headers', [])))
                }
            })
        
        return segments


def clean_page_content(content: str) -> str:
    """
    Clean page content by removing common artifacts
    
    Args:
        content: Raw page content
        
    Returns:
        Cleaned content
    """
    if not content:
        return ""
    
    # Remove excessive whitespace
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Clean individual line
        line = ' '.join(line.split())
        if line:
            cleaned_lines.append(line)
    
    # Join with single newlines
    content = '\n'.join(cleaned_lines)
    
    # Fix common extraction issues
    replacements = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        '…': '...',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',
        '—': '--'
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    return content.strip()