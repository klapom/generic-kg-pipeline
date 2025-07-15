"""
Fallback text extractor using pdfplumber - optimized for tables
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pdfplumber
from pdfplumber.utils import extract_text

from core.parsers.interfaces import Segment

logger = logging.getLogger(__name__)


class PDFPlumberExtractor:
    """
    Alternative fallback using pdfplumber - better for tables
    """
    
    def __init__(self):
        self.name = "PDFPlumberExtractor"
        logger.info(f"Initialized {self.name}")
    
    def extract(self, 
                pdf_path: Path, 
                pages: Optional[List[int]] = None,
                layout_settings: Optional[Dict[str, Any]] = None,
                separate_tables: bool = True,
                use_bbox_filtering: bool = False) -> List[Segment]:
        """
        Extract text from PDF using pdfplumber
        
        Args:
            pdf_path: Path to PDF file
            pages: Optional list of page numbers to extract (1-based)
            layout_settings: Layout extraction settings
            separate_tables: Whether to create separate segments for tables
            use_bbox_filtering: Whether to use bounding box filtering for text/table separation
            
        Returns:
            List of Segments with extracted text
        """
        segments = []
        
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Determine which pages to process
                if pages:
                    pages_to_process = [p for p in pages if 1 <= p <= total_pages]
                else:
                    pages_to_process = list(range(1, total_pages + 1))
                
                for page_num in pages_to_process:
                    page_data = self.extract_page_content(
                        pdf_path, 
                        page_num, 
                        layout_settings,
                        separate_tables,
                        use_bbox_filtering
                    )
                    
                    # Add text segment
                    if page_data.get('text'):
                        segment = Segment(
                            content=page_data['text'],
                            page_number=page_num,
                            segment_type="text",
                            metadata={
                                'extractor': self.name,
                                'table_count': page_data['metadata']['table_count'],
                                'char_count': page_data['metadata']['char_count']
                            }
                        )
                        segments.append(segment)
                    
                    # Add table segments if separated
                    if separate_tables and page_data.get('tables'):
                        for i, table in enumerate(page_data['tables']):
                            table_segment = Segment(
                                content=table['formatted'],
                                page_number=page_num,
                                segment_type="table",
                                metadata={
                                    'extractor': self.name,
                                    'table_index': i,
                                    'headers': table.get('headers', []),
                                    'row_count': table.get('row_count', 0)
                                }
                            )
                            segments.append(table_segment)
                
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
        
        return segments
    
    def extract_page_content(self, 
                           pdf_path: Path, 
                           page_number: int,
                           layout_settings: Optional[Dict[str, Any]] = None,
                           separate_tables: bool = True,
                           use_bbox_filtering: bool = False) -> Dict[str, Any]:
        """
        Extract text and tables using pdfplumber
        
        Args:
            pdf_path: Path to PDF file
            page_number: 1-based page number
            layout_settings: Layout extraction settings
            separate_tables: Whether to separate tables from text
            
        Returns:
            Dict with extracted content
        """
        try:
            logger.info(f"🔄 {self.name} extracting from page {page_number}")
            
            with pdfplumber.open(pdf_path) as pdf:
                # Get page (0-based index)
                page = pdf.pages[page_number - 1]
                
                # Extract tables first to get bounding boxes if needed
                table_settings = self._get_table_settings(layout_settings)
                tables = page.extract_tables(table_settings) or []
                
                # Get table bounding boxes if using bbox filtering
                table_bboxes = []
                if use_bbox_filtering:
                    table_bboxes = self._get_table_bboxes(page, table_settings)
                
                # Extract text
                if use_bbox_filtering and table_bboxes:
                    # Extract text outside of table regions
                    text = self._extract_text_outside_tables(
                        page, table_bboxes, layout_settings
                    )
                elif layout_settings and layout_settings.get('use_layout'):
                    text = page.extract_text(
                        x_tolerance=layout_settings.get('text_x_tolerance', 3),
                        y_tolerance=layout_settings.get('text_y_tolerance', 3)
                    ) or ""
                else:
                    text = page.extract_text() or ""
                
                # Format tables (already extracted above)
                
                # Format tables with proper structure
                formatted_tables = []
                for i, table in enumerate(tables):
                    if not table:
                        continue
                    
                    formatted_table = self._format_table(table)
                    if formatted_table:
                        # Add bbox info if available
                        if i < len(table_bboxes):
                            formatted_table['bbox'] = table_bboxes[i]
                        formatted_tables.append(formatted_table)
                
                # Remove tables from text if separating
                if separate_tables and formatted_tables:
                    text = self._remove_tables_from_text(text, formatted_tables)
                
                # Clean up text
                cleaned_text = self._clean_text(text)
                
                result = {
                    'text': cleaned_text,
                    'tables': formatted_tables,
                    'metadata': {
                        'extractor': self.name,
                        'page_number': page_number,
                        'table_count': len(formatted_tables),
                        'char_count': len(cleaned_text),
                        'bbox_filtered': use_bbox_filtering
                    }
                }
                
                logger.info(f"✅ Extracted {len(cleaned_text)} chars, "
                           f"{len(formatted_tables)} tables")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ {self.name} failed on page {page_number}: {e}")
            return {
                'text': '',
                'tables': [],
                'error': str(e),
                'metadata': {
                    'extractor': self.name,
                    'page_number': page_number,
                    'error': True
                }
            }
    
    def _format_table(self, table: List[List[Any]]) -> Dict[str, Any]:
        """Format table with headers and data"""
        if not table or len(table) < 2:
            return None
        
        # Clean cells
        cleaned_table = []
        for row in table:
            cleaned_row = [self._clean_cell(cell) for cell in row]
            # Skip empty rows
            if any(cell for cell in cleaned_row):
                cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) < 2:
            return None
        
        # First row as headers
        headers = cleaned_table[0]
        data_rows = cleaned_table[1:]
        
        # Format as text
        lines = []
        
        # Headers
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-" * (len(" | ".join(headers))))
        
        # Data rows
        for row in data_rows:
            lines.append(" | ".join(row))
        
        return {
            'headers': headers,
            'data_rows': data_rows,
            'row_count': len(data_rows),
            'formatted': "\n".join(lines),
            'raw': table
        }
    
    def _clean_cell(self, cell: Any) -> str:
        """Clean table cell content"""
        if cell is None:
            return ""
        
        # Convert to string and clean
        text = str(cell).strip()
        
        # Replace newlines with spaces in cells
        text = " ".join(text.split())
        
        return text
    
    def _remove_tables_from_text(self, text: str, tables: List[Dict[str, Any]]) -> str:
        """Remove table content from main text"""
        # This is a simple approach - could be improved with better matching
        for table in tables:
            # Try to remove the formatted table from text
            if table.get('formatted'):
                text = text.replace(table['formatted'], '')
            
            # Also try to remove based on headers and first few rows
            if table.get('headers'):
                header_text = " ".join(table['headers'])
                if header_text in text:
                    # Find and remove table section
                    start = text.find(header_text)
                    if start != -1:
                        # Look for the end of the table
                        lines = text[start:].split('\n')
                        table_lines = min(len(table.get('data_rows', [])) + 2, len(lines))
                        table_text = '\n'.join(lines[:table_lines])
                        text = text.replace(table_text, '')
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Clean individual line
            line = ' '.join(line.split())
            if line:
                cleaned_lines.append(line)
        
        # Join with single newlines
        text = '\n'.join(cleaned_lines)
        
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
            text = text.replace(old, new)
        
        return text.strip()
    
    def _get_table_settings(self, layout_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get table extraction settings"""
        if layout_settings:
            return {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_x_tolerance": layout_settings.get('table_x_tolerance', 3),
                "intersection_y_tolerance": layout_settings.get('table_y_tolerance', 3)
            }
        return {}
    
    def _get_table_bboxes(self, page: Any, table_settings: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
        """Get bounding boxes for detected tables"""
        bboxes = []
        
        # Try to find table bounding boxes
        if hasattr(page, 'find_tables'):
            table_finder = page.find_tables(table_settings)
            
            for table in table_finder:
                if hasattr(table, 'bbox'):
                    bboxes.append(table.bbox)
        
        return bboxes
    
    def _extract_text_outside_tables(self, 
                                   page: Any, 
                                   table_bboxes: List[Tuple[float, float, float, float]],
                                   layout_settings: Optional[Dict[str, Any]] = None) -> str:
        """Extract text that is outside of table bounding boxes"""
        try:
            # Get all characters on the page
            chars = page.chars
            
            # Filter characters outside table regions
            filtered_chars = []
            for char in chars:
                char_in_table = False
                
                # Check if character is inside any table bbox
                for bbox in table_bboxes:
                    x0, y0, x1, y1 = bbox
                    if (x0 <= char['x0'] <= x1 and 
                        y0 <= char['y0'] <= y1):
                        char_in_table = True
                        break
                
                if not char_in_table:
                    filtered_chars.append(char)
            
            # Extract text from filtered characters
            if filtered_chars:
                x_tolerance = layout_settings.get('text_x_tolerance', 5) if layout_settings else 5
                y_tolerance = layout_settings.get('text_y_tolerance', 5) if layout_settings else 5
                
                text = extract_text(
                    filtered_chars,
                    x_tolerance=x_tolerance,
                    y_tolerance=y_tolerance
                )
                return text or ""
            
            return ""
            
        except Exception as e:
            logger.warning(f"Failed to filter text by bbox: {e}")
            # Fallback to regular extraction
            return page.extract_text() or ""