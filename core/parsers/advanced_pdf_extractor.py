"""
Advanced PDF extraction with bounding box filtering and layout preservation
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pdfplumber
from pdfplumber.utils import extract_text

logger = logging.getLogger(__name__)


class AdvancedPDFExtractor:
    """
    Advanced PDF extraction with:
    - Bounding box based table/text separation
    - Layout preservation
    - Configurable tolerances
    """
    
    def __init__(self, layout_settings: Dict[str, Any] = None):
        self.name = "AdvancedPDFExtractor"
        
        # Default layout settings
        self.layout_settings = layout_settings or {
            'use_layout': True,
            'table_x_tolerance': 3,
            'table_y_tolerance': 3,
            'text_x_tolerance': 5,
            'text_y_tolerance': 5
        }
        
        logger.info(f"Initialized {self.name} with layout settings: {self.layout_settings}")
    
    def extract_page_with_bbox(self, pdf_path: Path, page_number: int) -> Dict[str, Any]:
        """
        Extract page content with bounding box filtering
        
        Args:
            pdf_path: Path to PDF file
            page_number: 1-based page number
            
        Returns:
            Dict with tables and text separated by bounding boxes
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Get page (0-based index)
                page = pdf.pages[page_number - 1]
                
                logger.info(f"🔄 {self.name} extracting from page {page_number} with bbox filtering")
                
                # 1. Find all tables with their bounding boxes
                tables_with_bbox = self._extract_tables_with_bbox(page)
                
                # 2. Extract text outside table areas
                text_outside_tables = self._extract_text_outside_tables(page, tables_with_bbox)
                
                # 3. Extract full page text with layout (for comparison)
                full_text = page.extract_text() or ""
                
                # 4. Calculate coverage
                table_coverage = self._calculate_table_coverage(page, tables_with_bbox)
                
                result = {
                    'tables': tables_with_bbox,
                    'text': text_outside_tables,
                    'full_text': full_text,
                    'metadata': {
                        'extractor': self.name,
                        'page_number': page_number,
                        'method': 'bbox_filtered',
                        'table_coverage': table_coverage,
                        'tables_found': len(tables_with_bbox),
                        'text_length': len(text_outside_tables),
                        'full_text_length': len(full_text)
                    }
                }
                
                logger.info(f"✅ Extracted {len(tables_with_bbox)} tables, "
                           f"{len(text_outside_tables)} chars of filtered text "
                           f"(full: {len(full_text)} chars)")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ {self.name} failed on page {page_number}: {e}")
            return {
                'tables': [],
                'text': '',
                'full_text': '',
                'error': str(e),
                'metadata': {'extractor': self.name, 'page_number': page_number}
            }
    
    def _extract_tables_with_bbox(self, page) -> List[Dict[str, Any]]:
        """Extract tables with their bounding boxes and layout-preserved content"""
        tables_with_bbox = []
        
        # Find tables
        table_finder = page.find_tables(table_settings={
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "edge_min_length": 3
        })
        
        for i, table in enumerate(table_finder):
            # Get table data
            table_data = table.extract()
            
            if not table_data:
                continue
            
            # Get bounding box
            bbox = table.bbox  # (x0, y0, x1, y1)
            
            # Extract text with layout preservation for this table region
            table_chars = self._get_chars_in_bbox(page.chars, bbox)
            
            if self.layout_settings['use_layout'] and table_chars:
                # Extract with tight tolerances for table
                layout_text = extract_text(
                    table_chars,
                    layout=True,
                    x_tolerance=self.layout_settings['table_x_tolerance'],
                    y_tolerance=self.layout_settings['table_y_tolerance']
                )
            else:
                layout_text = None
            
            # Process table data
            headers = []
            data_rows = []
            
            if len(table_data) > 0:
                # Check if first row looks like headers
                first_row = table_data[0]
                if self._is_header_row(first_row):
                    headers = [self._clean_cell(cell) for cell in first_row]
                    data_rows = table_data[1:]
                else:
                    # No clear headers, use generic
                    headers = [f"Column_{j+1}" for j in range(len(first_row))]
                    data_rows = table_data
            
            # Clean data rows
            cleaned_rows = []
            for row in data_rows:
                cleaned_row = [self._clean_cell(cell) for cell in row]
                if any(cell for cell in cleaned_row):  # Skip empty rows
                    cleaned_rows.append(cleaned_row)
            
            table_info = {
                'table_id': i + 1,
                'bbox': bbox,
                'headers': headers,
                'data': cleaned_rows,
                'row_count': len(cleaned_rows),
                'col_count': len(headers),
                'layout_text': layout_text,
                'table_type': self._detect_table_type(headers, cleaned_rows)
            }
            
            tables_with_bbox.append(table_info)
            
            logger.debug(f"   Table {i+1}: bbox={bbox}, "
                        f"{table_info['row_count']}x{table_info['col_count']}")
        
        return tables_with_bbox
    
    def _extract_text_outside_tables(self, page, tables: List[Dict]) -> str:
        """Extract text that is outside of table bounding boxes"""
        # Get all characters
        all_chars = page.chars
        
        # Filter out characters inside table bboxes
        filtered_chars = []
        for char in all_chars:
            if not self._char_in_any_table(char, tables):
                filtered_chars.append(char)
        
        # Extract text with layout
        if self.layout_settings['use_layout'] and filtered_chars:
            text = extract_text(
                filtered_chars,
                layout=True,
                x_tolerance=self.layout_settings['text_x_tolerance'],
                y_tolerance=self.layout_settings['text_y_tolerance']
            )
        else:
            text = extract_text(filtered_chars) if filtered_chars else ""
        
        return text
    
    def _get_chars_in_bbox(self, chars: List[Dict], bbox: Tuple[float, float, float, float]) -> List[Dict]:
        """Get characters within a bounding box"""
        x0, y0, x1, y1 = bbox
        return [
            char for char in chars
            if x0 <= char.get('x0', 0) <= x1 and y0 <= char.get('top', 0) <= y1
        ]
    
    def _char_in_any_table(self, char: Dict, tables: List[Dict]) -> bool:
        """Check if a character is inside any table bounding box"""
        for table in tables:
            if self._char_in_bbox(char, table['bbox']):
                return True
        return False
    
    def _char_in_bbox(self, char: Dict, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if a character is inside a bounding box"""
        x0, y0, x1, y1 = bbox
        char_x = char.get('x0', 0)
        char_y = char.get('top', 0)
        
        # Add small margin to avoid edge cases
        margin = 1
        return (x0 - margin <= char_x <= x1 + margin and 
                y0 - margin <= char_y <= y1 + margin)
    
    def _calculate_table_coverage(self, page, tables: List[Dict]) -> float:
        """Calculate percentage of page covered by tables"""
        if not tables:
            return 0.0
        
        page_width = page.width
        page_height = page.height
        page_area = page_width * page_height
        
        # Calculate total table area (handling overlaps)
        table_area = 0
        for table in tables:
            x0, y0, x1, y1 = table['bbox']
            table_area += (x1 - x0) * (y1 - y0)
        
        coverage = (table_area / page_area) * 100 if page_area > 0 else 0
        return round(coverage, 2)
    
    def _is_header_row(self, row: List[Any]) -> bool:
        """Detect if a row is likely headers"""
        if not row:
            return False
        
        # Check if cells contain typical header keywords
        header_keywords = ['name', 'id', 'type', 'value', 'price', 'date', 
                          'model', 'motor', 'preis', 'leistung', 'emission']
        
        text = ' '.join(str(cell).lower() for cell in row if cell)
        return any(keyword in text for keyword in header_keywords)
    
    def _clean_cell(self, cell: Any) -> str:
        """Clean cell content"""
        if cell is None:
            return ""
        
        # Convert to string and clean
        text = str(cell).strip()
        
        # Replace multiple spaces/newlines
        text = ' '.join(text.split())
        
        return text
    
    def _detect_table_type(self, headers: List[str], rows: List[List[str]]) -> str:
        """Detect table type based on content"""
        # Join headers for analysis
        header_text = ' '.join(headers).lower()
        
        # Check for specific table types
        if any(word in header_text for word in ['motor', 'modell', 'leistung', 'preis']):
            return 'vehicle_specifications'
        elif any(word in header_text for word in ['länge', 'breite', 'höhe', 'gewicht']):
            return 'technical_dimensions'
        elif any(word in header_text for word in ['price', 'cost', 'preis', 'kosten']):
            return 'pricing'
        else:
            return 'generic'