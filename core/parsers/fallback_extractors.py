"""
Fallback text extractors for complex PDF layouts
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


class PyPDF2TextExtractor:
    """
    Fallback text extractor using PyPDF2 for pages that SmolDocling 
    interprets as single complex images
    """
    
    def __init__(self):
        self.name = "PyPDF2TextExtractor"
        logger.info(f"Initialized {self.name}")
    
    def extract_page_text(self, pdf_path: Path, page_number: int) -> Dict[str, Any]:
        """
        Extract text from a specific page using PyPDF2
        
        Args:
            pdf_path: Path to PDF file
            page_number: 1-based page number (matches SmolDocling convention)
            
        Returns:
            Dict with extracted text and metadata
        """
        try:
            logger.info(f"🔄 {self.name} extracting text from page {page_number}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Validate page number
                if page_number > len(pdf_reader.pages) or page_number < 1:
                    raise ValueError(f"Invalid page number: {page_number}")
                
                # Get page (0-based index)
                page = pdf_reader.pages[page_number - 1]
                
                # Extract text
                text = page.extract_text()
                
                # Try to identify structure (basic heuristics)
                lines = text.split('\n')
                
                # Detect potential tables by looking for aligned columns
                potential_tables = self._detect_potential_tables(lines)
                
                # Extract lists/bullet points
                lists = self._extract_lists(lines)
                
                # Clean up text
                cleaned_text = self._clean_extracted_text(text)
                
                result = {
                    'text': cleaned_text,
                    'raw_text': text,
                    'lines': lines,
                    'potential_tables': potential_tables,
                    'lists': lists,
                    'metadata': {
                        'extractor': self.name,
                        'page_number': page_number,
                        'line_count': len(lines),
                        'char_count': len(cleaned_text)
                    }
                }
                
                logger.info(f"✅ Extracted {len(cleaned_text)} chars, "
                           f"{len(lines)} lines, "
                           f"{len(potential_tables)} potential tables")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ {self.name} failed on page {page_number}: {e}")
            return {
                'text': '',
                'error': str(e),
                'metadata': {
                    'extractor': self.name,
                    'page_number': page_number,
                    'error': True
                }
            }
    
    def _detect_potential_tables(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect potential tables based on text patterns"""
        tables = []
        current_table = []
        
        for i, line in enumerate(lines):
            # Simple heuristic: lines with multiple spaces/tabs might be table rows
            if line.count('  ') >= 2 or '\t' in line:
                current_table.append(line)
            else:
                if len(current_table) >= 2:  # At least 2 rows
                    tables.append({
                        'start_line': i - len(current_table),
                        'end_line': i - 1,
                        'rows': current_table,
                        'row_count': len(current_table)
                    })
                current_table = []
        
        # Don't forget last table
        if len(current_table) >= 2:
            tables.append({
                'start_line': len(lines) - len(current_table),
                'end_line': len(lines) - 1,
                'rows': current_table,
                'row_count': len(current_table)
            })
        
        return tables
    
    def _extract_lists(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract bullet points and numbered lists"""
        lists = []
        current_list = []
        list_type = None
        
        import re
        
        for line in lines:
            stripped = line.strip()
            
            # Check for bullet points
            if re.match(r'^[•·▸▹►\-\*]\s+', stripped):
                if list_type != 'bullet':
                    if current_list:
                        lists.append({
                            'type': list_type,
                            'items': current_list
                        })
                    current_list = []
                    list_type = 'bullet'
                current_list.append(stripped)
            
            # Check for numbered lists
            elif re.match(r'^\d+[\.\)]\s+', stripped):
                if list_type != 'numbered':
                    if current_list:
                        lists.append({
                            'type': list_type,
                            'items': current_list
                        })
                    current_list = []
                    list_type = 'numbered'
                current_list.append(stripped)
            
            # Check for lettered lists
            elif re.match(r'^[a-zA-Z][\.\)]\s+', stripped):
                if list_type != 'lettered':
                    if current_list:
                        lists.append({
                            'type': list_type,
                            'items': current_list
                        })
                    current_list = []
                    list_type = 'lettered'
                current_list.append(stripped)
            
            else:
                # End of list
                if current_list:
                    lists.append({
                        'type': list_type,
                        'items': current_list
                    })
                    current_list = []
                    list_type = None
        
        # Don't forget last list
        if current_list:
            lists.append({
                'type': list_type,
                'items': current_list
            })
        
        return lists
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean up extracted text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common extraction issues
        text = text.replace('  ', ' ')
        
        # Remove page numbers at start/end
        import re
        text = re.sub(r'^\d+\s*', '', text)
        text = re.sub(r'\s*\d+$', '', text)
        
        return text.strip()


class PDFPlumberExtractor:
    """
    Alternative fallback using pdfplumber - better for tables
    """
    
    def __init__(self):
        self.name = "PDFPlumberExtractor"
        logger.info(f"Initialized {self.name}")
    
    def extract_page_content(self, pdf_path: Path, page_number: int) -> Dict[str, Any]:
        """
        Extract text and tables using pdfplumber
        
        Args:
            pdf_path: Path to PDF file
            page_number: 1-based page number
            
        Returns:
            Dict with extracted content
        """
        try:
            import pdfplumber
            
            logger.info(f"🔄 {self.name} extracting from page {page_number}")
            
            with pdfplumber.open(pdf_path) as pdf:
                # Get page (0-based index)
                page = pdf.pages[page_number - 1]
                
                # Extract text
                text = page.extract_text() or ""
                
                # Extract tables with better structure
                tables = page.extract_tables() or []
                
                # Extract text with layout preservation
                layout_text = page.extract_text_simple() or ""
                
                # Format tables with proper structure
                formatted_tables = []
                for i, table in enumerate(tables):
                    if not table:
                        continue
                        
                    # Try to identify headers (usually first row)
                    headers = []
                    data_rows = []
                    
                    if len(table) > 0:
                        # First row as headers
                        headers = [self._clean_cell(cell) for cell in table[0]]
                        # Rest as data
                        data_rows = table[1:] if len(table) > 1 else []
                    
                    # Clean data rows
                    cleaned_rows = []
                    for row in data_rows:
                        cleaned_row = [self._clean_cell(cell) for cell in row]
                        # Skip empty rows
                        if any(cell for cell in cleaned_row):
                            cleaned_rows.append(cleaned_row)
                    
                    formatted_table = {
                        'table_id': i + 1,
                        'headers': headers,
                        'rows': cleaned_rows,
                        'row_count': len(cleaned_rows),
                        'col_count': len(headers),
                        'raw_data': table  # Keep original for debugging
                    }
                    
                    # Try to detect table type
                    table_type = self._detect_table_type(headers, cleaned_rows)
                    if table_type:
                        formatted_table['table_type'] = table_type
                    
                    formatted_tables.append(formatted_table)
                
                result = {
                    'text': text,
                    'layout_text': layout_text,
                    'tables': formatted_tables,
                    'metadata': {
                        'extractor': self.name,
                        'page_number': page_number,
                        'char_count': len(text),
                        'table_count': len(formatted_tables)
                    }
                }
                
                logger.info(f"✅ Extracted {len(text)} chars, "
                           f"{len(formatted_tables)} tables")
                
                # Log table details
                for table in formatted_tables:
                    logger.debug(f"   Table {table['table_id']}: "
                                f"{table['row_count']}x{table['col_count']}, "
                                f"type: {table.get('table_type', 'unknown')}")
                
                return result
                
        except ImportError:
            logger.warning(f"pdfplumber not installed, falling back to PyPDF2")
            return PyPDF2TextExtractor().extract_page_text(pdf_path, page_number)
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
    
    def _clean_cell(self, cell: Any) -> str:
        """Clean table cell content"""
        if cell is None:
            return ""
        # Convert to string and clean
        text = str(cell).strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text
    
    def _detect_table_type(self, headers: List[str], rows: List[List[str]]) -> Optional[str]:
        """Try to detect what kind of table this is"""
        # Join headers for pattern matching
        header_text = ' '.join(headers).lower()
        
        # Common table patterns
        if any(word in header_text for word in ['motor', 'modell', 'leistung', 'getriebe']):
            return 'vehicle_specifications'
        elif any(word in header_text for word in ['preis', 'kosten', 'euro', '€']):
            return 'pricing'
        elif any(word in header_text for word in ['datum', 'termin', 'zeit']):
            return 'timeline'
        elif any(word in header_text for word in ['feature', 'ausstattung', 'option']):
            return 'features'
        
        return None