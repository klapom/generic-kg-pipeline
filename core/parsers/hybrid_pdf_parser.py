"""
Hybrid PDF Parser that combines SmolDocling with fallback extractors
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from plugins.parsers.base_parser import BaseParser, Document, Segment, ParseError, DocumentType
from plugins.parsers.pdf_parser import PDFParser
from core.parsers.fallback_extractors import PyPDF2TextExtractor, PDFPlumberExtractor
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
from core.parsers.table_text_separator import TableTextSeparator, clean_page_content
from core.parsers.advanced_pdf_extractor import AdvancedPDFExtractor

logger = logging.getLogger(__name__)


class HybridPDFParser(BaseParser):
    """
    Hybrid parser that uses SmolDocling as primary parser
    and falls back to PyPDF2/pdfplumber for complex layouts
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_types = {DocumentType.PDF}
        
        # Initialize parsers
        self.smoldocling_client = VLLMSmolDoclingClient(
            max_pages=config.get('max_pages', 50) if config else 50,
            gpu_memory_utilization=config.get('gpu_memory_utilization', 0.2) if config else 0.2
        )
        
        # Fallback extractors
        self.pypdf2_extractor = PyPDF2TextExtractor()
        self.pdfplumber_extractor = PDFPlumberExtractor()
        
        # Table/Text separator
        self.separator = TableTextSeparator()
        
        # Advanced extractor for bbox filtering
        self.advanced_extractor = None  # Will be initialized if needed
        
        # Config
        self.use_pdfplumber = config.get('prefer_pdfplumber', False) if config else False
        self.fallback_threshold = config.get('fallback_confidence_threshold', 0.8) if config else 0.8
        self.separate_tables = config.get('separate_tables', True) if config else True
        
        # pdfplumber mode: 0=never, 1=fallback only, 2=always parallel
        self.pdfplumber_mode = config.get('pdfplumber_mode', 1) if config else 1
        
        # Layout settings for pdfplumber
        self.layout_settings = config.get('layout_settings', {
            'use_layout': True,
            'table_x_tolerance': 3,
            'table_y_tolerance': 3,
            'text_x_tolerance': 5,
            'text_y_tolerance': 5
        }) if config else {
            'use_layout': True,
            'table_x_tolerance': 3,
            'table_y_tolerance': 3,
            'text_x_tolerance': 5,
            'text_y_tolerance': 5
        }
        
        logger.info(f"Initialized HybridPDFParser with pdfplumber_mode={self.pdfplumber_mode} "
                   f"(0=never, 1=fallback, 2=parallel)")
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file"""
        if not file_path.exists():
            return False
        return file_path.suffix.lower() == '.pdf'
    
    async def aparse(self, file_path: Path) -> Document:
        """Async version - just calls sync version for now"""
        return self.parse(file_path)
    
    def parse(self, file_path: Path) -> Document:
        """
        Parse PDF using hybrid approach
        
        1. First pass with SmolDocling
        2. Identify pages needing fallback
        3. Re-process those pages with fallback extractor
        4. Merge results
        """
        try:
            if not file_path or not file_path.exists():
                raise ParseError(f"Invalid document path: {file_path}")
            
            pdf_path = Path(file_path)
            segments = []
            
            logger.info(f"🚀 Starting hybrid PDF parsing: {pdf_path.name}")
            start_time = datetime.now()
            
            # Step 1: Parse with SmolDocling
            logger.info("📄 Phase 1: SmolDocling parsing...")
            smol_result = self.smoldocling_client.parse_pdf(pdf_path)
            
            if not smol_result.success:
                logger.error(f"SmolDocling failed: {smol_result.error_message}")
                # Fall back to PyPDF2 for entire document
                return self._fallback_entire_document(pdf_path)
            
            # Step 2: Process each page
            complex_pages = []
            page_segments = {}
            
            for page in smol_result.pages:
                # Check if page needs fallback
                detection = page.layout_info.get('complex_layout_detection', {})
                is_complex = detection.get('is_complex_layout', False)
                confidence = detection.get('confidence', 0)
                
                if is_complex and confidence >= self.fallback_threshold:
                    complex_pages.append(page.page_number)
                    logger.warning(f"📑 Page {page.page_number} marked for fallback processing")
                    
                    # Create placeholder segment
                    segment = self._create_segment_from_page(
                        page, 
                        pdf_path,
                        needs_fallback=True
                    )
                else:
                    # Use SmolDocling result
                    segment = self._create_segment_from_page(
                        page,
                        pdf_path,
                        needs_fallback=False
                    )
                
                page_segments[page.page_number] = segment
            
            # Step 3: Process complex pages with fallback
            if complex_pages:
                logger.info(f"📄 Phase 2: Fallback processing for {len(complex_pages)} pages...")
                
                for page_num in complex_pages:
                    logger.info(f"🔄 Processing page {page_num} with fallback extractor...")
                    
                    # Always try pdfplumber first for better table extraction
                    try:
                        fallback_result = self.pdfplumber_extractor.extract_page_content(
                            pdf_path, page_num
                        )
                        extractor_used = 'pdfplumber'
                    except Exception as e:
                        logger.warning(f"PDFPlumber failed: {e}, falling back to PyPDF2")
                        fallback_result = self.pypdf2_extractor.extract_page_text(
                            pdf_path, page_num
                        )
                        extractor_used = 'pypdf2'
                    
                    # Update segment with fallback content
                    segment = page_segments[page_num]
                    
                    # Separate tables from text if enabled
                    if self.separate_tables and fallback_result.get('tables'):
                        separated = self.separator.separate_content(
                            fallback_result.get('text', ''),
                            fallback_result.get('tables', [])
                        )
                        segment.content = separated['pure_text']
                        segment.metadata['table_boundaries'] = separated['table_regions']
                        segment.metadata['original_text_length'] = len(fallback_result.get('text', ''))
                        segment.metadata['cleaned_text_length'] = len(separated['pure_text'])
                    else:
                        segment.content = fallback_result.get('text', '')
                    
                    # Add fallback metadata
                    segment.metadata.update({
                        'parser_used': 'fallback',
                        'fallback_extractor': extractor_used,
                        'original_parser': 'smoldocling',
                        'fallback_reason': 'complex_layout_detected',
                        'tables_extracted': len(fallback_result.get('tables', [])),
                        'lists_extracted': len(fallback_result.get('lists', []))
                    })
                    
                    # Add structured tables to metadata
                    if fallback_result.get('tables'):
                        # Process tables to ensure clean structure
                        processed_tables = []
                        for table in fallback_result['tables']:
                            # Skip the first row if it's the "Motorisierungen" title
                            if table['rows'] and table['rows'][0][0] == 'Modell':
                                # First data row contains actual headers
                                processed_table = {
                                    'table_id': table['table_id'],
                                    'table_type': table.get('table_type', 'unknown'),
                                    'headers': table['rows'][0],  # Real headers
                                    'data': table['rows'][1:],     # Actual data
                                    'row_count': len(table['rows']) - 1,
                                    'col_count': table['col_count']
                                }
                            else:
                                # Use as-is
                                processed_table = {
                                    'table_id': table['table_id'],
                                    'table_type': table.get('table_type', 'unknown'),
                                    'headers': table.get('headers', []),
                                    'data': table['rows'],
                                    'row_count': table['row_count'],
                                    'col_count': table['col_count']
                                }
                            processed_tables.append(processed_table)
                        
                        segment.metadata['extracted_tables'] = processed_tables
                        
                        # Log table info
                        for table in processed_tables:
                            logger.info(f"   📊 Table {table['table_id']}: "
                                       f"{table['table_type']}, "
                                       f"{table['row_count']}x{table['col_count']}")
                    
                    if fallback_result.get('lists'):
                        segment.metadata['extracted_lists'] = fallback_result['lists']
                    
                    logger.info(f"✅ Fallback extraction complete: "
                               f"{len(segment.content)} chars, "
                               f"{len(fallback_result.get('tables', []))} tables extracted")
            
            # Step 4: Compile final segments
            for page_num in sorted(page_segments.keys()):
                segments.append(page_segments[page_num])
            
            # Summary
            processing_time = (datetime.now() - start_time).total_seconds()
            normal_pages = len(page_segments) - len(complex_pages)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 HYBRID PARSING SUMMARY:")
            logger.info(f"{'='*80}")
            logger.info(f"Total pages: {len(page_segments)}")
            logger.info(f"SmolDocling pages: {normal_pages}")
            logger.info(f"Fallback pages: {len(complex_pages)} - {complex_pages}")
            logger.info(f"Processing time: {processing_time:.1f}s")
            logger.info(f"{'='*80}\n")
            
            # Create Document object
            from plugins.parsers.base_parser import DocumentMetadata
            
            metadata = DocumentMetadata(
                title=pdf_path.stem,
                file_size=pdf_path.stat().st_size,
                page_count=len(segments),
                created_date=datetime.now(),
                document_type=DocumentType.PDF
            )
            
            # Combine all segment content
            full_content = "\n\n".join(seg.content for seg in segments)
            
            document = Document(
                content=full_content,
                segments=segments,
                metadata=metadata,
                visual_elements=[]  # Could be populated from SmolDocling images
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Hybrid parsing failed: {e}")
            raise ParseError(f"Failed to parse PDF: {str(e)}")
    
    def _create_segment_from_page(self, page, pdf_path: Path, 
                                  needs_fallback: bool = False) -> Segment:
        """Create segment from SmolDocling page result"""
        # For pages needing fallback, create placeholder content
        content = page.text if not needs_fallback else f"[Page {page.page_number} - Complex layout detected, will be processed with fallback extractor]"
        
        segment = Segment(
            content=content,
            segment_type="pdf_page",
            metadata={
                "page_number": page.page_number,
                "source_file": str(pdf_path),
                "parser": "smoldocling",
                "needs_fallback": needs_fallback,
                "confidence_score": page.confidence_score,
                "has_tables": len(page.tables) > 0,
                "has_images": len(page.images) > 0,
                "table_count": len(page.tables),
                "image_count": len(page.images),
                "text_length": len(page.text)
            }
        )
        
        # Add detection info if available
        if 'complex_layout_detection' in page.layout_info:
            segment.metadata['complex_layout_detection'] = page.layout_info['complex_layout_detection']
        
        # Store images for potential VLM processing
        if page.images:
            segment.metadata['detected_images'] = page.images
        
        return segment
    
    def _fallback_entire_document(self, pdf_path: Path) -> Document:
        """Fallback to PyPDF2 for entire document"""
        logger.warning("Using fallback parser for entire document")
        segments = []
        
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for i in range(total_pages):
                    page = pdf_reader.pages[i]
                    text = page.extract_text()
                    
                    segment = Segment(
                        content=text,
                        segment_type="pdf_page",
                        metadata={
                            "page_number": i + 1,
                            "source_file": str(pdf_path),
                            "parser": "pypdf2_fallback",
                            "fallback_reason": "smoldocling_failed",
                            "text_length": len(text)
                        }
                    )
                    segments.append(segment)
            
            # Create Document
            from plugins.parsers.base_parser import DocumentMetadata
            from datetime import datetime
            
            metadata = DocumentMetadata(
                title=pdf_path.stem,
                file_size=pdf_path.stat().st_size,
                page_count=len(segments),
                created_date=datetime.now(),
                document_type=DocumentType.PDF
            )
            
            full_content = "\n\n".join(seg.content for seg in segments)
            
            document = Document(
                content=full_content,
                segments=segments,
                metadata=metadata,
                visual_elements=[]
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Fallback parsing also failed: {e}")
            raise ParseError(f"Both primary and fallback parsing failed: {str(e)}")