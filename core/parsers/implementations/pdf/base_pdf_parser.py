"""
Base PDF Parser - Primary document extraction using SmolDocling
Focuses on text and structure extraction without VLM analysis
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import hashlib

from core.parsers.interfaces.base_parser import BaseParser
from core.parsers.interfaces.data_models import (
    Document, DocumentMetadata, DocumentType, Segment, 
    SegmentType, TextSubtype, TableSubtype, VisualSubtype
)
from .pdf_preprocessor import PDFPreprocessor, PreprocessResult
# DoclingParser import removed - using alternative strategies
from core.parsers.strategies.table_text_separator import TableTextSeparator

# Conditional imports
try:
    from pdfplumber import PDF as PDFPlumberPDF
    from pdfplumber.page import Page as PDFPlumberPage
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logger.warning("PDFPlumber not available for fallback parsing")

logger = logging.getLogger(__name__)


class BasePDFParser(BaseParser):
    """
    Base PDF parser using SmolDocling for primary extraction
    - Uses preprocessed page images from PDFPreprocessor
    - Preserves SmolDocling's native tags (table, figure, etc.)
    - No ContentDetector or pattern matching
    - Falls back to PDFPlumber if needed
    """
    
    # SmolDocling tag to SegmentType mapping
    TAG_TO_SEGMENT_TYPE = {
        # Text types
        "text": (SegmentType.TEXT, TextSubtype.PARAGRAPH),
        "paragraph": (SegmentType.TEXT, TextSubtype.PARAGRAPH),
        "heading": (SegmentType.TEXT, TextSubtype.HEADING_1),
        "title": (SegmentType.TEXT, TextSubtype.TITLE),
        "subtitle": (SegmentType.TEXT, TextSubtype.SUBTITLE),
        "caption": (SegmentType.TEXT, TextSubtype.CAPTION),
        "list": (SegmentType.TEXT, TextSubtype.LIST),
        "footnote": (SegmentType.TEXT, TextSubtype.FOOTNOTE),
        "code": (SegmentType.TEXT, TextSubtype.CODE),
        
        # Table types
        "table": (SegmentType.TABLE, TableSubtype.DATA),
        
        # Visual types
        "figure": (SegmentType.VISUAL, VisualSubtype.IMAGE),
        "image": (SegmentType.VISUAL, VisualSubtype.IMAGE),
        "chart": (SegmentType.VISUAL, VisualSubtype.CHART),
        "diagram": (SegmentType.VISUAL, VisualSubtype.DIAGRAM),
        "formula": (SegmentType.VISUAL, VisualSubtype.FORMULA),
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize parser with configuration"""
        super().__init__(config)
        
        # Configuration
        self.use_docling = self.config.get("use_docling", True)
        self.fallback_to_pdfplumber = self.config.get("fallback_to_pdfplumber", True)
        self.preserve_native_tags = self.config.get("preserve_native_tags", True)
        self.table_separator_strategy = self.config.get("table_separator_strategy", True)
        
        # Initialize preprocessor (for getting cached images)
        cache_dir = self.config.get("cache_dir", Path("cache/images"))
        self.preprocessor = PDFPreprocessor(cache_dir=cache_dir, config=config)
        
        # Initialize strategies
        # Use SmolDocling via VLLMSmolDoclingFinalClient
        if self.use_docling:
            try:
                from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient
                self.smoldocling_client = VLLMSmolDoclingFinalClient(
                    max_pages=self.config.get('max_pages', 20),
                    gpu_memory_utilization=self.config.get('gpu_memory_utilization', 0.3),
                    environment=self.config.get('environment', 'production')
                )
                logger.info("✅ SmolDocling client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize SmolDocling: {e}")
                self.smoldocling_client = None
        else:
            self.smoldocling_client = None
        
        if self.table_separator_strategy:
            self.table_separator = TableTextSeparator()
    
    def parse(self, file_path: Path) -> Document:
        """
        Parse PDF document
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Parsed Document with segments
        """
        logger.info(f"📄 Parsing PDF with BasePDFParser: {file_path}")
        
        # Get preprocessing results (cached page images)
        preprocess_result = self.preprocessor.preprocess(file_path)
        
        # Extract metadata
        metadata = self._extract_metadata(file_path)
        
        # Parse segments
        segments = []
        
        if self.use_docling and self.smoldocling_client:
            try:
                segments = self._parse_with_smoldocling(file_path, preprocess_result)
            except Exception as e:
                logger.error(f"❌ SmolDocling parsing failed: {e}")
                if self.fallback_to_pdfplumber and HAS_PDFPLUMBER:
                    logger.info("🔄 Falling back to PDFPlumber")
                    segments = self._parse_with_pdfplumber(file_path)
        elif self.fallback_to_pdfplumber and HAS_PDFPLUMBER:
            segments = self._parse_with_pdfplumber(file_path)
        else:
            logger.error("❌ No parsing strategy available")
            
        # Apply table separator strategy if enabled
        # Note: process_segments method not available in current TableTextSeparator
        # if segments and self.table_separator_strategy and self.table_separator:
        #     segments = self.table_separator.process_segments(segments, metadata)
            
        # Create document
        document = Document(
            document_id=hashlib.md5(str(file_path).encode()).hexdigest(),
            source_path=str(file_path),
            document_type=DocumentType.PDF,
            content="",  # Will be populated by segment content
            metadata=metadata,
            segments=segments,
            visual_elements=[]  # Visual elements will be added by ImageAnalyzer
        )
        
        logger.info(f"✅ Parsed {len(segments)} segments from PDF")
        return document
    
    def _parse_with_smoldocling(self, file_path: Path, preprocess_result: PreprocessResult) -> List[Segment]:
        """Parse using SmolDocling"""
        logger.info("🔧 Using SmolDocling for parsing")
        
        # Parse with SmolDocling
        smoldocling_result = self.smoldocling_client.parse_pdf(file_path)
        
        if not smoldocling_result or not smoldocling_result.success:
            logger.warning("⚠️ SmolDocling returned no results")
            return []
        
        # Convert SmolDocling results to segments
        segments = []
        
        for page_data in smoldocling_result.pages:
            # Create segments from page text
            if page_data.text:
                segment = Segment(
                    content=page_data.text,
                    page_number=page_data.page_number - 1,  # 0-based
                    segment_index=len(segments),
                    segment_type=SegmentType.TEXT,
                    segment_subtype=TextSubtype.PARAGRAPH.value,
                    metadata={
                        "source": "smoldocling",
                        "confidence": page_data.confidence_score
                    }
                )
                segments.append(segment)
            
            # Create segments from tables
            for table in page_data.tables:
                segment = self._create_table_segment(
                    table, page_data.page_number - 1, len(segments)
                )
                if segment:
                    segments.append(segment)
            
            # Note: Images are handled separately by ImageAnalyzer
            # We just mark their presence here
            if page_data.images:
                for img in page_data.images:
                    # Create placeholder segment for image reference
                    segment = Segment(
                        content=f"[Image: {img.get('type', 'image')}]",
                        page_number=page_data.page_number - 1,
                        segment_index=len(segments),
                        segment_type=SegmentType.VISUAL,
                        segment_subtype=VisualSubtype.IMAGE.value,
                        metadata={
                            "source": "smoldocling",
                            "bbox": img.get('bbox'),
                            "image_ref": img.get('ref')
                        }
                    )
                    segments.append(segment)
        
        logger.info(f"✅ SmolDocling extracted {len(segments)} segments")
        return segments
    
    def _convert_docling_element_to_segment(
        self, 
        element: Dict[str, Any], 
        page_num: int, 
        segment_index: int
    ) -> Optional[Segment]:
        """Convert Docling element to Segment"""
        content = element.get("text", "").strip()
        if not content:
            return None
        
        # Get element type/tag
        element_tag = element.get("type", "text").lower()
        
        # Map to segment type/subtype
        segment_type, segment_subtype = self.TAG_TO_SEGMENT_TYPE.get(
            element_tag, 
            (SegmentType.TEXT, TextSubtype.PARAGRAPH)
        )
        
        # Create metadata
        metadata = {
            "docling_tag": element_tag,  # Preserve original tag
            "bbox": element.get("bbox"),
            "confidence": element.get("confidence", 1.0),
        }
        
        # Add table-specific metadata
        if segment_type == SegmentType.TABLE:
            if "cells" in element:
                metadata["cells"] = element["cells"]
            if "headers" in element:
                metadata["headers"] = element["headers"]
                
        # Add visual references if this is a figure/image reference
        visual_refs = []
        if segment_type == SegmentType.VISUAL and "image_ref" in element:
            visual_refs.append(element["image_ref"])
        
        return Segment(
            content=content,
            page_number=page_num,
            segment_index=segment_index,
            segment_type=segment_type,
            segment_subtype=segment_subtype.value if segment_subtype else None,
            metadata=metadata,
            visual_references=visual_refs
        )
    
    def _parse_with_pdfplumber(self, file_path: Path) -> List[Segment]:
        """Fallback parsing with PDFPlumber"""
        logger.info("🔧 Using PDFPlumber for parsing")
        
        segments = []
        
        try:
            with PDFPlumberPDF.open(file_path) as pdf:
                pages_to_process = min(len(pdf.pages), self.config.get("max_pages", 20))
                
                for page_num in range(pages_to_process):
                    page = pdf.pages[page_num]
                    
                    # Extract text
                    text = page.extract_text()
                    if text:
                        # Simple paragraph splitting
                        paragraphs = text.split('\n\n')
                        for para in paragraphs:
                            para = para.strip()
                            if para:
                                segments.append(Segment(
                                    content=para,
                                    page_number=page_num,
                                    segment_index=len(segments),
                                    segment_type=SegmentType.TEXT,
                                    segment_subtype=TextSubtype.PARAGRAPH.value,
                                    metadata={"source": "pdfplumber"}
                                ))
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # Convert table to text
                            table_text = self._table_to_text(table)
                            if table_text:
                                segments.append(Segment(
                                    content=table_text,
                                    page_number=page_num,
                                    segment_index=len(segments),
                                    segment_type=SegmentType.TABLE,
                                    segment_subtype=TableSubtype.DATA.value,
                                    metadata={
                                        "source": "pdfplumber",
                                        "raw_table": table
                                    }
                                ))
                                
        except Exception as e:
            logger.error(f"❌ PDFPlumber parsing failed: {e}")
            
        return segments
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to text format"""
        lines = []
        for row in table:
            # Clean None values
            row = [cell if cell else "" for cell in row]
            lines.append(" | ".join(row))
        return "\n".join(lines)
    
    def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract document metadata"""
        metadata = DocumentMetadata(
            file_path=file_path,
            document_type=DocumentType.PDF,
            title=file_path.stem,
        )
        
        try:
            import fitz
            doc = fitz.open(str(file_path))
            metadata.page_count = len(doc)
            
            # Extract PDF metadata
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata.title = pdf_metadata.get("title", file_path.stem)
                metadata.author = pdf_metadata.get("author")
                # Add more metadata fields as needed
                
            doc.close()
        except Exception as e:
            logger.warning(f"⚠️ Could not extract full metadata: {e}")
            
        return metadata
    
    def _create_table_segment(self, table: Dict[str, Any], page_num: int, segment_index: int) -> Optional[Segment]:
        """Create a segment from SmolDocling table data"""
        try:
            # Extract table content
            table_text = table.get('text', '')
            if not table_text and 'rows' in table:
                # Build table text from rows
                rows = table['rows']
                table_lines = []
                for row in rows:
                    if isinstance(row, list):
                        table_lines.append(' | '.join(str(cell) for cell in row))
                    elif isinstance(row, str):
                        table_lines.append(row)
                table_text = '\n'.join(table_lines)
            
            if not table_text:
                return None
            
            return Segment(
                content=table_text,
                page_number=page_num,
                segment_index=segment_index,
                segment_type=SegmentType.TABLE,
                segment_subtype=TableSubtype.DATA.value,
                metadata={
                    "source": "smoldocling",
                    "bbox": table.get('bbox'),
                    "headers": table.get('headers', []),
                    "rows": table.get('rows', [])
                }
            )
        except Exception as e:
            logger.warning(f"Failed to create table segment: {e}")
            return None
    
    def supports(self, file_path: Path) -> bool:
        """Check if parser supports the file type"""
        return file_path.suffix.lower() == '.pdf'