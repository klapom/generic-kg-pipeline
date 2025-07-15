"""
Standard PDF parser with vLLM SmolDocling and multi-modal support
"""

import asyncio
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import pdfplumber

# Import moved to _initialize_client to avoid circular dependency
from core.parsers.interfaces import (
    BaseParser,
    Document,
    DocumentMetadata,
    DocumentType,
    ParseError,
    Segment,
    VisualElement,
    VisualElementType,
)

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """
    PDF parser with vLLM SmolDocling integration
    
    Features:
    - Advanced PDF parsing with SmolDocling
    - Table extraction and structure analysis
    - Image and diagram extraction
    - Multi-modal content with VLM descriptions
    - Page-aware segmentation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_vlm: bool = True):
        """Initialize PDF parser with SmolDocling client"""
        super().__init__(config, enable_vlm)
        self.vllm_client = None
        self.supported_types = {DocumentType.PDF}
        
        # PDF-specific configuration
        self.config.setdefault("extract_images", True)
        self.config.setdefault("extract_tables", True)
        self.config.setdefault("extract_formulas", True)
        self.config.setdefault("image_min_size", 50)  # Min pixels for image extraction
        self.config.setdefault("image_formats", ["PNG", "JPEG", "JPG"])
        
        logger.info(f"Initialized PDF parser with VLM: {enable_vlm}")
    
    async def parse(self, file_path: Path) -> Document:
        """
        Parse PDF document with SmolDocling
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document with text segments and visual elements
            
        Raises:
            ParseError: If parsing fails
        """
        self.validate_file(file_path)
        
        try:
            logger.info(f"🔍 Starting PDF parsing: {file_path.name}")
            
            # Extract basic metadata
            logger.info("📊 Extracting PDF metadata...")
            metadata = self._extract_pdf_metadata(file_path)
            logger.info(f"📄 PDF metadata: {metadata.page_count} pages, {metadata.file_size} bytes")
            
            # Parse with SmolDocling
            logger.info("🤖 Starting SmolDocling parsing (via API)...")
            parsed_result = await self._parse_with_smoldocling(file_path)
            logger.info(f"✅ SmolDocling parsing completed - {len(parsed_result.pages)} pages processed")
            
            # Log each page processing
            for i, page in enumerate(parsed_result.pages, 1):
                text_length = len(page.text) if page.text else 0
                tables_count = len(page.tables) if page.tables else 0
                images_count = len(page.images) if page.images else 0
                formulas_count = len(page.formulas) if page.formulas else 0
                logger.info(f"📑 Page {i}: {text_length} chars text, {tables_count} tables, {images_count} images, {formulas_count} formulas")
            
            # Convert SmolDocling result to Document
            logger.info("🔄 Converting SmolDocling result to Document...")
            document = await self._convert_smoldocling_to_document(parsed_result, file_path, metadata)
            
            logger.info(f"✅ PDF parsing completed: {len(document.segments)} segments, "
                       f"{len(document.visual_elements)} visual elements")
            
            # Analyze visual elements with VLM if enabled
            if self.enable_vlm and document.visual_elements:
                logger.info(f"🧠 Analyzing {len(document.visual_elements)} visual elements with Qwen2.5-VL...")
                document.visual_elements = await self.analyze_visual_elements(
                    document.visual_elements,
                    {"document_type": "pdf", "title": metadata.title}
                )
                logger.info("✅ Visual analysis completed")
            
            return document
            
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise ParseError(f"Failed to parse PDF: {e}") from e
    
    async def _parse_with_smoldocling(self, file_path: Path) -> Any:
        """Parse PDF using vLLM SmolDocling"""
        if self.vllm_client is None:
            # Lazy import to avoid circular dependency
            from core.clients.vllm_smoldocling import VLLMSmolDoclingClient
            self.vllm_client = VLLMSmolDoclingClient()
        
        async with self.vllm_client as client:
            result = await client.parse_pdf(file_path)
            return result
    
    def _extract_pdf_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from PDF file using pdfplumber"""
        metadata = self.extract_metadata(file_path)
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Page count
                metadata.page_count = len(pdf.pages)
                
                # Extract document metadata if available
                if hasattr(pdf, 'doc') and hasattr(pdf.doc, 'info') and pdf.doc.info:
                    info = pdf.doc.info[0] if isinstance(pdf.doc.info, list) else pdf.doc.info
                    
                    # Extract title
                    if 'Title' in info:
                        title = info['Title']
                        if isinstance(title, bytes):
                            title = title.decode('utf-8', errors='ignore')
                        if title:
                            metadata.title = title
                    
                    # Extract author
                    if 'Author' in info:
                        author = info['Author']
                        if isinstance(author, bytes):
                            author = author.decode('utf-8', errors='ignore')
                        if author:
                            metadata.author = author
                    
                    # Try to parse creation date
                    if 'CreationDate' in info:
                        try:
                            date_str = info['CreationDate']
                            if isinstance(date_str, bytes):
                                date_str = date_str.decode('utf-8')
                            
                            # PDF date format: D:YYYYMMDDHHmmSS
                            if date_str.startswith('D:'):
                                date_str = date_str[2:]
                            
                            # Basic parsing
                            from datetime import datetime
                            year = int(date_str[:4])
                            month = int(date_str[4:6])
                            day = int(date_str[6:8])
                            metadata.created_date = datetime(year, month, day)
                        except:
                            pass
                
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
        
        return metadata
    
    async def _convert_smoldocling_to_document(self, 
                                             smoldocling_result: Any,
                                             file_path: Path,
                                             metadata: DocumentMetadata) -> Document:
        """Convert SmolDocling result to our Document format"""
        segments = []
        visual_elements = []
        
        # Process each page
        for page_data in smoldocling_result.pages:
            page_num = page_data.page_number
            
            # Add text segment
            if page_data.text:
                segments.append(Segment(
                    content=page_data.text,
                    page_number=page_num,
                    segment_index=len(segments),
                    segment_type="text",
                    metadata={"confidence": page_data.confidence_score}
                ))
            
            # Add table segments
            for table in page_data.tables:
                table_text = self._format_table(table)
                segments.append(Segment(
                    content=table_text,
                    page_number=page_num,
                    segment_index=len(segments),
                    segment_type="table",
                    metadata={
                        "headers": table.headers,
                        "rows": table.rows,
                        "caption": table.caption
                    }
                ))
            
            # Process images
            for image in page_data.images:
                visual_elem = VisualElement(
                    element_type=VisualElementType.IMAGE,
                    source_format=DocumentType.PDF,
                    content_hash=VisualElement.create_hash(str(image).encode()),
                    page_or_slide=page_num,
                    vlm_description=image.description,
                    analysis_metadata={
                        "caption": image.caption,
                        "type": image.image_type
                    }
                )
                visual_elements.append(visual_elem)
            
            # Process formulas
            for formula in page_data.formulas:
                visual_elem = VisualElement(
                    element_type=VisualElementType.FORMULA,
                    source_format=DocumentType.PDF,
                    content_hash=VisualElement.create_hash(str(formula).encode()),
                    page_or_slide=page_num,
                    extracted_data={
                        "latex": formula.latex,
                        "mathml": formula.mathml
                    },
                    vlm_description=formula.description
                )
                visual_elements.append(visual_elem)
        
        # Create document
        return self.create_document(
            file_path=file_path,
            segments=segments,
            metadata=metadata,
            visual_elements=visual_elements
        )
    
    def _format_table(self, table: Any) -> str:
        """Format table data as text"""
        lines = []
        if hasattr(table, 'caption') and table.caption:
            lines.append(f"Table: {table.caption}")
            lines.append("")
        
        if hasattr(table, 'headers') and table.headers:
            lines.append(" | ".join(table.headers))
            lines.append("-" * (len(" | ".join(table.headers))))
        
        if hasattr(table, 'rows'):
            for row in table.rows:
                lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)