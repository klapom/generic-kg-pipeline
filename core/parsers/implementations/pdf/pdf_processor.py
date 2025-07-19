"""
PDF Processor - Orchestrator for the new PDF processing architecture
Coordinates PDFPreprocessor, BasePDFParser, and ImageAnalyzer
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from core.parsers.interfaces.base_parser import BaseParser
from core.parsers.interfaces.data_models import Document, DocumentType, Segment
from .pdf_preprocessor import PDFPreprocessor
from .base_pdf_parser import BasePDFParser
from .image_analyzer import ImageAnalyzer, AnalysisMode
from core.parsers.utils.segment_context_enhancer import SegmentContextEnhancer

logger = logging.getLogger(__name__)


class PDFProcessor(BaseParser):
    """
    Main PDF processing orchestrator
    - Coordinates all components
    - Manages caching
    - Produces final Document
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processor
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Component configuration
        self.enable_preprocessing = self.config.get("enable_preprocessing", True)
        self.enable_image_analysis = self.config.get("enable_image_analysis", True)
        self.enable_page_analysis = self.config.get("enable_page_analysis", False)
        self.enable_context_enhancement = self.config.get("enable_context_enhancement", True)
        
        # Cache configuration
        cache_dir = self.config.get("cache_dir", Path("cache/images"))
        
        # Initialize components
        self.preprocessor = PDFPreprocessor(
            cache_dir=cache_dir,
            config=self.config.get("preprocessor_config", {})
        )
        
        self.base_parser = BasePDFParser(
            config=self.config.get("parser_config", {})
        )
        
        # Configure image analyzer
        image_analyzer_config = self.config.get("image_analyzer_config", {})
        if self.enable_image_analysis and self.enable_page_analysis:
            image_analyzer_config["analysis_mode"] = AnalysisMode.BOTH.value
        elif self.enable_image_analysis:
            image_analyzer_config["analysis_mode"] = AnalysisMode.EMBEDDED.value
        elif self.enable_page_analysis:
            image_analyzer_config["analysis_mode"] = AnalysisMode.PAGES.value
        
        self.image_analyzer = ImageAnalyzer(config=image_analyzer_config)
        
        logger.info(
            f"🚀 PDFProcessor initialized - "
            f"preprocessing: {self.enable_preprocessing}, "
            f"image_analysis: {self.enable_image_analysis}, "
            f"page_analysis: {self.enable_page_analysis}, "
            f"context_enhancement: {self.enable_context_enhancement}"
        )
    
    def parse(self, file_path: Path) -> Document:
        """
        Parse PDF document (synchronous wrapper)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Parsed Document
        """
        # Run async method in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.parse_async(file_path))
    
    async def parse_async(self, file_path: Path) -> Document:
        """
        Parse PDF document asynchronously
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Parsed Document with all analyses
        """
        logger.info(f"📚 Processing PDF: {file_path}")
        start_time = datetime.now()
        
        # Step 1: Preprocessing (extract and cache all images)
        if self.enable_preprocessing:
            logger.info("🔧 Step 1: Preprocessing PDF")
            preprocess_result = self.preprocessor.preprocess(file_path)
        else:
            # Minimal preprocessing for parser
            preprocess_result = self.preprocessor.preprocess(file_path)
        
        # Step 2: Base parsing (text and structure)
        logger.info("📄 Step 2: Extracting text and structure")
        document = self.base_parser.parse(file_path)
        
        # Step 2.5: Context enhancement (if enabled)
        if self.enable_context_enhancement and document.segments:
            logger.info("🔍 Step 2.5: Enhancing segments with context")
            try:
                SegmentContextEnhancer.enhance_segments(
                    document.segments, 
                    document.metadata
                )
                logger.debug(f"Context enhancement complete for {len(document.segments)} segments")
            except Exception as e:
                logger.warning(f"Context enhancement failed: {e}, continuing without context")
        
        # Step 3: Image analysis (if enabled)
        if self.enable_image_analysis or self.enable_page_analysis:
            logger.info("🔍 Step 3: Analyzing visual elements")
            analysis_result = await self.image_analyzer.analyze(
                file_path, 
                preprocess_result,
                document.segments  # Pass segments with context
            )
            
            # Add visual elements to document
            document.visual_elements.extend(analysis_result.embedded_images)
            
            # Integrate visual segments
            self._integrate_visual_segments(
                document, 
                analysis_result.visual_segments
            )
            
            # Add page context if available
            if self.enable_page_analysis and analysis_result.page_analyses:
                self._add_page_context(
                    document, 
                    analysis_result.page_analyses
                )
        
        # Step 4: Final processing
        logger.info("🏁 Step 4: Finalizing document")
        self._finalize_document(document)
        
        # Log processing stats
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✅ Processing complete in {processing_time:.2f}s - "
            f"{len(document.segments)} segments, "
            f"{len(document.visual_elements)} visual elements"
        )
        
        return document
    
    def _integrate_visual_segments(
        self, 
        document: Document, 
        visual_segments: List[Segment]
    ):
        """Integrate visual segments into document"""
        if not visual_segments:
            return
        
        logger.info(f"🔗 Integrating {len(visual_segments)} visual segments")
        
        # Group segments by page
        segments_by_page = {}
        for segment in document.segments:
            page = segment.page_number or 0
            if page not in segments_by_page:
                segments_by_page[page] = []
            segments_by_page[page].append(segment)
        
        # Add visual segments
        for visual_seg in visual_segments:
            page = visual_seg.page_number or 0
            if page not in segments_by_page:
                segments_by_page[page] = []
            segments_by_page[page].append(visual_seg)
        
        # Rebuild segments in order
        all_segments = []
        for page in sorted(segments_by_page.keys()):
            page_segments = segments_by_page[page]
            # Sort by type (text first, then visual) to maintain reading order
            page_segments.sort(key=lambda s: (
                0 if s.segment_type != "visual" else 1,
                s.segment_index
            ))
            all_segments.extend(page_segments)
        
        # Update segment indices
        for i, segment in enumerate(all_segments):
            segment.segment_index = i
        
        document.segments = all_segments
    
    def _add_page_context(
        self, 
        document: Document, 
        page_analyses: Dict[int, Any]
    ):
        """Add page-level context to segments"""
        logger.info(f"📑 Adding page context for {len(page_analyses)} pages")
        
        for segment in document.segments:
            if segment.page_number in page_analyses:
                page_context = self.image_analyzer.get_page_context(
                    segment.page_number, 
                    page_analyses
                )
                if page_context:
                    segment.metadata["page_context"] = page_context
    
    def _finalize_document(self, document: Document):
        """Final document processing and validation"""
        # Update document content from segments
        if document.segments:
            document.content = "\n\n".join(
                seg.content for seg in document.segments 
                if seg.content.strip()
            )
        
        # Add processing metadata
        document.metadata.custom_metadata["processor"] = "PDFProcessor"
        document.metadata.custom_metadata["components"] = {
            "preprocessing": self.enable_preprocessing,
            "image_analysis": self.enable_image_analysis,
            "page_analysis": self.enable_page_analysis
        }
        
        # Validate visual references
        visual_hashes = {ve.content_hash for ve in document.visual_elements}
        for segment in document.segments:
            # Ensure visual references are valid
            segment.visual_references = [
                ref for ref in segment.visual_references 
                if ref in visual_hashes
            ]
    
    def supports(self, file_path: Path) -> bool:
        """Check if processor supports the file type"""
        return file_path.suffix.lower() == '.pdf'
    
    @classmethod
    def create_for_image_extraction_only(cls, config: Optional[Dict[str, Any]] = None) -> 'PDFProcessor':
        """
        Factory method to create processor for image extraction only
        (Replacement for ImageExtractionPDFParser)
        """
        extraction_config = config or {}
        extraction_config.update({
            "enable_preprocessing": True,
            "enable_image_analysis": True,
            "enable_page_analysis": False,
            "enable_context_enhancement": True,  # Enable context for better VLM analysis
            "parser_config": {
                "use_docling": True,
                "preserve_native_tags": True,
                "table_separator_strategy": False  # No content detection
            }
        })
        return cls(extraction_config)