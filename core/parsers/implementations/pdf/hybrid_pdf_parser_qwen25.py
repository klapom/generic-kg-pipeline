"""
Enhanced Hybrid PDF Parser with Qwen2.5-VL Integration

This version integrates the new single-stage Qwen2.5-VL processor and
improved image extraction strategy.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from core.parsers.interfaces import (
    BaseParser, Document, Segment, ParseError, DocumentType,
    VisualElement, VisualElementType, SegmentType, TextSubtype,
    VisualSubtype, TableSubtype, DocumentMetadata
)
from core.parsers.implementations.pdf.extractors.pdfplumber_extractor import PDFPlumberExtractor
from core.parsers.strategies.table_text_separator import TableTextSeparator, clean_page_content
from core.vlm.qwen25_processor import Qwen25VLMProcessor, PageContext
from core.vlm.image_extraction import ImageExtractionStrategy
from core.parsers.utils.content_detector import ContentDetector

logger = logging.getLogger(__name__)


class HybridPDFParserQwen25(BaseParser):
    """
    Enhanced Hybrid PDF parser with Qwen2.5-VL integration
    
    Features:
    - Advanced PDF parsing with SmolDocling
    - Single-stage Qwen2.5-VL for visual analysis
    - Improved image extraction with fallback strategies
    - Page-level context analysis
    - Structured JSON parsing from VLM responses
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_vlm: bool = True):
        super().__init__(config, enable_vlm)
        self.supported_types = {DocumentType.PDF}
        
        # VLM Configuration
        self.enable_vlm = enable_vlm
        self.config.setdefault("extract_images", True)
        self.config.setdefault("extract_tables", True)
        self.config.setdefault("extract_formulas", True)
        self.config.setdefault("image_min_size", 50)
        
        # Initialize parsers
        environment = config.get('environment', 'production') if config else 'production'
        
        logger.info(f"Initializing SmolDocling client for {environment} environment")
        from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient
        self.smoldocling_client = VLLMSmolDoclingFinalClient(
            max_pages=config.get('max_pages', 50) if config else 50,
            gpu_memory_utilization=config.get('gpu_memory_utilization', 0.3) if config else 0.3,
            environment=environment
        )
        
        # Fallback extractor
        self.pdfplumber_extractor = PDFPlumberExtractor()
        
        # Table/Text separator
        self.separator = TableTextSeparator()
        
        # Initialize Qwen2.5-VL processor if VLM enabled
        if self.enable_vlm:
            vlm_config = config.get('vlm', {}) if config else {}
            self.vlm_processor = Qwen25VLMProcessor({
                'temperature': vlm_config.get('temperature', 0.2),
                'max_new_tokens': vlm_config.get('max_new_tokens', 512),
                'batch_size': vlm_config.get('batch_size', 4),
                'enable_page_context': vlm_config.get('enable_page_context', True),
                'enable_structured_parsing': vlm_config.get('enable_structured_parsing', True)
            })
        else:
            self.vlm_processor = None
        
        # Initialize image extraction strategy
        image_config = config.get('image_extraction', {}) if config else {}
        self.image_extractor = ImageExtractionStrategy({
            'min_size': image_config.get('min_size', 100),
            'extract_embedded': image_config.get('extract_embedded', True),
            'render_fallback': image_config.get('render_fallback', True),
            'page_render_dpi': image_config.get('page_render_dpi', 150),
            'render_visual_elements': image_config.get('render_visual_elements', True)
        })
        
        # Config
        self.use_pdfplumber = config.get('prefer_pdfplumber', False) if config else False
        self.fallback_threshold = config.get('fallback_confidence_threshold', 0.8) if config else 0.8
        self.separate_tables = config.get('separate_tables', True) if config else True
        
        # Content detection config (for table/chart detection)
        self.enable_content_detection = config.get('enable_content_detection', True) if config else True
        
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
        
        # Page context analysis settings
        self.enable_page_context = config.get('enable_page_context', True) if config else True
        self.page_context_pages = config.get('page_context_pages', 20) if config else 20  # Max pages for context
        
        logger.info(f"Initialized HybridPDFParserQwen25 with pdfplumber_mode={self.pdfplumber_mode} "
                   f"(0=never, 1=fallback, 2=parallel), VLM={enable_vlm}, "
                   f"page_context={self.enable_page_context}, "
                   f"content_detection={self.enable_content_detection}")
    
    async def parse(self, file_path: Path) -> Document:
        """Parse PDF using hybrid approach with enhanced VLM support"""
        # Get base document without VLM analysis
        document = self._parse_sync(file_path)
        
        # Process visual elements with Qwen2.5-VL if enabled
        if self.enable_vlm and self.vlm_processor and document.visual_elements:
            logger.info(f"🧠 Processing {len(document.visual_elements)} visual elements with Qwen2.5-VL...")
            
            # Ensure all visual elements have raw_data
            await self._ensure_visual_element_images(document, file_path)
            
            # Get page contexts if enabled
            page_contexts = None
            if self.enable_page_context:
                logger.info("📄 Analyzing page contexts...")
                page_contexts = await self._analyze_page_contexts(file_path, document)
            
            # Process visual elements
            analysis_results = await self.vlm_processor.process_visual_elements(
                document.visual_elements,
                page_contexts
            )
            
            # Update visual elements with results
            for ve, result in zip(document.visual_elements, analysis_results):
                if result.success:
                    ve.vlm_description = result.description
                    ve.confidence_score = result.confidence
                    ve.ocr_text = result.ocr_text
                    
                    if not ve.analysis_metadata:
                        ve.analysis_metadata = {}
                    
                    ve.analysis_metadata.update({
                        "vlm_model": "Qwen2.5-VL-7B",
                        "analysis_time": datetime.now().isoformat()
                    })
                    
                    # Add structured data if available
                    if result.structured_data:
                        ve.analysis_metadata['structured_data'] = result.structured_data
                        
                        # If it's a table with structured data, update extraction
                        if (ve.element_type == VisualElementType.TABLE and 
                            result.structured_data.get('type') == 'structured'):
                            ve.extracted_data = result.structured_data.get('data', {})
                
                # Enhance with page context if available
                if page_contexts and ve.page_or_slide in page_contexts:
                    self.vlm_processor.enhance_with_context(
                        ve, result, page_contexts[ve.page_or_slide]
                    )
            
            logger.info("✅ Qwen2.5-VL analysis completed")
            
            # Update visual segments with VLM descriptions
            logger.info("📝 Updating visual segments with VLM descriptions...")
            document.segments = await self._update_visual_segments_enhanced(
                document.segments,
                document.visual_elements
            )
            logger.info("✅ Visual segments updated")
        
        return document
    
    async def _ensure_visual_element_images(self, document: Document, pdf_path: Path) -> None:
        """Ensure all visual elements have raw_data populated"""
        for ve in document.visual_elements:
            if not ve.raw_data:
                logger.info(f"🖼️ Extracting image for visual element on page {ve.page_or_slide}")
                ve.raw_data = self.image_extractor.extract_image_bytes(
                    pdf_path,
                    ve.page_or_slide,
                    ve.bounding_box
                )
    
    async def _analyze_page_contexts(self, 
                                   pdf_path: Path, 
                                   document: Document) -> Dict[int, PageContext]:
        """Analyze page contexts for better visual element understanding"""
        page_contexts = {}
        
        # Limit to configured number of pages
        max_pages = min(
            document.metadata.page_count,
            self.page_context_pages
        )
        
        for page_num in range(max_pages):
            logger.info(f"📄 Analyzing context for page {page_num + 1}")
            
            # Render page as image
            page_images = self.image_extractor.extract_images(pdf_path, page_num)
            if page_images:
                # Use the rendered page image
                page_image = next(
                    (img for img in page_images if img.source == 'rendered'),
                    page_images[0]
                )
                
                # Analyze page context
                context = await self.vlm_processor.analyze_page_context(
                    page_image.data,
                    page_num + 1
                )
                
                page_contexts[page_num + 1] = context
        
        return page_contexts
    
    async def _update_visual_segments_enhanced(self,
                                             segments: List[Segment],
                                             visual_elements: List[VisualElement]
                                             ) -> List[Segment]:
        """Update visual segments with enhanced VLM descriptions and structured data"""
        # Create lookup dictionary
        ve_lookup = {ve.content_hash: ve for ve in visual_elements}
        
        for seg in segments:
            if seg.segment_type == SegmentType.VISUAL and seg.visual_references:
                # Find corresponding visual element
                for ref in seg.visual_references:
                    if ref in ve_lookup:
                        ve = ve_lookup[ref]
                        
                        # Update content with VLM description
                        if ve.vlm_description:
                            seg.content = f"[{ve.element_type.value}] {ve.vlm_description}"
                            seg.metadata['vlm_analyzed'] = True
                            
                            # Add structured data if available
                            if hasattr(ve, 'analysis_metadata') and ve.analysis_metadata.get('structured_data'):
                                seg.metadata['structured_data'] = ve.analysis_metadata['structured_data']
                                
                                # For tables, add extracted data
                                if ve.extracted_data:
                                    seg.metadata['table_data'] = ve.extracted_data
                        
                        break
        
        return segments
    
    def _parse_sync(self, file_path: Path) -> Document:
        """
        Parse PDF using hybrid approach (same as original)
        """
        try:
            if not file_path or not file_path.exists():
                raise ParseError(f"Invalid document path: {file_path}")
            
            pdf_path = Path(file_path)
            segments = []
            visual_elements = []
            
            logger.info(f"🚀 Starting enhanced hybrid PDF parsing: {pdf_path.name}")
            start_time = datetime.now()
            
            # Extract metadata
            metadata = self.extract_metadata(pdf_path)
            
            # Mode 2: Always use parallel extraction
            if self.pdfplumber_mode == 2:
                logger.info("📚 Mode 2: Using parallel extraction (SmolDocling + pdfplumber)")
                segments, visual_elements = self._parallel_extraction(pdf_path)
            
            # Mode 0 or 1: SmolDocling first
            else:
                # Primary extraction with SmolDocling
                logger.info("🤖 Primary extraction with SmolDocling...")
                try:
                    smoldocling_result = self.smoldocling_client.parse_pdf(pdf_path)
                    
                    # Analyze results and decide on fallback
                    pages_needing_fallback = []
                    
                    for idx, page_data in enumerate(smoldocling_result.pages):
                        # Ensure page_data has required attributes
                        if not hasattr(page_data, 'confidence_score'):
                            page_data.confidence_score = 0.0
                        if not hasattr(page_data, 'page_number'):
                            page_data.page_number = idx + 1
                        
                        if page_data.confidence_score >= self.fallback_threshold:
                            # Good confidence - use SmolDocling result
                            segment, page_visuals = self._create_segment_from_smoldocling(page_data)
                            segments.append(segment)
                            visual_elements.extend(page_visuals)
                        else:
                            # Low confidence - mark for fallback
                            pages_needing_fallback.append(page_data.page_number)
                            segment, page_visuals = self._create_segment_from_smoldocling(page_data)
                            segment.metadata['needs_review'] = True
                            segments.append(segment)
                            visual_elements.extend(page_visuals)
                    
                    # Mode 1: Use fallback for low-confidence pages
                    if self.pdfplumber_mode == 1 and pages_needing_fallback:
                        logger.info(f"⚠️ {len(pages_needing_fallback)} pages need fallback extraction")
                        self._apply_fallback_extraction(pdf_path, pages_needing_fallback, segments)
                    
                except Exception as e:
                    logger.error(f"❌ SmolDocling failed: {e}")
                    if self.pdfplumber_mode == 0:
                        raise ParseError(f"SmolDocling failed and fallback disabled: {e}")
                    else:
                        logger.info("📚 Falling back to pdfplumber for entire document")
                        segments, visual_elements = self._extract_with_pdfplumber(pdf_path)
            
            # Sort segments by page number for consistency
            segments.sort(key=lambda s: (s.page_number, s.segment_index))
            
            # Post-process segments
            segments = self._post_process_segments(segments)
            
            # Create visual segments
            visual_segments = self._create_visual_segments(visual_elements)
            
            # Merge all segments, maintaining page order
            all_segments = self._merge_segments_by_page(segments, visual_segments)
            
            # Create and return document
            elapsed = datetime.now() - start_time
            logger.info(f"✅ Parsing completed in {elapsed.total_seconds():.2f}s")
            logger.info(f"   - Total segments: {len(all_segments)}")
            logger.info(f"   - Visual elements: {len(visual_elements)}")
            
            return self.create_document(
                file_path=pdf_path,
                segments=all_segments,
                metadata=metadata,
                visual_elements=visual_elements
            )
            
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise ParseError(f"Failed to parse PDF: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'smoldocling_client') and hasattr(self.smoldocling_client, 'cleanup'):
            self.smoldocling_client.cleanup()
        if hasattr(self, 'vlm_processor') and self.vlm_processor:
            self.vlm_processor.cleanup()
        logger.info("HybridPDFParserQwen25 cleanup completed")
    
    # Methods copied from original HybridPDFParser
    def _create_segment_from_smoldocling(self, page_data) -> Tuple[Segment, List[VisualElement]]:
        """Convert SmolDocling page data to Segment"""
        # Simplified implementation - full implementation would be copied from original
        content = getattr(page_data, 'text', '')
        if not content:
            content = f"[Page {getattr(page_data, 'page_number', 1)}: Visual content only]"
        
        segment = Segment(
            content=content,
            page_number=getattr(page_data, 'page_number', 1),
            segment_type=SegmentType.TEXT,
            segment_subtype=TextSubtype.PARAGRAPH.value,
            metadata={
                "confidence": getattr(page_data, 'confidence_score', 0.0),
                "extractor": "smoldocling"
            }
        )
        
        # Extract visual elements if present
        visual_elements = []
        if hasattr(page_data, 'visual_elements') and page_data.visual_elements:
            visual_elements.extend(page_data.visual_elements)
            for ve in visual_elements:
                segment.visual_references.append(ve.content_hash)
        
        return segment, visual_elements
    
    def _parallel_extraction(self, pdf_path: Path) -> Tuple[List[Segment], List[VisualElement]]:
        """Parallel extraction with SmolDocling and pdfplumber"""
        segments = []
        visual_elements = []
        
        # Get both extractions
        try:
            smoldocling_result = self.smoldocling_client.parse_pdf(pdf_path)
            smoldocling_segments = {}
            for page in smoldocling_result.pages:
                segment, page_visuals = self._create_segment_from_smoldocling(page)
                smoldocling_segments[page.page_number] = segment
                visual_elements.extend(page_visuals)
        except:
            smoldocling_segments = {}
        
        pdfplumber_segments, _ = self._extract_with_pdfplumber(pdf_path)
        
        # Merge results
        all_pages = set(smoldocling_segments.keys()) | {s.page_number for s in pdfplumber_segments}
        
        for page_num in sorted(all_pages):
            smol_seg = smoldocling_segments.get(page_num)
            plumb_seg = next((s for s in pdfplumber_segments if s.page_number == page_num), None)
            
            if smol_seg and smol_seg.metadata.get('confidence', 0) >= self.fallback_threshold:
                segments.append(smol_seg)
            elif plumb_seg:
                segments.append(plumb_seg)
            elif smol_seg:
                segments.append(smol_seg)
        
        return segments, visual_elements
    
    def _apply_fallback_extraction(self, pdf_path: Path, pages: List[int], segments: List[Segment]) -> None:
        """Apply fallback extraction for specific pages"""
        fallback_segments, _ = self._extract_with_pdfplumber(pdf_path, pages)
        
        # Replace low-confidence segments
        for fallback_seg in fallback_segments:
            for i, seg in enumerate(segments):
                if seg.page_number == fallback_seg.page_number:
                    if len(fallback_seg.content) > len(seg.content) * 1.2:
                        segments[i] = fallback_seg
                    break
    
    def _extract_with_pdfplumber(self, pdf_path: Path, pages: Optional[List[int]] = None) -> Tuple[List[Segment], List[VisualElement]]:
        """Extract using pdfplumber"""
        segments = self.pdfplumber_extractor.extract(
            pdf_path, 
            pages=pages,
            layout_settings=self.layout_settings,
            separate_tables=self.separate_tables,
            use_bbox_filtering=False
        )
        return segments, []  # pdfplumber doesn't extract visual elements
    
    def _post_process_segments(self, segments: List[Segment]) -> List[Segment]:
        """Post-process segments for quality improvement and optional content detection"""
        processed = []
        
        for segment in segments:
            # Clean content
            if segment.content and segment.content.strip():
                # Only enhance with ContentDetector if enabled
                if self.enable_content_detection:
                    # Enhance segment with detected content type and structure
                    enhanced_segment = ContentDetector.enhance_segment_with_structure(segment)
                    
                    # Log if we detected special content
                    if enhanced_segment.segment_type == SegmentType.TABLE:
                        logger.info(f"📊 Detected table in segment {enhanced_segment.segment_index} on page {enhanced_segment.page_number}")
                        if 'table_structure' in enhanced_segment.metadata:
                            triple_count = enhanced_segment.metadata.get('triple_count', 0)
                            logger.info(f"   - Extracted {triple_count} potential triples")
                    elif enhanced_segment.segment_type == SegmentType.VISUAL and enhanced_segment.segment_subtype == VisualSubtype.CHART:
                        logger.info(f"📈 Detected chart reference in segment {enhanced_segment.segment_index}")
                    
                    processed.append(enhanced_segment)
                else:
                    # Skip content detection, just add the segment
                    processed.append(segment)
        
        return processed
    
    def _create_visual_segments(self, visual_elements: List[VisualElement]) -> List[Segment]:
        """Create segments from visual elements"""
        visual_segments = []
        
        for i, visual_elem in enumerate(visual_elements):
            # Determine visual subtype
            if visual_elem.element_type == VisualElementType.IMAGE:
                subtype = VisualSubtype.IMAGE.value
            elif visual_elem.element_type == VisualElementType.CHART:
                subtype = VisualSubtype.CHART.value
            elif visual_elem.element_type == VisualElementType.DIAGRAM:
                subtype = VisualSubtype.DIAGRAM.value
            elif visual_elem.element_type == VisualElementType.FORMULA:
                subtype = VisualSubtype.FORMULA.value
            else:
                subtype = VisualSubtype.OTHER.value
            
            # Create segment content
            content = f"[Visual Element: {visual_elem.element_type.value}]"
            if visual_elem.vlm_description:
                content = f"[{visual_elem.element_type.value}] {visual_elem.vlm_description}"
            
            segment = Segment(
                content=content,
                page_number=visual_elem.page_or_slide,
                segment_type=SegmentType.VISUAL,
                segment_subtype=subtype,
                segment_index=i,
                visual_references=[visual_elem.content_hash],
                metadata={
                    "visual_type": visual_elem.element_type.value,
                    "confidence": visual_elem.confidence,
                    "source": "pdf_visual_extraction"
                }
            )
            
            visual_segments.append(segment)
        
        return visual_segments
    
    def _merge_segments_by_page(self, text_segments: List[Segment], visual_segments: List[Segment]) -> List[Segment]:
        """Merge text and visual segments, maintaining page order"""
        # Group segments by page
        segments_by_page = {}
        
        # Add text segments
        for segment in text_segments:
            page = segment.page_number or 1
            if page not in segments_by_page:
                segments_by_page[page] = []
            segments_by_page[page].append(segment)
        
        # Add visual segments
        for segment in visual_segments:
            page = segment.page_number or 1
            if page not in segments_by_page:
                segments_by_page[page] = []
            segments_by_page[page].append(segment)
        
        # Merge and re-index
        all_segments = []
        for page in sorted(segments_by_page.keys()):
            page_segments = segments_by_page[page]
            for segment in page_segments:
                segment.segment_index = len(all_segments)
                all_segments.append(segment)
        
        return all_segments