"""
Image Analyzer - Unified image analysis using Qwen2.5-VL
Handles both embedded images and full page analysis
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from core.parsers.interfaces.data_models import (
    VisualElement, VisualElementType, DocumentType, 
    Segment, SegmentType, VisualSubtype
)
from .pdf_preprocessor import PreprocessResult
from core.vlm.qwen25_processor import Qwen25VLMProcessor, VisualAnalysisResult

logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Analysis modes for ImageAnalyzer"""
    EMBEDDED = "embedded"      # Only analyze embedded images
    PAGES = "pages"           # Only analyze full pages
    BOTH = "both"            # Analyze both embedded and pages


@dataclass
class ImageAnalysisResult:
    """Container for all image analysis results"""
    embedded_images: List[VisualElement]
    page_analyses: Dict[int, VisualAnalysisResult]  # page_num -> analysis
    visual_segments: List[Segment]  # Segments for visual elements
    
    def __init__(self):
        self.embedded_images = []
        self.page_analyses = {}
        self.visual_segments = []


class ImageAnalyzer:
    """
    Analyzes images from PDFs using Qwen2.5-VL
    - Supports embedded images and full page analysis
    - Configurable analysis modes
    - Creates visual segments with VLM descriptions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Analysis configuration
        self.analysis_mode = AnalysisMode(
            self.config.get("analysis_mode", AnalysisMode.EMBEDDED.value)
        )
        self.max_pages = self.config.get("max_pages", 20)
        self.analyze_embedded = self.analysis_mode in [AnalysisMode.EMBEDDED, AnalysisMode.BOTH]
        self.analyze_pages = self.analysis_mode in [AnalysisMode.PAGES, AnalysisMode.BOTH]
        
        # VLM configuration
        vlm_config = self.config.get("vlm_config", {})
        self.vlm_processor = Qwen25VLMProcessor(vlm_config)
        
        # Analysis prompts
        self.embedded_prompt = self.config.get(
            "embedded_prompt",
            "Describe this image in detail. What type of visual is it (chart, diagram, photo, etc.)? "
            "What information does it convey?"
        )
        self.page_prompt = self.config.get(
            "page_prompt",
            "Analyze this document page. Describe the overall layout, key visual elements, "
            "and how different components relate to each other."
        )
        
        logger.info(f"🎯 ImageAnalyzer initialized with mode: {self.analysis_mode.value}")
    
    async def analyze(
        self, 
        pdf_path: Path, 
        preprocess_result: PreprocessResult
    ) -> ImageAnalysisResult:
        """
        Main analysis method
        
        Args:
            pdf_path: Path to PDF file
            preprocess_result: Preprocessing results with image paths
            
        Returns:
            ImageAnalysisResult with all analyses
        """
        logger.info(f"🔍 Starting image analysis for: {pdf_path}")
        
        result = ImageAnalysisResult()
        
        # Analyze embedded images
        if self.analyze_embedded:
            embedded_results = await self._analyze_embedded_images(
                preprocess_result.embedded_images
            )
            result.embedded_images.extend(embedded_results)
            
            # Create segments for embedded images
            for img in embedded_results:
                segment = self._create_visual_segment(img)
                if segment:
                    result.visual_segments.append(segment)
        
        # Analyze full pages
        if self.analyze_pages:
            page_results = await self._analyze_pages(
                preprocess_result.page_images
            )
            result.page_analyses.update(page_results)
        
        logger.info(
            f"✅ Analysis complete: {len(result.embedded_images)} embedded images, "
            f"{len(result.page_analyses)} page analyses"
        )
        
        return result
    
    async def _analyze_embedded_images(
        self, 
        embedded_images: List[Dict[str, Any]]
    ) -> List[VisualElement]:
        """Analyze embedded images"""
        logger.info(f"📸 Analyzing {len(embedded_images)} embedded images")
        
        visual_elements = []
        
        # Process in batches for efficiency
        batch_size = 5
        for i in range(0, len(embedded_images), batch_size):
            batch = embedded_images[i:i + batch_size]
            
            # Create tasks for parallel processing
            tasks = []
            for img_info in batch:
                task = self._analyze_single_image(img_info)
                tasks.append(task)
            
            # Run batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for img_info, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to analyze image {img_info['path']}: {result}")
                    continue
                    
                if result:
                    visual_elements.append(result)
        
        return visual_elements
    
    async def _analyze_single_image(self, img_info: Dict[str, Any]) -> Optional[VisualElement]:
        """Analyze a single embedded image"""
        try:
            image_path = Path(img_info["path"])
            if not image_path.exists():
                logger.warning(f"⚠️ Image not found: {image_path}")
                return None
            
            # Analyze with VLM
            logger.debug(f"🔍 Analyzing image: {image_path}")
            analysis = await self.vlm_processor.analyze_image(
                str(image_path),
                prompt=self.embedded_prompt
            )
            
            if not analysis:
                logger.warning(f"⚠️ No analysis result for: {image_path}")
                return None
            
            # Determine element type from analysis
            element_type = self._determine_element_type(analysis)
            
            # Create VisualElement
            visual_element = VisualElement(
                element_type=element_type,
                source_format=DocumentType.PDF,
                content_hash=img_info["hash_ref"],
                vlm_description=analysis.description,
                extracted_data=analysis.structured_data,
                confidence=analysis.confidence,
                bounding_box=img_info.get("bbox"),
                page_or_slide=img_info.get("page"),
                file_extension=img_info.get("format", "png"),
                analysis_metadata={
                    "prompt": self.embedded_prompt,
                    "model": "qwen2.5-vl",
                    "index": img_info.get("index"),
                    "processing_time": analysis.processing_time
                }
            )
            
            return visual_element
            
        except Exception as e:
            logger.error(f"❌ Error analyzing image {img_info.get('path')}: {e}")
            return None
    
    async def _analyze_pages(
        self, 
        page_images: Dict[int, str]
    ) -> Dict[int, VisualAnalysisResult]:
        """Analyze full pages"""
        logger.info(f"📄 Analyzing {len(page_images)} pages")
        
        page_analyses = {}
        
        # Limit pages to analyze
        pages_to_analyze = list(page_images.items())[:self.max_pages]
        
        # Process in batches
        batch_size = 3
        for i in range(0, len(pages_to_analyze), batch_size):
            batch = pages_to_analyze[i:i + batch_size]
            
            # Create tasks
            tasks = []
            for page_num, image_path in batch:
                task = self._analyze_single_page(page_num, image_path)
                tasks.append(task)
            
            # Run batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for (page_num, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to analyze page {page_num}: {result}")
                    continue
                    
                if result:
                    page_analyses[page_num] = result
        
        return page_analyses
    
    async def _analyze_single_page(
        self, 
        page_num: int, 
        image_path: str
    ) -> Optional[VisualAnalysisResult]:
        """Analyze a single page"""
        try:
            path = Path(image_path)
            if not path.exists():
                logger.warning(f"⚠️ Page image not found: {path}")
                return None
            
            logger.debug(f"🔍 Analyzing page {page_num}")
            analysis = await self.vlm_processor.analyze_image(
                str(path),
                prompt=self.page_prompt
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing page {page_num}: {e}")
            return None
    
    def _determine_element_type(self, analysis: VisualAnalysisResult) -> VisualElementType:
        """Determine visual element type from VLM analysis"""
        description_lower = analysis.description.lower()
        
        # Check for specific types in description
        type_keywords = {
            VisualElementType.CHART: ["chart", "graph", "plot", "visualization"],
            VisualElementType.DIAGRAM: ["diagram", "flowchart", "schematic", "architecture"],
            VisualElementType.TABLE: ["table", "grid", "spreadsheet"],
            VisualElementType.SCREENSHOT: ["screenshot", "screen capture", "interface"],
            VisualElementType.FORMULA: ["formula", "equation", "mathematical"],
            VisualElementType.MAP: ["map", "geographic", "location"],
            VisualElementType.DRAWING: ["drawing", "sketch", "illustration"],
        }
        
        for element_type, keywords in type_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return element_type
        
        # Default to IMAGE
        return VisualElementType.IMAGE
    
    def _create_visual_segment(self, visual_element: VisualElement) -> Optional[Segment]:
        """Create a segment for a visual element"""
        if not visual_element.vlm_description:
            return None
        
        # Map element type to segment subtype
        subtype_mapping = {
            VisualElementType.CHART: VisualSubtype.CHART,
            VisualElementType.DIAGRAM: VisualSubtype.DIAGRAM,
            VisualElementType.FORMULA: VisualSubtype.FORMULA,
            VisualElementType.SCREENSHOT: VisualSubtype.SCREENSHOT,
        }
        
        subtype = subtype_mapping.get(
            visual_element.element_type, 
            VisualSubtype.IMAGE
        )
        
        return Segment(
            content=f"[Visual: {visual_element.element_type.value}] {visual_element.vlm_description}",
            page_number=visual_element.page_or_slide,
            segment_index=0,  # Will be set by processor
            segment_type=SegmentType.VISUAL,
            segment_subtype=subtype.value,
            metadata={
                "visual_hash": visual_element.content_hash,
                "confidence": visual_element.confidence,
                "bbox": visual_element.bounding_box,
                "analysis_metadata": visual_element.analysis_metadata
            },
            visual_references=[visual_element.content_hash]
        )
    
    def get_page_context(self, page_num: int, page_analyses: Dict[int, VisualAnalysisResult]) -> Optional[str]:
        """Get page context description for a specific page"""
        if page_num in page_analyses:
            return page_analyses[page_num].description
        return None