"""
Final production-ready VLLMSmolDoclingClient with docling integration
Combines best of both approaches with direct image extraction
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from pdf2image import convert_from_path
import fitz  # PyMuPDF

from core.vllm.base_client import BaseVLLMClient
from core.parsers.interfaces.data_models import (
    VisualElement, VisualElementType, DocumentType
)
from core.parsers import ParseError
from config.docling_config import get_config, should_use_docling_for_document

logger = logging.getLogger(__name__)


@dataclass
class SmolDoclingPage:
    """Enhanced page data with visual elements"""
    page_number: int
    text: str
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    formulas: List[Dict[str, Any]]
    visual_elements: List[VisualElement] = field(default_factory=list)
    layout_info: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    _docling_doc: Optional[Any] = None  # DoclingDocument when available


@dataclass
class SmolDoclingResult:
    """Enhanced result with integrated visual elements"""
    pages: List[SmolDoclingPage]
    visual_elements: List[VisualElement]  # All visual elements
    metadata: Dict[str, Any]
    processing_time_seconds: float
    model_version: str
    total_pages: int
    success: bool
    error_message: Optional[str] = None
    extraction_method: str = "docling"  # docling only, no legacy fallback


class VLLMSmolDoclingFinalClient(BaseVLLMClient):
    """
    Production-ready SmolDocling client with optional docling integration
    """
    
    # Default prompt for SmolDocling
    PROMPT_TEXT = "Convert this page to docling."
    
    def __init__(
        self,
        model_id: str = "smoldocling",
        max_pages: int = 30,
        gpu_memory_utilization: float = 0.3,
        environment: str = "development",
        **kwargs
    ):
        """
        Initialize SmolDocling client with configuration-based setup
        
        Args:
            environment: Environment for configuration (development/production/testing)
        """
        # Load configuration first
        self.config = get_config(environment)
        self.environment = environment
        
        # Basic settings
        self.max_pages = max_pages
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # Configuration-driven settings
        self.use_docling = self.config["use_docling"]
        self.extract_images_directly = self.config["extract_images_directly"]
        # Legacy fallback removed - always use docling
        self.log_performance = self.config["log_performance"]
        
        # Import required classes for vLLM
        from core.vllm.model_manager import ModelConfig, SamplingConfig
        
        # Create model configuration
        model_config = ModelConfig(
            model_name="ds4sd/SmolDocling-256M-preview",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=8192,
            trust_remote_code=False,
            limit_mm_per_prompt={"image": 1},
            dtype="auto"
        )
        
        # Create sampling configuration
        sampling_config = SamplingConfig(
            temperature=0.0,
            max_tokens=8192
        )
        
        # Initialize parent class
        super().__init__(
            model_id=model_id,
            model_config=model_config,
            sampling_config=sampling_config,
            auto_load=False
        )
        
        # Check docling availability
        self._docling_available = False
        if self.use_docling:
            self._docling_available = self._check_docling()
            if not self._docling_available:
                logger.error("Docling not available, cannot proceed with parsing")
                self.use_docling = False
    
    def _check_docling(self) -> bool:
        """Check if docling is available"""
        try:
            from docling_core.types.doc import DoclingDocument
            from docling_core.types.doc.document import DocTagsDocument
            return True
        except ImportError:
            return False
    
    def _get_document_hash(self, pdf_path: Path) -> str:
        """Generate consistent hash for document routing"""
        import hashlib
        
        # Use file path and size for consistent but privacy-aware hashing
        content = f"{pdf_path.name}_{pdf_path.stat().st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def parse_pdf(self, pdf_path: Path) -> SmolDoclingResult:
        """
        Parse PDF with configuration-based method selection
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        start_time = time.time()
        
        # Check document-specific routing for rollout
        document_hash = self._get_document_hash(pdf_path)
        should_use_docling = should_use_docling_for_document(document_hash, self.environment)
        
        # Performance logging
        if self.log_performance:
            logger.info(f"Processing {pdf_path.name} (hash: {document_hash[:8]})")
        
        # Check file size limits
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        max_size = self.config["memory_limits"]["max_pdf_size_mb"]
        
        if file_size_mb > max_size:
            logger.warning(f"PDF too large ({file_size_mb:.1f}MB > {max_size}MB), skipping docling extraction")
            should_use_docling = False
        
        # Decide which method to use
        try:
            if should_use_docling and self._docling_available and self.extract_images_directly:
                logger.info("Using docling with direct image extraction")
                result = self._parse_with_docling_direct(pdf_path)
            elif should_use_docling and self._docling_available:
                logger.info("Using docling with deferred image extraction") 
                result = self._parse_with_docling_deferred(pdf_path)
            else:
                logger.info("Processing without docling extraction")
                # Process with basic SmolDocling only
            
            # Performance logging
            if self.log_performance:
                processing_time = time.time() - start_time
                threshold = self.config["performance_threshold_seconds"]
                if processing_time > threshold:
                    logger.warning(f"Slow processing: {processing_time:.1f}s > {threshold}s for {pdf_path.name}")
                else:
                    logger.info(f"Processing completed in {processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            if should_use_docling:
                logger.error(f"Docling parsing failed: {e}")
                # Re-raise the error - no fallback
            else:
                raise
    
    def _parse_with_docling_direct(self, pdf_path: Path) -> SmolDoclingResult:
        """
        Parse with docling and extract images immediately
        RECOMMENDED APPROACH
        """
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
        
        start_time = time.time()
        pages = []
        all_visual_elements = []
        
        try:
            # Open PDF once for image extraction
            pdf_doc = fitz.open(str(pdf_path))
            
            # Apply page limits from configuration
            max_pages_config = self.config["memory_limits"]["max_pages_per_batch"]
            actual_max_pages = min(self.max_pages, max_pages_config, pdf_doc.page_count)
            
            # Convert to images
            logger.info(f"Converting {pdf_path.name} to images (144 DPI), processing {actual_max_pages} pages...")
            page_images = convert_from_path(
                str(pdf_path),
                dpi=144,
                first_page=1,
                last_page=actual_max_pages,
                fmt='PNG'
            )
            
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Process pages
            for page_num, page_image in enumerate(page_images, 1):
                logger.info(f"Processing page {page_num}/{len(page_images)}...")
                
                try:
                    # Generate DocTags
                    doctags = self._generate_doctags(page_image)
                    
                    # Debug: Log raw DocTags output
                    logger.debug(f"Raw DocTags for page {page_num} (first 500 chars): {doctags[:500]}...")
                    
                    # Parse with docling
                    # Strip any leading/trailing whitespace from doctags
                    doctags_clean = doctags.strip()
                    
                    # Transform SmolDocling tags to docling-core compatible tags
                    doctags_transformed = self._transform_doctags(doctags_clean)
                    
                    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
                        [doctags_transformed], 
                        [page_image]
                    )
                    
                    # WICHTIG: load_from_doctags ist eine statische Methode, die ein neues Dokument zurückgibt
                    doc = DoclingDocument.load_from_doctags(
                        doctag_document=doctags_doc,
                        document_name=f"Page_{page_num}"
                    )
                    
                    # Debug: Log parsed document info
                    logger.debug(f"Parsed DoclingDocument for page {page_num}: texts={len(doc.texts)}, tables={len(doc.tables)}, pictures={len(doc.pictures)}")
                    
                    # Extract visuals WITH images
                    visual_elements = self._extract_visuals_direct(
                        doc, pdf_doc, page_num, page_image
                    )
                    all_visual_elements.extend(visual_elements)
                    
                    # Create page
                    page = self._create_page_from_docling(
                        doc, page_num, visual_elements, doctags
                    )
                    pages.append(page)
                    
                except Exception as e:
                    logger.error(f"Failed to process page {page_num}: {e}")
                    logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
                    
                    # Try fallback to direct DocTags parsing
                    try:
                        logger.info(f"Attempting direct DocTags parsing for page {page_num}")
                        page = self._parse_doctags_directly(
                            doctags, page_num, pdf_doc, page_image
                        )
                        pages.append(page)
                    except Exception as fallback_error:
                        logger.error(f"Fallback parsing also failed: {fallback_error}")
                        # Create empty page
                        pages.append(SmolDoclingPage(
                            page_number=page_num,
                            text=f"[Error processing page: {str(e)}]",
                            tables=[], images=[], formulas=[],
                            visual_elements=[], layout_info={},
                            confidence_score=0.0
                        ))
            
            pdf_doc.close()
            
            return SmolDoclingResult(
                pages=pages,
                visual_elements=all_visual_elements,
                metadata={
                    "filename": pdf_path.name,
                    "page_count": len(pages),
                    "visual_count": len(all_visual_elements)
                },
                processing_time_seconds=time.time() - start_time,
                model_version="smoldocling-docling-direct",
                total_pages=len(pages),
                success=True,
                extraction_method="docling"
            )
            
        except Exception as e:
            logger.error(f"Docling parsing failed: {e}")
            if hasattr(locals(), 'pdf_doc'):
                pdf_doc.close()
            
            # No fallback - raise error
            logger.error("SmolDocling parsing failed")
            raise ParseError(f"Failed to parse {pdf_path.name}: {str(e)}")
    
    def _generate_doctags(self, page_image) -> str:
        """Generate DocTags for a page using vLLM"""
        # Create chat template
        chat_template = f"<|im_start|>User:<image>{self.PROMPT_TEXT}<end_of_utterance>\nAssistant:"
        
        # Create multimodal input
        multimodal_input = {
            "prompt": chat_template,
            "multi_modal_data": {"image": page_image}
        }
        
        # Generate
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            skip_special_tokens=False
        )
        
        model = self.model_manager._models[self.model_id]
        outputs = model.generate(multimodal_input, sampling_params)
        
        if outputs and outputs[0].outputs:
            doctags = outputs[0].outputs[0].text
            # Check for repetition bug
            doctags = self._check_and_fix_repetitions(doctags)
            return doctags
        
        raise ParseError("No output from model")
    
    def _extract_visuals_direct(
        self, 
        doc,  # DoclingDocument
        pdf_doc,  # fitz.Document
        page_num: int,
        page_image  # PIL.Image
    ) -> List[VisualElement]:
        """Extract visuals with images directly"""
        visual_elements = []
        page = pdf_doc[page_num - 1]
        
        # Get scaling factors
        page_rect = page.rect
        scale_x = page_rect.width / 500.0
        scale_y = page_rect.height / 500.0
        
        # Extract pictures
        for picture in doc.pictures:
            bbox = self._extract_bbox_from_prov(picture)
            
            if bbox:
                try:
                    # Scale and extract
                    x0, y0, x1, y1 = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]
                    
                    rect = fitz.Rect(x0, y0, x1, y1)
                    
                    # Apply configuration-based image extraction settings
                    max_size = self.config["image_extraction"]["max_image_size"]
                    rect_width = rect.width
                    rect_height = rect.height
                    
                    # Calculate appropriate zoom to respect max_image_size
                    zoom = min(2.0, max_size / max(rect_width, rect_height)) if max(rect_width, rect_height) > 0 else 2.0
                    mat = fitz.Matrix(zoom, zoom)
                    
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    
                    # Use configured image quality for PNG
                    img_bytes = pix.tobytes("png")
                    
                    # Create visual element
                    visual = VisualElement(
                        element_type=VisualElementType.FIGURE,
                        source_format=DocumentType.PDF,
                        content_hash=VisualElement.create_hash(img_bytes),
                        page_or_slide=page_num,
                        bounding_box=bbox,
                        raw_data=img_bytes,
                        analysis_metadata={
                            "caption": getattr(picture, 'caption', ''),
                            "extracted_by": "docling_direct",
                            "original_type": "picture",
                            "raw_bbox": bbox
                        }
                    )
                    visual_elements.append(visual)
                    
                    logger.debug(
                        f"Extracted picture: "
                        f"page={page_num}, bbox={bbox}, "
                        f"size={len(img_bytes)} bytes"
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to extract visual: {e}")
        
        # Extract tables as images if configured
        if self.config["image_extraction"]["extract_tables_as_images"]:
            for table in doc.tables:
                bbox = self._extract_bbox_from_prov(table)
                
                if bbox:
                    try:
                        # Scale and extract
                        x0, y0, x1, y1 = [
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y
                        ]
                        
                        rect = fitz.Rect(x0, y0, x1, y1)
                        
                        # Apply configuration-based image extraction settings
                        max_size = self.config["image_extraction"]["max_image_size"]
                        zoom = min(2.0, max_size / max(rect.width, rect.height)) if max(rect.width, rect.height) > 0 else 2.0
                        mat = fitz.Matrix(zoom, zoom)
                        
                        pix = page.get_pixmap(matrix=mat, clip=rect)
                        img_bytes = pix.tobytes("png")
                        
                        # Create visual element
                        visual = VisualElement(
                            element_type=VisualElementType.TABLE,
                            source_format=DocumentType.PDF,
                            content_hash=VisualElement.create_hash(img_bytes),
                            page_or_slide=page_num,
                            bounding_box=bbox,
                            raw_data=img_bytes,
                            analysis_metadata={
                                "caption": getattr(table, 'caption', ''),
                                "extracted_by": "docling_direct",
                                "original_type": "table",
                                "raw_bbox": bbox
                            }
                        )
                        visual_elements.append(visual)
                        
                        logger.debug(
                            f"Extracted table as image: "
                            f"page={page_num}, bbox={bbox}, "
                            f"size={len(img_bytes)} bytes"
                        )
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract table as visual: {e}")
        
        return visual_elements
    
    def _extract_bbox_from_prov(self, element) -> Optional[List[int]]:
        """Extract bbox from element's provenance information"""
        # Try to get bbox from provenance
        if hasattr(element, 'prov') and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, 'bbox') and prov.bbox:
                bbox = prov.bbox
                if hasattr(bbox, 'x0'):
                    return [int(bbox.x0), int(bbox.y0), 
                           int(bbox.x1), int(bbox.y1)]
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    return [int(x) for x in bbox[:4]]
        
        # Fallback to element.bbox if exists
        if hasattr(element, 'bbox'):
            bbox = element.bbox
            if hasattr(bbox, 'x0'):
                return [int(bbox.x0), int(bbox.y0), 
                       int(bbox.x1), int(bbox.y1)]
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return [int(x) for x in bbox[:4]]
        
        # Try parsing from string as last resort
        element_str = str(element)
        
        # Pattern 1: <loc_x><loc_y>...
        loc_match = re.search(
            r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>', 
            element_str
        )
        if loc_match:
            return [int(loc_match.group(i)) for i in range(1, 5)]
        
        # Pattern 2: x>y>x2>y2>
        coord_match = re.search(
            r'(\d+)>(\d+)>(\d+)>(\d+)>', 
            element_str
        )
        if coord_match:
            return [int(coord_match.group(i)) for i in range(1, 5)]
        
        return None
    
    def _map_element_type(self, docling_type: str) -> VisualElementType:
        """Map docling type to our type"""
        mapping = {
            'picture': VisualElementType.IMAGE,
            'image': VisualElementType.IMAGE,
            'figure': VisualElementType.FIGURE,
            'chart': VisualElementType.CHART,
            'diagram': VisualElementType.DIAGRAM,
            'graph': VisualElementType.GRAPH,
            'table': VisualElementType.TABLE_IMAGE,
            'formula': VisualElementType.FORMULA
        }
        return mapping.get(docling_type, VisualElementType.UNKNOWN_VISUAL)
    
    def _create_page_from_docling(
        self, 
        doc,  # DoclingDocument
        page_num: int,
        visual_elements: List[VisualElement],
        raw_doctags: str
    ) -> SmolDoclingPage:
        """Create page from DoclingDocument"""
        # Extract content
        text_parts = []
        tables = []
        formulas = []
        images = []
        
        # Extract text elements
        logger.debug(f"Extracting from page {page_num}: {len(doc.texts)} text items")
        for i, text_item in enumerate(doc.texts):
            logger.debug(f"  Text item {i}: type={type(text_item).__name__}, has_text={hasattr(text_item, 'text')}")
            if hasattr(text_item, 'text'):
                text_parts.append(text_item.text)
                logger.debug(f"    Added text: {text_item.text[:50]}..." if text_item.text else "    Empty text")
            # Check if it's a formula item
            if hasattr(text_item, '__class__') and text_item.__class__.__name__ == 'FormulaItem':
                if hasattr(text_item, 'text'):
                    formulas.append({
                        "latex": text_item.text,
                        "type": "formula"
                    })
        
        # Extract tables
        for table_item in doc.tables:
            if hasattr(table_item, 'data'):
                tables.append({
                    "content": self._format_table(table_item.data),
                    "format": "text"
                })
        
        # Create images for compatibility
        for ve in visual_elements:
            images.append({
                "bbox": ve.bounding_box,
                "caption": ve.analysis_metadata.get("caption", ""),
                "type": ve.element_type.value,
                "has_data": ve.raw_data is not None
            })
        
        # If no text extracted from docling, try parsing DocTags directly
        if not text_parts and raw_doctags:
            logger.debug("No text from docling, attempting direct DocTags extraction")
            import re
            # Pattern to match DocTag elements with content
            pattern = r'<(\w+)>(?:<loc_\d+>){4}([^<]+)</\1>'
            
            for match in re.finditer(pattern, raw_doctags):
                tag_type = match.group(1)
                content = match.group(2).strip()
                
                if content and tag_type in ['paragraph', 'text', 'section_header', 'title', 'caption']:
                    text_parts.append(content)
                    logger.debug(f"  Extracted {tag_type}: {content[:50]}...")
                elif content and tag_type in ['formula', 'equation']:
                    formulas.append({
                        "latex": content,
                        "type": "formula"
                    })
        
        # Also try to extract from body.children if available
        if not text_parts and hasattr(doc, 'body') and hasattr(doc.body, 'children'):
            logger.debug(f"Checking body children: {len(doc.body.children)} items")
            for child in doc.body.children:
                if hasattr(child, 'text'):
                    text_parts.append(child.text)
                    logger.debug(f"  Extracted from body child: {child.text[:50]}...")
        
        return SmolDoclingPage(
            page_number=page_num,
            text="\n\n".join(text_parts),
            tables=tables,
            images=images,
            formulas=formulas,
            visual_elements=visual_elements,
            layout_info={
                "raw_content": raw_doctags[:1000] + "...",
                "source": "docling",
                "element_counts": {
                    "text": len(text_parts),
                    "tables": len(tables),
                    "images": len(images),
                    "formulas": len(formulas)
                }
            },
            confidence_score=1.0 if text_parts else 0.5,
            _docling_doc=doc
        )
    
    def _get_text(self, element) -> str:
        """Get text from element"""
        if hasattr(element, 'text'):
            return element.text
        elif hasattr(element, 'content'):
            return element.content
        return str(element)
    
    def _format_table(self, data) -> str:
        """Convert table to text"""
        if hasattr(data, 'to_markdown'):
            return data.to_markdown()
        elif hasattr(data, 'to_text'):
            return data.to_text()
        return str(data)
    
    def _ensure_model_loaded(self):
        """Ensure vLLM model is loaded"""
        if not self.model_manager.is_model_loaded(self.model_id):
            logger.info("Loading SmolDocling model...")
            self.model_manager.load_model(
                self.model_id,
                self.model_config
            )
    
    # Legacy parsing method removed - no fallback available
    
    def parse_model_output(self, output: Any) -> Dict[str, Any]:
        """
        Parse model output for compatibility with BaseVLLMClient
        Note: This client primarily uses parse_pdf with docling integration
        """
        try:
            # Extract text from vLLM output
            if hasattr(output, 'outputs') and output.outputs:
                content = output.outputs[0].text
            elif hasattr(output, 'text'):
                content = output.text
            else:
                content = str(output)
            
            # For compatibility, return basic structure
            # In practice, this client uses parse_pdf with docling integration
            return {
                "text": content.strip(),
                "tables": [],
                "images": [],  # Images are handled by parse_pdf with direct extraction
                "formulas": [],
                "code_blocks": [],
                "text_blocks": [{"text": content.strip(), "bbox": None}],
                "titles": [],
                "layout_info": {
                    "raw_content": content,
                    "source": "final_client_fallback",
                    "note": "Use parse_pdf for full docling integration"
                },
                "confidence_score": 1.0 if content.strip() else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to parse model output: {e}")
            return {
                "text": "",
                "tables": [],
                "images": [],
                "formulas": [],
                "code_blocks": [],
                "text_blocks": [],
                "titles": [],
                "layout_info": {"error": str(e)},
                "confidence_score": 0.0
            }
    
    def _parse_with_docling_deferred(self, pdf_path: Path) -> SmolDoclingResult:
        """Parse with docling but defer image extraction"""
        # This would be similar to _parse_with_docling_direct
        # but without immediate image extraction
        # Images would be extracted later by HybridPDFParser
        pass
    
    def _check_and_fix_repetitions(self, doctags: str) -> str:
        """
        Check for and fix the known SmolDocling repetition bug
        where tokens repeat endlessly from some point in the output
        """
        import re
        
        # Detect repetition patterns
        # Look for sequences that repeat more than a threshold
        lines = doctags.split('\n')
        
        # Track consecutive identical lines
        repetition_threshold = 10  # If same content repeats >10 times, it's likely a bug
        last_line = None
        repetition_count = 0
        truncate_at = None
        
        for i, line in enumerate(lines):
            # Skip empty lines in comparison
            if not line.strip():
                continue
                
            if line == last_line:
                repetition_count += 1
                if repetition_count >= repetition_threshold:
                    truncate_at = i - repetition_count
                    logger.warning(
                        f"Detected token repetition bug at line {i}. "
                        f"Same content repeated {repetition_count + 1} times: '{line[:50]}...'"
                    )
                    break
            else:
                repetition_count = 0
                last_line = line
        
        # Also check for repeating patterns in tags (e.g., same cell value repeated)
        # Pattern: multiple consecutive identical tags with same content
        tag_pattern = r'<(\w+)>(?:<loc_\d+>)*([^<]+)</\1>'
        matches = list(re.finditer(tag_pattern, doctags))
        
        # Check for suspicious repetitions in tag content
        content_counts = {}
        for match in matches:
            tag_type = match.group(1)
            content = match.group(2).strip()
            
            # Skip common repetitive values that are legitimate
            if content in ['', ' ', '-', 'n.a.', 'n.a', 'N/A', '0', '1']:
                continue
                
            key = f"{tag_type}:{content}"
            content_counts[key] = content_counts.get(key, 0) + 1
        
        # Log suspicious repetitions
        for key, count in content_counts.items():
            if count > 20:  # More than 20 identical tag contents is suspicious
                tag_type, content = key.split(':', 1)
                logger.warning(
                    f"Suspicious repetition: Tag '{tag_type}' "
                    f"contains '{content[:30]}...' {count} times"
                )
        
        # If we detected line-based repetition, truncate
        if truncate_at is not None:
            logger.warning(f"Truncating DocTags output at line {truncate_at} due to repetition bug")
            lines = lines[:truncate_at]
            
            # Check if we have enough content before the repetition
            doctags_truncated = '\n'.join(lines)
            
            # Count meaningful tags (exclude positioning tags)
            meaningful_tags = re.findall(r'<(text|paragraph|section_header|title|caption|table|picture|formula|code)>', doctags_truncated)
            
            if len(meaningful_tags) < 2:  # Too little content before repetition
                error_msg = (
                    f"SmolDocling repetition bug detected too early (line {truncate_at}). "
                    f"Only {len(meaningful_tags)} meaningful tags found before repetition. "
                    "Page cannot be parsed reliably."
                )
                logger.error(error_msg)
                raise ParseError(error_msg)
            
            # Try to close any open tags
            open_tags = []
            for match in re.finditer(r'<(\w+)>', doctags_truncated):
                tag = match.group(1)
                if tag not in ['loc', 'nl']:  # Skip self-closing tags
                    open_tags.append(tag)
            
            for match in re.finditer(r'</(\w+)>', doctags_truncated):
                tag = match.group(1)
                if tag in open_tags:
                    open_tags.remove(tag)
            
            # Add closing tags
            for tag in reversed(open_tags):
                doctags_truncated += f'\n</{tag}>'
            
            # Add warning tag for debugging
            doctags_truncated += '\n<!-- WARNING: Output truncated due to SmolDocling repetition bug -->'
            
            return doctags_truncated
        
        return doctags
    
    def _transform_doctags(self, doctags: str) -> str:
        """Transform SmolDocling DocTags to docling-core compatible format"""
        # SmolDocling generiert Tags, die docling-core nicht erkennt
        # Basierend auf der Analyse des tag_to_doclabel Mappings in load_from_doctags
        
        tag_mapping = {
            # Text-bezogene Tags
            '<paragraph>': '<text>',
            '</paragraph>': '</text>',
            '<para>': '<text>',
            '</para>': '</text>',
            
            # Section headers - docling erwartet numbered levels
            '<section_header>': '<section_header_level_1>',
            '</section_header>': '</section_header_level_1>',
            
            # Diese bleiben gleich (bereits im Mapping)
            # 'title', 'caption', 'formula', 'code', 'table', 'picture'
        }
        
        transformed = doctags
        for old_tag, new_tag in tag_mapping.items():
            transformed = transformed.replace(old_tag, new_tag)
        
        return transformed
    
    def _parse_doctags_directly(self, doctags: str, page_num: int, pdf_doc: Any, page_image: Any) -> SmolDoclingPage:
        """Parse DocTags directly without docling library as fallback"""
        import re
        
        text_parts = []
        tables = []
        images = []
        formulas = []
        visual_elements = []
        
        # Clean doctags
        doctags_clean = doctags.strip()
        
        # Extract text from various DocTag elements
        # Pattern to match DocTag elements with content
        # Format: <tag><loc_x1><loc_y1><loc_x2><loc_y2>content</tag>
        pattern = r'<(\w+)>(?:<loc_\d+>){4}([^<]+)</\1>'
        
        for match in re.finditer(pattern, doctags_clean):
            tag_type = match.group(1)
            content = match.group(2).strip()
            
            if content and tag_type in ['paragraph', 'text', 'section_header', 'title', 'caption']:
                text_parts.append(content)
                logger.debug(f"  Extracted {tag_type}: {content[:50]}...")
            elif content and tag_type == 'page_header':
                # Skip page numbers unless they contain meaningful text
                if not content.isdigit():
                    text_parts.append(content)
            elif content and tag_type in ['formula', 'equation']:
                formulas.append({
                    "latex": content,
                    "type": "formula"
                })
            elif tag_type == 'table':
                # Extract table content if available
                tables.append({
                    "content": content,
                    "format": "text"
                })
        
        # Extract visual elements (pictures, figures, charts)
        visual_pattern = r'<(picture|image|figure|chart)>(?:<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>)(?:([^<]*)</\1>)?'
        
        for match in re.finditer(visual_pattern, doctags_clean):
            tag_type = match.group(1)
            bbox = [int(match.group(i)) for i in range(2, 6)]
            caption = match.group(6) or ""
            
            images.append({
                "bbox": bbox,
                "caption": caption.strip() if caption else "",
                "type": tag_type,
                "has_data": False
            })
            
            # Try to extract visual element with image data
            try:
                page = pdf_doc[page_num - 1]
                page_rect = page.rect
                scale_x = page_rect.width / 500.0
                scale_y = page_rect.height / 500.0
                
                x0, y0, x1, y1 = [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                ]
                
                rect = fitz.Rect(x0, y0, x1, y1)
                
                # Use configuration for image extraction
                max_size = self.config["image_extraction"]["max_image_size"]
                zoom = min(2.0, max_size / max(rect.width, rect.height)) if max(rect.width, rect.height) > 0 else 2.0
                mat = fitz.Matrix(zoom, zoom)
                
                pix = page.get_pixmap(matrix=mat, clip=rect)
                img_bytes = pix.tobytes("png")
                
                visual = VisualElement(
                    element_type=self._map_element_type(tag_type),
                    source_format=DocumentType.PDF,
                    content_hash=VisualElement.create_hash(img_bytes),
                    page_or_slide=page_num,
                    bounding_box=bbox,
                    raw_data=img_bytes,
                    analysis_metadata={
                        "caption": caption.strip() if caption else "",
                        "extracted_by": "direct_doctags_fallback",
                        "original_type": tag_type
                    }
                )
                visual_elements.append(visual)
                images[-1]["has_data"] = True
                
            except Exception as e:
                logger.debug(f"Failed to extract visual from DocTags: {e}")
        
        # If no text found, extract any text-like content more broadly
        if not text_parts:
            # Try to find any text between tags that's not a location
            text_pattern = r'>([^<]+)<'
            for match in re.finditer(text_pattern, doctags_clean):
                text = match.group(1).strip()
                # Filter out location tags, single numbers, and very short text
                if (text and 
                    not text.startswith('loc_') and 
                    not (text.isdigit() and len(text) <= 2) and 
                    len(text) > 3):
                    text_parts.append(text)
                    logger.debug(f"  Extracted generic text: {text[:50]}...")
        
        # Log extraction summary
        logger.info(f"Direct DocTags parsing for page {page_num}: "
                   f"{len(text_parts)} text blocks, {len(images)} images, "
                   f"{len(tables)} tables, {len(formulas)} formulas")
        
        return SmolDoclingPage(
            page_number=page_num,
            text="\n\n".join(text_parts) if text_parts else "[No text content detected]",
            tables=tables,
            images=images,
            formulas=formulas,
            visual_elements=visual_elements,
            layout_info={
                "raw_content": doctags_clean[:1000] + "...",
                "source": "direct_doctags_fallback",
                "parsed_elements": len(text_parts) + len(images) + len(formulas) + len(tables)
            },
            confidence_score=0.8  # Lower confidence for fallback parsing
        )