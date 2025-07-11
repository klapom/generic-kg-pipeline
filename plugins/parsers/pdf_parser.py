"""PDF parser with vLLM SmolDocling and multi-modal support"""

import asyncio
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from PyPDF2 import PdfReader

from core.clients.vllm_smoldocling import VLLMSmolDoclingClient
from plugins.parsers.base_parser import (
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
        self.supported_extensions = {".pdf"}
        
        # PDF-specific configuration
        self.config.setdefault("extract_images", True)
        self.config.setdefault("extract_tables", True)
        self.config.setdefault("extract_formulas", True)
        self.config.setdefault("image_min_size", 50)  # Min pixels for image extraction
        self.config.setdefault("image_formats", ["PNG", "JPEG", "JPG"])
        
        logger.info(f"Initialized PDF parser with VLM: {enable_vlm}")
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a PDF"""
        return file_path.suffix.lower() in self.supported_extensions
    
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
            logger.info("🤖 Starting SmolDocling parsing (LOCAL vLLM)...")
            parsed_result = await self._parse_with_smoldocling(file_path)
            logger.info(f"✅ SmolDocling parsing completed - {len(parsed_result.pages)} pages processed")
            
            # Log each page processing
            for i, page in enumerate(parsed_result.pages, 1):
                text_length = len(page.text) if page.text else 0
                tables_count = len(page.tables) if page.tables else 0
                images_count = len(page.images) if page.images else 0
                formulas_count = len(page.formulas) if page.formulas else 0
                logger.info(f"📑 Page {i}: {text_length} chars text, {tables_count} tables, {images_count} images, {formulas_count} formulas")
            
            # Convert SmolDocling result to Document using the built-in method
            logger.info("🔄 Converting SmolDocling result to Document...")
            document = self.vllm_client.convert_to_document(parsed_result, file_path)
            
            logger.info(f"✅ PDF parsing completed: {len(document.segments)} segments, "
                       f"{len(document.visual_elements)} visual elements")
            
            # Analyze visual elements with VLM if enabled
            if document.visual_elements and self.enable_vlm:
                logger.info("🔍 Starting VLM analysis of visual elements (LOCAL Qwen2.5-VL)...")
                document_context = {
                    "document_title": document.metadata.title,
                    "document_type": "PDF",
                    "total_pages": document.metadata.page_count
                }
                document.visual_elements = await self.analyze_visual_elements(document.visual_elements, document_context)
                logger.info(f"✅ VLM analysis completed for {len(document.visual_elements)} elements")
            
            return document
            
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}", exc_info=True)
            raise ParseError(f"Failed to parse PDF {file_path.name}: {str(e)}")
    
    def _extract_pdf_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from PDF file"""
        try:
            # Get base metadata
            metadata = self.extract_metadata(file_path)
            metadata.document_type = DocumentType.PDF
            
            # Extract PDF-specific metadata
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Update page count
                metadata.page_count = len(reader.pages)
                
                # Extract PDF metadata if available
                if reader.metadata:
                    pdf_meta = reader.metadata
                    metadata.title = pdf_meta.get('/Title', metadata.title)
                    metadata.author = pdf_meta.get('/Author')
                    
                    # Parse creation date
                    creation_date = pdf_meta.get('/CreationDate')
                    if creation_date:
                        # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
                        try:
                            from datetime import datetime
                            if creation_date.startswith('D:'):
                                date_str = creation_date[2:16]  # YYYYMMDDHHMMSS
                                metadata.created_date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                        except (ValueError, TypeError):
                            pass
                
                # Add custom metadata
                metadata.custom_metadata.update({
                    "pdf_version": reader.pdf_header if hasattr(reader, 'pdf_header') else "Unknown",
                    "encrypted": reader.is_encrypted,
                    "pages": metadata.page_count
                })
                
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
            # Return basic metadata
            metadata = self.extract_metadata(file_path)
            metadata.document_type = DocumentType.PDF
            return metadata
    
    async def _parse_with_smoldocling(self, file_path: Path):
        """Parse PDF using vLLM SmolDocling"""
        try:
            # Initialize SmolDocling client if not already done
            if not self.vllm_client:
                from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
                self.vllm_client = VLLMSmolDoclingClient()
            
            # Parse with SmolDocling (local client is synchronous)
            result = self.vllm_client.parse_pdf(file_path)
            
            if not result.success:
                raise ParseError(f"SmolDocling parsing failed: {result.error_message}")
            
            # Return the SmolDoclingResult object directly
            return result
            
        except Exception as e:
            logger.error(f"SmolDocling parsing failed: {e}")
            raise ParseError(f"SmolDocling parsing failed: {str(e)}")
    
    def _extract_visual_elements(self, file_path: Path, parsed_result) -> List[VisualElement]:
        """Extract visual elements from PDF"""
        visual_elements = []
        
        try:
            # Extract images using PyPDF2 for visual elements
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                
                for page_num, page in enumerate(reader.pages, 1):
                    if '/XObject' in page['/Resources']:
                        xobjects = page['/Resources']['/XObject'].get_object()
                        
                        for obj_name, obj_ref in xobjects.items():
                            obj = obj_ref.get_object()
                            
                            # Check if it's an image
                            if obj.get('/Subtype') == '/Image':
                                try:
                                    image_data = self._extract_image_data(obj)
                                    if image_data and len(image_data) > 1000:  # Minimum size check
                                        
                                        # Determine image type
                                        img_type = self._determine_image_type(image_data)
                                        
                                        # Create visual element
                                        visual_element = VisualElement(
                                            element_type=img_type,
                                            source_format=DocumentType.PDF,
                                            content_hash=VisualElement.create_hash(image_data),
                                            raw_data=image_data,
                                            page_or_slide=page_num,
                                            file_extension=self._get_image_extension(obj),
                                            analysis_metadata={
                                                "pdf_object_name": obj_name,
                                                "extraction_method": "PyPDF2"
                                            }
                                        )
                                        
                                        visual_elements.append(visual_element)
                                        
                                except Exception as e:
                                    logger.warning(f"Failed to extract image {obj_name} from page {page_num}: {e}")
                                    continue
            
            # Also extract images from SmolDocling result if available
            if parsed_result and "images" in parsed_result:
                for img_info in parsed_result["images"]:
                    try:
                        if "image_data" in img_info:
                            image_data = img_info["image_data"]
                            if isinstance(image_data, str):
                                # Base64 decode if necessary
                                import base64
                                image_data = base64.b64decode(image_data)
                            
                            img_type = self._determine_image_type(image_data)
                            
                            visual_element = VisualElement(
                                element_type=img_type,
                                source_format=DocumentType.PDF,
                                content_hash=VisualElement.create_hash(image_data),
                                raw_data=image_data,
                                page_or_slide=img_info.get("page", 1),
                                bounding_box=img_info.get("bbox"),
                                file_extension="png",
                                analysis_metadata={
                                    "extraction_method": "SmolDocling",
                                    "smoldocling_metadata": img_info
                                }
                            )
                            
                            visual_elements.append(visual_element)
                            
                    except Exception as e:
                        logger.warning(f"Failed to process SmolDocling image: {e}")
                        continue
            
            # Remove duplicates based on content hash
            unique_elements = {}
            for element in visual_elements:
                if element.content_hash not in unique_elements:
                    unique_elements[element.content_hash] = element
            
            visual_elements = list(unique_elements.values())
            
            logger.info(f"Extracted {len(visual_elements)} visual elements from PDF")
            return visual_elements
            
        except Exception as e:
            logger.error(f"Visual element extraction failed: {e}")
            return []
    
    def _extract_image_data(self, img_obj) -> Optional[bytes]:
        """Extract raw image data from PDF image object"""
        try:
            # Get image data
            data = img_obj.get_data()
            
            # Handle different image formats
            if img_obj.get('/Filter') == '/DCTDecode':
                # JPEG image
                return data
            elif img_obj.get('/Filter') == '/FlateDecode':
                # PNG or other compressed format
                return data
            else:
                # Try to convert to PNG
                width = img_obj.get('/Width')
                height = img_obj.get('/Height')
                
                if width and height:
                    # Create PIL image and convert to PNG
                    img = Image.frombytes('RGB', (width, height), data)
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    return img_buffer.getvalue()
                    
        except Exception as e:
            logger.debug(f"Failed to extract image data: {e}")
            return None
    
    def _determine_image_type(self, image_data: bytes) -> VisualElementType:
        """Determine the type of visual element based on image data"""
        try:
            # Check image content to classify type
            img = Image.open(io.BytesIO(image_data))
            
            # Simple heuristics for image type classification
            width, height = img.size
            
            # Very wide or tall images might be charts/graphs
            aspect_ratio = width / height
            if aspect_ratio > 3 or aspect_ratio < 0.3:
                return VisualElementType.CHART
            
            # Square-ish images might be diagrams
            if 0.8 <= aspect_ratio <= 1.2:
                return VisualElementType.DIAGRAM
            
            # Default to generic image
            return VisualElementType.IMAGE
            
        except Exception:
            return VisualElementType.UNKNOWN_VISUAL
    
    def _get_image_extension(self, img_obj) -> str:
        """Get appropriate file extension for image"""
        filter_type = img_obj.get('/Filter')
        if filter_type == '/DCTDecode':
            return 'jpg'
        elif filter_type == '/FlateDecode':
            return 'png'
        else:
            return 'png'  # Default
    
    def _create_segments(self, parsed_result, visual_elements: List[VisualElement]) -> List[Segment]:
        """Create text segments from SmolDocling structured content"""
        try:
            # Use the enhanced SmolDocling context parser
            from plugins.parsers.smoldocling_context_parser import SmolDoclingContextParser
            
            context_parser = SmolDoclingContextParser(self.config)
            segments, enhanced_visual_elements = context_parser.parse_smoldocling_result(parsed_result)
            
            # Update visual elements with SmolDocling-provided information
            visual_elements.extend(enhanced_visual_elements)
            
            logger.info(f"Created {len(segments)} text segments from SmolDocling structured data")
            return segments
            
        except Exception as e:
            logger.error(f"SmolDocling context parsing failed, falling back to basic parsing: {e}")
            # Fallback to basic parsing
            return self._create_segments_fallback(parsed_result, visual_elements)
    
    def _create_segments_fallback(self, parsed_result: Dict[str, Any], visual_elements: List[VisualElement]) -> List[Segment]:
        """Fallback segment creation for when SmolDocling structure is not available"""
        segments = []
        
        try:
            # Extract text content from SmolDocling result
            if "text_blocks" in parsed_result:
                for i, block in enumerate(parsed_result["text_blocks"]):
                    try:
                        content = block.get("text", "").strip()
                        if content:
                            # Create visual references for this segment
                            visual_refs = []
                            page_num = block.get("page", 1)
                            
                            # Link to visual elements on same page
                            for ve in visual_elements:
                                if ve.page_or_slide == page_num:
                                    visual_refs.append(ve.content_hash)
                            
                            segment = Segment(
                                content=content,
                                page_number=page_num,
                                segment_index=i,
                                segment_type=block.get("type", "text"),
                                visual_references=visual_refs,
                                metadata={
                                    "bbox": block.get("bbox"),
                                    "confidence": block.get("confidence", 1.0),
                                    "extraction_method": "SmolDocling_fallback"
                                }
                            )
                            segments.append(segment)
                            
                    except Exception as e:
                        logger.warning(f"Failed to create segment {i}: {e}")
                        continue
            
            # Fallback: create segments from raw text if text_blocks not available
            elif "text" in parsed_result:
                text_content = parsed_result["text"]
                if text_content:
                    # Simple paragraph-based segmentation
                    paragraphs = text_content.split('\n\n')
                    for i, paragraph in enumerate(paragraphs):
                        paragraph = paragraph.strip()
                        if paragraph:
                            segment = Segment(
                                content=paragraph,
                                segment_index=i,
                                segment_type="paragraph",
                                metadata={"extraction_method": "basic_fallback"}
                            )
                            segments.append(segment)
            
            # If no segments created, create a single segment with all content
            if not segments and parsed_result:
                content = str(parsed_result.get("text", ""))
                if content:
                    segment = Segment(
                        content=content,
                        segment_index=0,
                        segment_type="document",
                        metadata={"extraction_method": "final_fallback"}
                    )
                    segments.append(segment)
            
            logger.info(f"Created {len(segments)} text segments from PDF (fallback)")
            return segments
            
        except Exception as e:
            logger.error(f"Segment creation failed: {e}")
            # Return empty segments list rather than failing
            return []
    
    def _build_document_content(self, segments: List[Segment], visual_elements: List[VisualElement]) -> str:
        """Build full document content from segments and visual descriptions"""
        content_parts = []
        
        # Add text content
        for segment in segments:
            content_parts.append(segment.content)
        
        # Add visual element descriptions
        for visual in visual_elements:
            if visual.vlm_description:
                visual_desc = f"\n[VISUAL ELEMENT - {visual.element_type.value.upper()}]"
                if visual.page_or_slide:
                    visual_desc += f" (Page {visual.page_or_slide})"
                visual_desc += f": {visual.vlm_description}"
                content_parts.append(visual_desc)
        
        return "\n\n".join(content_parts)
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.vllm_client:
            await self.vllm_client.close()