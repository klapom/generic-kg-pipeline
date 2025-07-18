#!/usr/bin/env python3
"""
Image Extraction Strategy for Visual Elements

Provides a hierarchical approach to extracting images from PDFs:
1. Docling direct extraction (if available)
2. PyMuPDF embedded images
3. PyMuPDF page rendering (fallback)
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import fitz  # PyMuPDF
from PIL import Image
import io

from ..parsers.interfaces import VisualElement, VisualElementType, DocumentType

logger = logging.getLogger(__name__)


class ImageData:
    """Container for extracted image data"""
    def __init__(self,
                 data: bytes,
                 width: int,
                 height: int,
                 source: str,
                 page_num: int,
                 bbox: Optional[List[float]] = None):
        self.data = data
        self.width = width
        self.height = height
        self.source = source  # 'embedded', 'rendered', 'docling'
        self.page_num = page_num
        self.bbox = bbox or [0, 0, width, height]
        self.content_hash = hashlib.sha256(data).hexdigest()[:16]


class ImageExtractionStrategy:
    """
    Hierarchical image extraction strategy
    
    Tries multiple methods in order:
    1. Embedded images from PDF
    2. Rendered page regions
    3. Full page rendering (fallback)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize extraction strategy
        
        Args:
            config: Configuration with:
                - min_size: Minimum image dimension (default: 100)
                - extract_embedded: Extract embedded images (default: True)
                - render_fallback: Render pages without images (default: True)
                - page_render_dpi: DPI for page rendering (default: 150)
                - render_visual_elements: Render specific regions (default: True)
        """
        self.config = config
        self.min_size = config.get('min_size', 100)
        self.extract_embedded = config.get('extract_embedded', True)
        self.render_fallback = config.get('render_fallback', True)
        self.page_render_dpi = config.get('page_render_dpi', 150)
        self.render_visual_elements = config.get('render_visual_elements', True)
        
        logger.info(f"Initialized ImageExtractionStrategy with config: {config}")
    
    def extract_images(self, pdf_path: Path, page_num: int) -> List[ImageData]:
        """
        Extract all images from a PDF page
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            
        Returns:
            List of ImageData objects
        """
        images = []
        
        try:
            # Open PDF
            doc = fitz.open(str(pdf_path))
            
            if page_num >= len(doc):
                logger.warning(f"Page {page_num} out of range for {pdf_path}")
                return images
            
            page = doc[page_num]
            
            # 1. Try embedded images first
            if self.extract_embedded:
                embedded = self._extract_embedded_images(doc, page, page_num)
                images.extend(embedded)
                logger.info(f"Extracted {len(embedded)} embedded images from page {page_num + 1}")
            
            # 2. If no images and fallback enabled, render page
            if not images and self.render_fallback:
                rendered = self._render_page_as_image(page, page_num)
                if rendered:
                    images.append(rendered)
                    logger.info(f"Rendered page {page_num + 1} as fallback image")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
        
        return images
    
    def extract_image_bytes(self,
                          pdf_path: Path,
                          page_num: int,
                          bbox: Optional[List[float]] = None) -> Optional[bytes]:
        """
        Extract image bytes for a specific region
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            bbox: Bounding box [x0, y0, x1, y1] in SmolDocling coordinates (0-500)
            
        Returns:
            Image bytes or None
        """
        try:
            doc = fitz.open(str(pdf_path))
            
            # Convert to 0-indexed
            page_idx = page_num - 1
            if page_idx >= len(doc) or page_idx < 0:
                logger.warning(f"Page {page_num} out of range")
                return None
            
            page = doc[page_idx]
            
            # If bbox provided, render specific region
            if bbox and self.render_visual_elements:
                img_data = self._render_bbox_region(page, bbox)
            else:
                # Render full page
                mat = fitz.Matrix(self.page_render_dpi / 72.0, self.page_render_dpi / 72.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
            
            doc.close()
            return img_data
            
        except Exception as e:
            logger.error(f"Error extracting image bytes: {e}")
            return None
    
    def _extract_embedded_images(self,
                               doc: fitz.Document,
                               page: fitz.Page,
                               page_num: int) -> List[ImageData]:
        """Extract embedded images from a page"""
        images = []
        
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Skip small images
                if pix.width < self.min_size or pix.height < self.min_size:
                    continue
                
                # Convert to RGB if necessary
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # Get image bytes
                img_data = pix.tobytes("png")
                
                # Create ImageData
                image = ImageData(
                    data=img_data,
                    width=pix.width,
                    height=pix.height,
                    source='embedded',
                    page_num=page_num,
                    bbox=[0, 0, pix.width, pix.height]
                )
                
                images.append(image)
                
            except Exception as e:
                logger.warning(f"Failed to extract embedded image {img_index}: {e}")
        
        return images
    
    def _render_page_as_image(self, page: fitz.Page, page_num: int) -> Optional[ImageData]:
        """Render entire page as image"""
        try:
            # Calculate matrix for desired DPI
            mat = fitz.Matrix(self.page_render_dpi / 72.0, self.page_render_dpi / 72.0)
            pix = page.get_pixmap(matrix=mat)
            
            # Get image bytes
            img_data = pix.tobytes("png")
            
            return ImageData(
                data=img_data,
                width=pix.width,
                height=pix.height,
                source='rendered',
                page_num=page_num,
                bbox=[0, 0, pix.width, pix.height]
            )
            
        except Exception as e:
            logger.error(f"Failed to render page {page_num}: {e}")
            return None
    
    def _render_bbox_region(self, page: fitz.Page, bbox: List[float]) -> Optional[bytes]:
        """
        Render a specific region of a page
        
        Args:
            page: PyMuPDF page object
            bbox: Bounding box in SmolDocling coordinates (0-500 scale)
            
        Returns:
            Image bytes or None
        """
        try:
            # Convert SmolDocling coordinates (0-500) to page coordinates
            page_rect = page.rect
            scale_x = page_rect.width / 500.0
            scale_y = page_rect.height / 500.0
            
            # Convert coordinates
            x0 = bbox[0] * scale_x
            y0 = bbox[1] * scale_y
            x1 = bbox[2] * scale_x
            y1 = bbox[3] * scale_y
            
            # Create clip rectangle
            clip_rect = fitz.Rect(x0, y0, x1, y1)
            
            # Render with clipping
            mat = fitz.Matrix(self.page_render_dpi / 72.0, self.page_render_dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            
            return pix.tobytes("png")
            
        except Exception as e:
            logger.error(f"Failed to render bbox region: {e}")
            return None
    
    def create_visual_elements_from_images(self,
                                         images: List[ImageData],
                                         source_format: DocumentType = DocumentType.PDF
                                         ) -> List[VisualElement]:
        """
        Create VisualElement objects from extracted images
        
        Args:
            images: List of ImageData objects
            source_format: Source document format
            
        Returns:
            List of VisualElement objects
        """
        visual_elements = []
        
        for img in images:
            ve = VisualElement(
                element_type=VisualElementType.IMAGE,
                source_format=source_format,
                content_hash=img.content_hash,
                page_or_slide=img.page_num + 1,  # Convert to 1-indexed
                raw_data=img.data,
                bounding_box=img.bbox,
                analysis_metadata={
                    "source": img.source,
                    "original_size": [img.width, img.height],
                    "extraction_method": "ImageExtractionStrategy"
                }
            )
            visual_elements.append(ve)
        
        return visual_elements